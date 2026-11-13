// Exact Q8_0 matrix products on an NVIDIA GPU.
//
// One output is the dot product of a token's Q16.16 activations with one row of
// Q8_0 weights, exactly as det_num::q8::dot_row computes it:
//   - within a 32-weight block the products are summed exactly,
//   - each block sum is multiplied by the block's Q.24 scale and the blocks are
//     summed in 128 bits,
//   - the total is rounded once to Q16.16 (ties to even) and saturated to i32.
// Everything is integer arithmetic, so a value does not depend on the order of
// the sums or on the device. The Rust side checks a sample of every product
// against det_num::q8::dot_row.

#include <cuda_runtime.h>

#include <cstdint>
#include <cstdlib>

namespace {

constexpr int BLOCK = 32;      // weights per Q8_0 block
constexpr int TILE_ROWS = 16;  // matrix rows per thread block
constexpr int TILE_TOKENS = 16;

// total / 2^24, rounded to nearest with ties to even, saturated to i32.
// `hi:lo` is the 128-bit two's-complement total.
__device__ __forceinline__ int round_q24(long long hi, unsigned long long lo) {
    long long floor_hi = hi >> 24;
    unsigned long long floor_lo = (lo >> 24) | ((unsigned long long)hi << 40);
    const unsigned int remainder = (unsigned int)(lo & 0xFFFFFFull);
    const bool up = remainder > 0x800000u || (remainder == 0x800000u && (floor_lo & 1ull));
    if (up) {
        floor_lo += 1ull;
        if (floor_lo == 0ull) floor_hi += 1;
    }
    if (floor_hi > 0 || (floor_hi == 0 && floor_lo > 0x7FFFFFFFull)) return 2147483647;
    if (floor_hi < -1 || (floor_hi == -1 && floor_lo < 0xFFFFFFFF80000000ull))
        return (-2147483647 - 1);
    return (int)(long long)floor_lo;
}

// Every output of one matrix against a batch of tokens. A thread block owns 16
// matrix rows and 16 tokens. For each 32-column slice it stages those rows'
// weights and those tokens' activations in shared memory, and each of its 256
// threads then forms one (row, token) block sum from the staged values.
//
// An activation is split as v = (v >> 16) * 65536 + (v & 0xFFFF). With weights
// in [-128, 127], thirty-two products of either half stay below 2^31 in
// magnitude (32 * 128 * 65535 and 32 * 128 * 32768), so both partial sums are
// exact in 32-bit arithmetic and recombine to the exact 64-bit block sum.
__global__ void q8_product(const signed char* __restrict__ weights,
                           const long long* __restrict__ scales,
                           const int* __restrict__ activations, int* __restrict__ out, int rows,
                           int cols, int tokens) {
    __shared__ signed char tile_weights[TILE_ROWS][BLOCK];
    __shared__ int tile_activations[TILE_TOKENS][BLOCK];
    const int r = threadIdx.x;
    const int t = threadIdx.y;
    const int first_row = blockIdx.x * TILE_ROWS;
    const int first_token = blockIdx.y * TILE_TOKENS;
    const int row = first_row + r;
    const int token = first_token + t;
    const bool live = row < rows && token < tokens;
    const int blocks = cols / BLOCK;
    const int thread = t * TILE_ROWS + r;
    const int threads = TILE_ROWS * TILE_TOKENS;
    long long hi = 0;
    unsigned long long lo = 0;
    for (int b = 0; b < blocks; ++b) {
        for (int k = thread; k < TILE_ROWS * BLOCK; k += threads) {
            const int tile_row = k / BLOCK, j = k % BLOCK;
            const int source_row = first_row + tile_row;
            tile_weights[tile_row][j] =
                source_row < rows ? weights[(size_t)source_row * cols + b * BLOCK + j] : 0;
        }
        for (int k = thread; k < TILE_TOKENS * BLOCK; k += threads) {
            const int tile_token = k / BLOCK, j = k % BLOCK;
            const int source_token = first_token + tile_token;
            tile_activations[tile_token][j] =
                source_token < tokens ? activations[(size_t)source_token * cols + b * BLOCK + j]
                                      : 0;
        }
        __syncthreads();
        if (live) {
            int sum_lo = 0, sum_hi = 0;
            for (int j = 0; j < BLOCK; ++j) {
                const int q = tile_weights[r][j];
                const int v = tile_activations[t][j];
                sum_lo += q * (v & 0xFFFF);
                sum_hi += q * (v >> 16);
            }
            const long long block_sum = ((long long)sum_hi << 16) + (long long)sum_lo;
            const long long scale = scales[(size_t)row * blocks + b];
            const unsigned long long product_lo =
                (unsigned long long)block_sum * (unsigned long long)scale;
            lo += product_lo;
            hi += __mul64hi(block_sum, scale) + (lo < product_lo ? 1 : 0);
        }
        __syncthreads();
    }
    if (live) out[(size_t)token * rows + row] = round_q24(hi, lo);
}

// Device buffers for one product's activations and outputs, grown as needed
// and reused. The Rust side runs one product at a time.
int* scratch_activations = nullptr;
int* scratch_out = nullptr;
size_t scratch_activation_bytes = 0;
size_t scratch_out_bytes = 0;

cudaError_t reserve(int** buffer, size_t* capacity, size_t bytes) {
    if (bytes <= *capacity) return cudaSuccess;
    cudaFree(*buffer);
    *buffer = nullptr;
    *capacity = 0;
    const cudaError_t status = cudaMalloc((void**)buffer, bytes);
    if (status == cudaSuccess) *capacity = bytes;
    return status;
}

}  // namespace

struct Q8GpuMatrix {
    signed char* weights;
    long long* scales;
    uint32_t rows;
    uint32_t cols;
};

extern "C" {

const char* q8gpu_error(int code) { return cudaGetErrorString((cudaError_t)code); }

// Creates the device context. Fails when there is no usable device.
int q8gpu_init() {
    void* probe = nullptr;
    const cudaError_t status = cudaMalloc(&probe, 1 << 20);
    if (status != cudaSuccess) return (int)status;
    return (int)cudaFree(probe);
}

void q8gpu_free(Q8GpuMatrix* matrix) {
    if (!matrix) return;
    cudaFree(matrix->weights);
    cudaFree(matrix->scales);
    std::free(matrix);
}

// `weights` is rows * cols bytes, row by row; `scales` is one Q.24 scale per
// 32-weight block, rows * (cols / 32) values.
int q8gpu_upload(const signed char* weights, const long long* scales, uint32_t rows, uint32_t cols,
                 Q8GpuMatrix** out_matrix) {
    *out_matrix = nullptr;
    Q8GpuMatrix* matrix = (Q8GpuMatrix*)std::calloc(1, sizeof(Q8GpuMatrix));
    if (!matrix) return (int)cudaErrorMemoryAllocation;
    matrix->rows = rows;
    matrix->cols = cols;
    const size_t weight_bytes = (size_t)rows * cols;
    const size_t scale_bytes = (size_t)rows * (cols / BLOCK) * sizeof(long long);
    cudaError_t status = cudaMalloc((void**)&matrix->weights, weight_bytes);
    if (status == cudaSuccess) status = cudaMalloc((void**)&matrix->scales, scale_bytes);
    if (status == cudaSuccess)
        status = cudaMemcpy(matrix->weights, weights, weight_bytes, cudaMemcpyHostToDevice);
    if (status == cudaSuccess)
        status = cudaMemcpy(matrix->scales, scales, scale_bytes, cudaMemcpyHostToDevice);
    if (status != cudaSuccess) {
        q8gpu_free(matrix);
        return (int)status;
    }
    *out_matrix = matrix;
    return (int)cudaSuccess;
}

// out[token * rows + row] = the Q8_0 product of matrix row `row` with token
// `token`'s activations; `activations` is tokens * cols values, token by token.
int q8gpu_product(const Q8GpuMatrix* matrix, const int* activations, uint32_t tokens, int* out) {
    const size_t activation_bytes = (size_t)tokens * matrix->cols * sizeof(int);
    const size_t out_bytes = (size_t)tokens * matrix->rows * sizeof(int);
    cudaError_t status = reserve(&scratch_activations, &scratch_activation_bytes, activation_bytes);
    if (status == cudaSuccess) status = reserve(&scratch_out, &scratch_out_bytes, out_bytes);
    if (status == cudaSuccess)
        status = cudaMemcpy(scratch_activations, activations, activation_bytes,
                            cudaMemcpyHostToDevice);
    if (status != cudaSuccess) return (int)status;

    const dim3 block(TILE_ROWS, TILE_TOKENS);
    const dim3 grid((matrix->rows + TILE_ROWS - 1) / TILE_ROWS,
                    (tokens + TILE_TOKENS - 1) / TILE_TOKENS);
    q8_product<<<grid, block>>>(matrix->weights, matrix->scales, scratch_activations, scratch_out,
                                (int)matrix->rows, (int)matrix->cols, (int)tokens);
    status = cudaGetLastError();
    if (status == cudaSuccess) status = cudaDeviceSynchronize();
    if (status == cudaSuccess)
        status = cudaMemcpy(out, scratch_out, out_bytes, cudaMemcpyDeviceToHost);
    return (int)status;
}

}  // extern "C"
