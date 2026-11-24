use heavy_window_fixture::*;
use raster::prelude::*;
#[sequence]
fn main(source: Values) -> u64 {
    let values = select!(List<u64>, source.values);
    let result = call_recur!(tile = heavy, input = values, state = 0u64, args = ());
    call!(finish, result)
}
