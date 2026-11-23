//! Celestia as the data-availability layer: blobs posted to and read from a
//! celestia-node's blob API over JSON-RPC.
//!
//! The node is the local one of `devstack/` (or any node whose key can pay
//! for blobs). A blob is named by the block it was included in, its namespace
//! and Celestia's own commitment to its bytes, which iris computes itself
//! (`inference_artifacts::celestia`) before it posts and checks on what it
//! reads back.
//!
//! What is still the host's word here: that a block's data root is one
//! Blobstream attested to, and the availability proof. `included_at` asks the
//! node, which stands in for a contract's view of the chain, and `proves`
//! reads the blob as the stand-in layer does.

use std::io::{Read, Write};
use std::net::TcpStream;
use std::time::Duration;

use anyhow::{bail, ensure, Context, Result};
use inference_artifacts::availability::{
    AvailabilityLayer, AvailabilityProof, BlobPointer, PostedBlob, NAMESPACE_BYTES,
};
use inference_artifacts::celestia::blob_commitment;
use serde_json::{json, Value};

use crate::availability::{contents_of, BlockRoot, PostingLayer};

/// The node's JSON-RPC address, as `http://host:port`. When set, iris posts
/// to and reads from that node instead of the stand-in directory.
pub const CELESTIA_RPC_ENV: &str = "IRIS_CELESTIA_RPC";

/// A celestia-node's blob API.
pub struct CelestiaLayer {
    /// `host:port`.
    address: String,
}

impl CelestiaLayer {
    /// A node at `http://host:port`.
    pub fn open(url: &str) -> Result<Self> {
        let address = url
            .strip_prefix("http://")
            .with_context(|| format!("{url} is not an http:// address"))?
            .trim_end_matches('/');
        ensure!(
            !address.is_empty() && !address.contains('/'),
            "{url} is not http://host:port"
        );
        Ok(Self {
            address: address.to_string(),
        })
    }

    /// One JSON-RPC call, and its `result`.
    fn call(&self, method: &str, params: Value) -> Result<Value> {
        let body = json!({"jsonrpc": "2.0", "id": 1, "method": method, "params": params});
        let response: Value = serde_json::from_slice(&http_post(&self.address, &body.to_string())?)
            .with_context(|| format!("{method}: the node's answer is not JSON"))?;
        if let Some(error) = response.get("error") {
            bail!("{method}: {error}");
        }
        response
            .get("result")
            .cloned()
            .with_context(|| format!("{method}: the node's answer has no result"))
    }

    fn namespace(pointer: &BlobPointer) -> Result<[u8; NAMESPACE_BYTES]> {
        hex::decode(&pointer.namespace)
            .ok()
            .and_then(|bytes| bytes.try_into().ok())
            .with_context(|| format!("{} is not a namespace", pointer.namespace))
    }

    /// The bytes of the blob a pointer names, when the node holds a blob with
    /// that commitment in that block and the bytes are what the commitment
    /// is of.
    fn included(&self, pointer: &BlobPointer) -> Option<Vec<u8>> {
        let bytes = self.read(pointer).ok()?;
        let namespace = Self::namespace(pointer).ok()?;
        (hex::encode(blob_commitment(&namespace, &bytes)) == pointer.commitment).then_some(bytes)
    }
}

impl PostingLayer for CelestiaLayer {
    fn post(
        &self,
        namespace: &[u8; NAMESPACE_BYTES],
        blobs: &[Vec<u8>],
    ) -> Result<Vec<BlobPointer>> {
        let commitments: Vec<[u8; 32]> = blobs
            .iter()
            .map(|bytes| blob_commitment(namespace, bytes))
            .collect();
        let posted: Vec<Value> = blobs
            .iter()
            .zip(&commitments)
            .map(|(bytes, commitment)| {
                json!({
                    "namespace": base64(namespace),
                    "data": base64(bytes),
                    "share_version": 0,
                    "commitment": base64(commitment),
                })
            })
            .collect();
        // The node signs with one key, so two posts at once can collide on
        // its sequence number: a failed post is tried again.
        let mut last = None;
        for attempt in 0..4 {
            if attempt > 0 {
                std::thread::sleep(Duration::from_secs(2));
            }
            match self.call("blob.Submit", json!([posted, {}])) {
                Ok(height) => {
                    let height = height
                        .as_u64()
                        .context("blob.Submit: the node gave no height")?;
                    return Ok(commitments
                        .iter()
                        .map(|commitment| BlobPointer {
                            height,
                            namespace: hex::encode(namespace),
                            commitment: hex::encode(commitment),
                        })
                        .collect());
                }
                Err(error) => last = Some(error),
            }
        }
        Err(last.expect("at least one attempt was made"))
    }

    fn read(&self, pointer: &BlobPointer) -> Result<Vec<u8>> {
        let namespace = Self::namespace(pointer)?;
        let commitment = hex::decode(&pointer.commitment)
            .with_context(|| format!("{} is not a commitment", pointer.commitment))?;
        let blob = self.call(
            "blob.Get",
            json!([pointer.height, base64(&namespace), base64(&commitment)]),
        )?;
        unbase64(
            blob.get("data")
                .and_then(Value::as_str)
                .context("blob.Get: the blob has no data")?,
        )
        .context("blob.Get: the blob's data is not base64")
    }

    fn block_root(&self, height: u64) -> Result<BlockRoot> {
        self.header_root(height)
    }
}

impl CelestiaLayer {
    /// The root and time of the block at `height`, as the node's header has
    /// them.
    fn header_root(&self, height: u64) -> Result<BlockRoot> {
        let header = self.call("header.GetByHeight", json!([height]))?;
        let field = |name: &str| {
            header
                .pointer(&format!("/header/{name}"))
                .and_then(Value::as_str)
                .with_context(|| format!("block {height}'s header has no {name}"))
        };
        Ok(BlockRoot {
            height,
            data_root: field("data_hash")?.to_ascii_lowercase(),
            time: unix_seconds(field("time")?)
                .with_context(|| format!("block {height}'s time is not a timestamp"))?,
        })
    }
}

impl AvailabilityLayer for CelestiaLayer {
    fn included_at(&self, pointer: &BlobPointer, length: u64) -> Option<u64> {
        let bytes = self.included(pointer)?;
        (bytes.len() as u64 == length)
            .then(|| self.block_root(pointer.height).ok())
            .flatten()
            .map(|block| block.time)
    }

    fn proves(&self, blob: &PostedBlob) -> bool {
        match &blob.proof {
            AvailabilityProof::HostChecked { outputs, .. } => {
                self.included(&blob.pointer).is_some_and(|bytes| {
                    contents_of(&bytes, outputs).ok().as_ref() == Some(&blob.contents)
                })
            }
        }
    }
}

/// POST a JSON body to `host:port` and return the response's body. Plain
/// HTTP/1.1 to a local node: no TLS, no redirects, one request a connection.
pub(crate) fn http_post(address: &str, body: &str) -> Result<Vec<u8>> {
    let mut stream = TcpStream::connect(address)
        .with_context(|| format!("failed to reach the node at {address}"))?;
    stream.set_read_timeout(Some(Duration::from_secs(120)))?;
    write!(
        stream,
        "POST / HTTP/1.1\r\nHost: {address}\r\nContent-Type: application/json\r\n\
         Content-Length: {}\r\nConnection: close\r\n\r\n{body}",
        body.len()
    )?;
    let mut response = Vec::new();
    stream
        .read_to_end(&mut response)
        .with_context(|| format!("the node at {address} did not answer"))?;
    http_body(&response)
}

/// The body of an HTTP/1.1 response read to its end, with chunked transfer
/// encoding undone.
fn http_body(response: &[u8]) -> Result<Vec<u8>> {
    let split = response
        .windows(4)
        .position(|window| window == b"\r\n\r\n")
        .context("the response has no end of headers")?;
    let head = String::from_utf8_lossy(&response[..split]);
    let mut lines = head.lines();
    let status = lines.next().unwrap_or_default();
    ensure!(
        status.split_whitespace().nth(1) == Some("200"),
        "the node answered `{status}`"
    );
    let chunked = lines.any(|line| {
        let line = line.to_ascii_lowercase();
        line.starts_with("transfer-encoding:") && line.contains("chunked")
    });
    let mut body = &response[split + 4..];
    if !chunked {
        return Ok(body.to_vec());
    }
    let mut whole = Vec::new();
    loop {
        let line_end = body
            .windows(2)
            .position(|window| window == b"\r\n")
            .context("a chunk has no size line")?;
        let size_text = std::str::from_utf8(&body[..line_end])?;
        let size =
            usize::from_str_radix(size_text.split(';').next().unwrap_or_default().trim(), 16)
                .with_context(|| format!("`{size_text}` is not a chunk size"))?;
        body = &body[line_end + 2..];
        if size == 0 {
            return Ok(whole);
        }
        ensure!(body.len() >= size + 2, "a chunk is cut short");
        whole.extend_from_slice(&body[..size]);
        body = &body[size + 2..];
    }
}

const BASE64: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";

/// Standard base64 with padding, as the node's JSON carries bytes.
fn base64(bytes: &[u8]) -> String {
    let mut text = String::with_capacity(bytes.len().div_ceil(3) * 4);
    for chunk in bytes.chunks(3) {
        let group = (u32::from(chunk[0]) << 16)
            | (u32::from(*chunk.get(1).unwrap_or(&0)) << 8)
            | u32::from(*chunk.get(2).unwrap_or(&0));
        for (position, shift) in [18, 12, 6, 0].into_iter().enumerate() {
            text.push(if position <= chunk.len() {
                BASE64[(group >> shift) as usize & 63] as char
            } else {
                '='
            });
        }
    }
    text
}

fn unbase64(text: &str) -> Result<Vec<u8>> {
    let digits = text.trim_end_matches('=');
    ensure!(
        text.len() % 4 == 0 && text.len() - digits.len() <= 2,
        "not padded base64"
    );
    let mut bytes = Vec::with_capacity(text.len() / 4 * 3);
    let (mut group, mut bits) = (0u32, 0);
    for digit in digits.bytes() {
        let value = BASE64
            .iter()
            .position(|known| *known == digit)
            .context("not a base64 digit")?;
        group = (group << 6) | value as u32;
        bits += 6;
        if bits >= 8 {
            bits -= 8;
            bytes.push((group >> bits) as u8);
        }
    }
    Ok(bytes)
}

/// Unix seconds of an RFC 3339 time in UTC, as a block header states it:
/// `2026-11-19T00:53:42.177867124Z`. The fraction is dropped.
fn unix_seconds(time: &str) -> Option<u64> {
    let time = time.strip_suffix('Z')?;
    let (date, clock) = time.split_once('T')?;
    let mut date = date.split('-').map(|part| part.parse::<i64>().ok());
    let (year, month, day) = (date.next()??, date.next()??, date.next()??);
    let clock = clock.split('.').next()?;
    let mut clock = clock.split(':').map(|part| part.parse::<i64>().ok());
    let (hour, minute, second) = (clock.next()??, clock.next()??, clock.next()??);
    // Days since 1970-01-01, by the usual civil-calendar count.
    let year = if month <= 2 { year - 1 } else { year };
    let era = year.div_euclid(400);
    let year_of_era = year.rem_euclid(400);
    let day_of_year = (153 * (if month > 2 { month - 3 } else { month + 9 }) + 2) / 5 + day - 1;
    let day_of_era = year_of_era * 365 + year_of_era / 4 - year_of_era / 100 + day_of_year;
    let days = era * 146_097 + day_of_era - 719_468;
    u64::try_from(days * 86_400 + hour * 3_600 + minute * 60 + second).ok()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bytes_cross_the_nodes_json_as_padded_base64() {
        for (bytes, text) in [
            (&b""[..], ""),
            (b"f", "Zg=="),
            (b"fo", "Zm8="),
            (b"foo", "Zm9v"),
            (b"hello iris", "aGVsbG8gaXJpcw=="),
        ] {
            assert_eq!(base64(bytes), text);
            assert_eq!(unbase64(text).unwrap(), bytes);
        }
        let all: Vec<u8> = (0..=255).collect();
        assert_eq!(unbase64(&base64(&all)).unwrap(), all);
        assert!(unbase64("Zg=").is_err());
        assert!(unbase64("Z!==").is_err());
    }

    #[test]
    fn a_headers_time_is_read_as_unix_seconds() {
        assert_eq!(unix_seconds("1970-01-01T00:00:00Z"), Some(0));
        assert_eq!(unix_seconds("2000-03-01T00:00:00Z"), Some(951_868_800));
        assert_eq!(
            unix_seconds("2026-11-19T00:53:42.177867124Z"),
            Some(1_795_049_622)
        );
        assert_eq!(unix_seconds("2026-11-19 00:53:42"), None);
    }

    #[test]
    fn a_response_is_read_whole_or_in_chunks() {
        let whole = b"HTTP/1.1 200 OK\r\nContent-Length: 2\r\n\r\n{}";
        assert_eq!(http_body(whole).unwrap(), b"{}");
        let chunked =
            b"HTTP/1.1 200 OK\r\nTransfer-Encoding: chunked\r\n\r\n3\r\n{\"a\r\n4\r\n\":1}\r\n0\r\n\r\n";
        assert_eq!(http_body(chunked).unwrap(), b"{\"a\":1}");
        assert!(http_body(b"HTTP/1.1 503 Service Unavailable\r\n\r\n").is_err());
        assert!(CelestiaLayer::open("https://example").is_err());
        assert_eq!(
            CelestiaLayer::open("http://127.0.0.1:26658/")
                .unwrap()
                .address,
            "127.0.0.1:26658"
        );
    }

    /// Against a running node: `IRIS_CELESTIA_RPC=http://127.0.0.1:26658`,
    /// with the stack of `devstack/` up.
    #[test]
    #[ignore = "needs the local Celestia node of devstack/"]
    fn a_blob_is_posted_read_back_and_found_in_its_block() {
        let layer = CelestiaLayer::open(&std::env::var(CELESTIA_RPC_ENV).unwrap()).unwrap();
        let blobs = vec![b"a blob of iris's".to_vec(), vec![7u8; 5_000]];
        let pointers = layer
            .post(&crate::availability::CLAIM_NAMESPACE, &blobs)
            .unwrap();
        assert_eq!(pointers.len(), 2);
        for (pointer, bytes) in pointers.iter().zip(&blobs) {
            assert_eq!(&layer.read(pointer).unwrap(), bytes);
            let time = layer.included_at(pointer, bytes.len() as u64).unwrap();
            assert_eq!(time, layer.block_root(pointer.height).unwrap().time);
            // Another length, block or commitment is not this blob.
            assert!(layer.included_at(pointer, bytes.len() as u64 + 1).is_none());
            let mut elsewhere = pointer.clone();
            elsewhere.height += 1;
            assert!(layer.included_at(&elsewhere, bytes.len() as u64).is_none());
        }
        let root = layer.block_root(pointers[0].height).unwrap();
        assert_eq!(root.data_root.len(), 64);
        assert!(root.time > 1_700_000_000);
    }
}
