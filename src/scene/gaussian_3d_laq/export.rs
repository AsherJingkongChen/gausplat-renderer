//! 3DGS-LAQ export implementation.

pub use super::*;
pub use crate::error::Error;

use burn::record::{BinBytesRecorder, HalfPrecisionSettings, Recorder};
use miniz_oxide::deflate::compress_to_vec;

impl<B: Backend> Gaussian3dSceneLAQ<B> {
    /// Export to compressed byte stream.
    pub fn encode_compressed_bytes(&self) -> Result<Vec<u8>, Error> {
        let recorder = BinBytesRecorder::<HalfPrecisionSettings>::new();
        let bytes = recorder.record(self.to_owned().into_record(), ())?;
        let compressed_bytes = compress_to_vec(&bytes, 6);
        Ok(compressed_bytes)
    }
}
