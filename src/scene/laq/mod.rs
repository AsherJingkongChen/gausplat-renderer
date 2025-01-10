//! LAQ, or Learned Attribute Quantization,
//! is a limited representation of high precision attributes.

pub mod normalize;
pub mod quantize;
pub mod round;
pub mod tinynet;

pub use burn::{
    config::Config,
    module::Module,
    tensor::{
        backend::{AutodiffBackend, Backend},
        Tensor,
    },
};

pub use quantize::{Dequantizer, DequantizerConfig};
