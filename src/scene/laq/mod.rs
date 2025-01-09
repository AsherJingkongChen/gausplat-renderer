//! LAQ, or Learned Attribute Quantization,
//! is a limited representation of high precision attributes.

pub mod dequantize;
pub mod normalize;
pub mod round;
pub mod tinynet;

pub use burn::{
    config::Config,
    module::Module,
    tensor::{backend::Backend, Tensor},
};
