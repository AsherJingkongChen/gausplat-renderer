//! Tiny neural network.

pub use super::*;
pub use burn::nn::{Linear, Relu};

use burn::nn::LinearConfig;

/// The configuration for [`TinyNet`]
#[derive(Config, Copy, Debug)]
pub struct TinyNetConfig {
    /// Input dimension.
    pub dim_input: usize,
    /// Output dimension.
    pub dim_output: usize,
    /// With bias.
    #[config(default = true)]
    pub bias: bool,
}

/// Tiny neural network using two linear layers and a ReLU activation.
///
/// `W_2 * ReLU(W_1 * input + B_1) + B_2`
///
/// ## Details
///
/// The only hidden layer dimension is the sum of the input and output dimensions.
#[derive(Debug, Module)]
pub struct TinyNet<B: Backend> {
    /// The 1st linear layer.
    pub fc1: Linear<B>,
    /// The 1st ReLU activation.
    pub ac1: Relu,
    /// The 2nd linear layer.
    pub fc2: Linear<B>,
}

impl TinyNetConfig {
    /// Initialize from the configuration.
    pub fn init<B: Backend>(
        &self,
        device: &B::Device,
    ) -> TinyNet<B> {
        let dim_hidden = self.dim_input + self.dim_output;
        let bias = self.bias;
        let fc1 = LinearConfig::new(self.dim_input, dim_hidden)
            .with_bias(bias)
            .init(device);
        let ac1 = Relu::new();
        let fc2 = LinearConfig::new(dim_hidden, self.dim_output)
            .with_bias(bias)
            .init(device);
        TinyNet { fc1, ac1, fc2 }
    }
}

impl<B: Backend> TinyNet<B> {
    /// Applies the forward pass on the input tensor.
    ///
    /// ## Shapes
    ///
    /// * `input` - [`[..., dim_input]`](TinyNetConfig::dim_input)
    /// * `output` - [`[..., dim_output]`](TinyNetConfig::dim_output)
    pub fn forward<const D: usize>(
        &self,
        mut input: Tensor<B, D>,
    ) -> Tensor<B, D> {
        input = self.fc1.forward(input);
        input = self.ac1.forward(input);
        input = self.fc2.forward(input);
        input
    }
}

impl Default for TinyNetConfig {
    #[inline]
    fn default() -> Self {
        Self {
            dim_input: 1,
            dim_output: 1,
            bias: true,
        }
    }
}
