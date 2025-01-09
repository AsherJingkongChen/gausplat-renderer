//! De-quantization.

pub use super::*;
pub use normalize::{MinMaxNorm, MinMaxNormConfig};
pub use round::StraightThroughEstimator;
pub use tinynet::{TinyNet, TinyNetConfig};

/// The configuration for [`Dequantizer`].
#[derive(Config, Copy, Debug)]
pub struct DequantizerConfig {
    /// Network for de-quantization.
    pub net: TinyNetConfig,
}

/// De-quantize the quantized input tensor.
#[derive(Debug, Module)]
pub struct Dequantizer<B: Backend> {
    /// Normalization for pre-quantization.
    pub norm_prequant: MinMaxNorm,
    /// Rounding for pre-quantization.
    pub round_prequant: StraightThroughEstimator,
    /// Normalization for de-quantization.
    pub norm_dequant: MinMaxNorm,
    /// Network for de-quantization.
    pub net_dequant: TinyNet<B>,
}

impl DequantizerConfig {
    /// Initialize from the configuration.
    pub fn init<B: Backend>(
        &self,
        device: &B::Device,
    ) -> Dequantizer<B> {
        let norm_prequant = MinMaxNormConfig::default().with_range_max(255.999).init();
        let round_prequant = StraightThroughEstimator::init();
        let norm_dequant = MinMaxNormConfig::default().init();
        let net_dequant = self.net.init(device);
        Dequantizer {
            norm_prequant,
            round_prequant,
            norm_dequant,
            net_dequant,
        }
    }
}

impl<B: Backend> Dequantizer<B> {
    /// Pre-quantize any input tensor.
    pub fn prequantize<const D: usize>(
        &self,
        mut input: Tensor<B, D>,
    ) -> Tensor<B, D> {
        input = self.norm_prequant.forward(input);
        input = self.round_prequant.forward(input);
        input
    }

    /// De-quantize the quantized input tensor.
    pub fn dequantize<const D: usize>(
        &self,
        mut input: Tensor<B, D>,
    ) -> Tensor<B, D> {
        input = self.norm_dequant.forward(input);
        input = self.net_dequant.forward(input);
        input
    }

    /// Applies the forward pass on the input tensor.
    ///
    /// If the input tensor is not quantized, it will be pre-quantized.
    ///
    /// ## Shapes
    ///
    /// * `input` - [`[..., dim_input]`](DequantizerConfig::net)
    /// * `output` - [`[..., dim_output]`](DequantizerConfig::net)
    ///
    /// ## Details
    ///
    /// The process is completely differentiable.
    pub fn forward<const D: usize>(
        &self,
        mut input: Tensor<B, D>,
        is_quantized: bool,
    ) -> Tensor<B, D> {
        if !is_quantized {
            input = self.prequantize(input);
        }
        input = self.dequantize(input);
        input
    }
}

impl Default for DequantizerConfig {
    #[inline]
    fn default() -> Self {
        let net = TinyNetConfig::default();
        Self { net }
    }
}
