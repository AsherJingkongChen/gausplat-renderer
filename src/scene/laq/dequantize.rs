//! De-quantization.

pub use super::*;
pub use normalize::{MinMaxNorm, MinMaxNormConfig};
pub use round::StraightThroughEstimator;
pub use tinynet::{TinyNet, TinyNetConfig};

use burn::{
    nn::loss::{HuberLossConfig, Reduction},
    optim::{AdamConfig, GradientsParams, Optimizer},
    tensor::ElementConversion,
};
use kdam::{tqdm, BarExt};

/// The configuration for [`Dequantizer`].
#[derive(Config, Copy, Debug)]
pub struct DequantizerConfig {
    /// Network for de-quantization.
    pub net: TinyNetConfig,
}

/// De-quantize the quantized input attributes.
///
/// 1. `a_h = dequant(prequant(q_h))`
/// 2. `a_h = dequant(q)`
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
    /// Pre-quantize any input attributes.
    pub fn prequantize<const D: usize>(
        &self,
        mut input: Tensor<B, D>,
    ) -> Tensor<B, D> {
        input = self.norm_prequant.forward(input);
        input = self.round_prequant.forward(input);
        input
    }

    /// De-quantize the quantized input attributes.
    pub fn dequantize<const D: usize>(
        &self,
        mut input: Tensor<B, D>,
    ) -> Tensor<B, D> {
        input = self.norm_dequant.forward(input);
        input = self.net_dequant.forward(input);
        input
    }

    /// Applies the forward pass on the input attributes.
    ///
    /// If the input attributes is not quantized, it will be pre-quantized.
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

impl<AB: AutodiffBackend> Dequantizer<AB> {
    /// Fit the de-quantizer to the quantized input attributes.
    ///
    /// 1. `a ~ a_h = dequant(prequant(q_h))`
    /// 2. `q_h = a`
    pub fn fit<const D: usize>(
        attributes: Tensor<AB, D>,
        learning_rate: f64,
        loss_threshold: f32,
    ) -> Self {
        const ITERATION_COUNT: usize = 1000;

        let attributes = attributes.to_owned().set_require_grad(false);
        let quants_h = attributes.to_owned().set_require_grad(false);

        let dim_input = attributes.dims()[D - 1];
        let dim_output = quants_h.dims()[D - 1];
        let device = &attributes.device();
        let mut dequantizer =
            DequantizerConfig::new(TinyNetConfig::new(dim_input, dim_output))
                .init(device);

        let metric_reconstruction = HuberLossConfig::new(1.0).init();
        let mut optimizer_dequantizer = AdamConfig::new().init::<AB, Dequantizer<AB>>();
        let mut bar = tqdm!(total = ITERATION_COUNT); //

        for iteration in 0..ITERATION_COUNT {
            let attributes_h = dequantizer.dequantize(quants_h.to_owned());
            let loss = metric_reconstruction.forward(
                attributes_h,
                attributes.to_owned(),
                Reduction::Auto,
            );
            let grads = GradientsParams::from_grads(loss.backward(), &dequantizer);
            dequantizer = optimizer_dequantizer.step(learning_rate, dequantizer, grads);

            bar.update(1).unwrap(); //

            if (iteration + 1) % 10 == 0 {
                let loss = loss.into_scalar().elem::<f32>();
                bar.set_postfix(format!("loss: {:?}", loss)); //
                bar.refresh().unwrap(); //
                if loss < loss_threshold {
                    bar.refresh().unwrap(); //
                    break;
                }
            }
        }

        dequantizer
    }
}

impl Default for DequantizerConfig {
    #[inline]
    fn default() -> Self {
        let net = TinyNetConfig::default();
        Self { net }
    }
}
