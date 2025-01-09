//! Normalization.

pub use super::*;

/// The configuration for [`MinMaxNorm`].
#[derive(Config, Copy, Debug)]
pub struct MinMaxNormConfig {
    /// Maximum value.
    #[config(default = 1.0)]
    pub range_max: f32,
    /// Minimum value.
    #[config(default = 0.0)]
    pub range_min: f32,
}

/// Min-max normalize the input tensor uniformly.
///
/// `(input - min(input)) / (max(input) - min(input)) * (range_max - range_min) + range_min`
#[derive(Clone, Debug, Module)]
pub struct MinMaxNorm {
    /// Multiplier.
    ///
    /// `range_max - range_min`
    pub scale: f32,
    /// Offset.
    ///
    /// `range_min`
    pub shift: f32,
}

impl MinMaxNormConfig {
    /// Initialize from the configuration.
    pub fn init(&self) -> MinMaxNorm {
        MinMaxNorm {
            scale: self.range_max - self.range_min,
            shift: self.range_min,
        }
    }
}

impl MinMaxNorm {
    /// Normalize the input tensor.
    pub fn forward<B: Backend, const D: usize>(
        &self,
        input: Tensor<B, D>,
    ) -> Tensor<B, D> {
        let dim_end = input.shape().num_dims().saturating_sub(1);
        let shape = input.shape();
        let input = input.flatten(0, dim_end);
        let input_max = input.to_owned().max();
        let input_min = input.to_owned().min();
        let offset = input_min.to_owned();
        let scale = (input_max - input_min + f32::EPSILON)
            .recip()
            .mul_scalar(self.scale);
        let output = (input - offset).mul(scale).add_scalar(self.shift);
        output.reshape(shape)
    }
}

impl Default for MinMaxNormConfig {
    #[inline]
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn forward() {
        use super::*;
        use burn::backend::NdArray;

        type B = NdArray<f32>;
        let device = &Default::default();

        let norm = MinMaxNormConfig::default().init();
        let input = Tensor::<B, 2>::zeros([5, 3], device);
        let output = norm.forward(input);
        output
            .into_data()
            .assert_eq(&Tensor::<B, 2>::zeros([5, 3], device).into_data(), true);

        let norm = MinMaxNormConfig::default().init();
        let input =
            Tensor::<B, 2>::from_data([[0.0, 2.0, 4.0], [8.0, 4.0, 16.0]], device);
        let output = norm.forward(input);
        output.into_data().assert_eq(
            &Tensor::<B, 2>::from_data([[0.0, 0.125, 0.25], [0.5, 0.25, 1.0]], device)
                .into_data(),
            true,
        );

        let norm = MinMaxNormConfig::default().with_range_max(255.0).init();
        let input =
            Tensor::<B, 2>::from_data([[0.0, 0.1], [0.6, 0.4], [0.5, 1.0]], device);
        let output = norm.forward(input);
        output.into_data().assert_approx_eq_diff(
            &Tensor::<B, 2>::from_data(
                [[0.0, 25.5], [153.0, 102.0], [127.5, 255.0]],
                device,
            )
            .into_data(),
            1e-6,
        );
    }
}
