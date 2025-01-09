//! Rounding.

pub use super::*;

/// STE, Straight-through estimator for quantization.
///
/// `output = round(input), d output = d input`
#[derive(Clone, Copy, Debug, Default, Module)]
pub struct StraightThroughEstimator;

impl StraightThroughEstimator {
    /// Initialize the estimator.
    #[inline]
    pub const fn init() -> Self {
        Self
    }

    /// Round down the input tensor to the nearest integer.
    pub fn forward<B: Backend, const D: usize>(
        &self,
        input: Tensor<B, D>,
    ) -> Tensor<B, D> {
        (input.to_owned().floor() - input.to_owned()).detach() + input
    }
}
