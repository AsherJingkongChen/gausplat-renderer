pub mod rasterizer;
pub mod spherical_harmonics;

use burn::tensor::activation;
pub use burn::{
    backend,
    module::Module,
    tensor::{self, backend::*, Tensor},
};

#[derive(Debug, Module)]
pub struct Gaussian3dScene<B: Backend> {
    /// `[P, (D + 1) ^ 2, 3]`
    ///
    /// The colors represented as orthonormalized spherical harmonics.
    pub colors_sh: Tensor<B, 3>,

    /// `[P, 1]`
    pub opacities: Tensor<B, 2>,

    /// `[P, 3]`
    pub positions: Tensor<B, 2>,

    /// `[P, 4]`
    pub rotations: Tensor<B, 2>,

    /// `[P, 3]`
    pub scalings: Tensor<B, 2>,
}

impl<B: Backend> Gaussian3dScene<B> {
    pub fn colors_sh(&self) -> Tensor<B, 3> {
        self.colors_sh.to_owned()
    }

    pub fn opacities(&self) -> Tensor<B, 2> {
        activation::sigmoid(self.opacities.to_owned())
    }

    pub fn positions(&self) -> Tensor<B, 2> {
        self.positions.to_owned()
    }

    pub fn rotations(&self) -> Tensor<B, 2> {
        self.rotations.to_owned().exp()
    }

    pub fn scalings(&self) -> Tensor<B, 2> {
        let norm = (self.scalings.to_owned() * self.scalings.to_owned())
            .sum_dim(1)
            .sqrt();
        self.scalings.to_owned().div(norm)
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn gaussian_3d_scalings_normalized() {
        use super::*;

        type Backend = backend::NdArray;
        let device = Default::default();

        let scene = Gaussian3dScene::<Backend> {
            colors_sh: Tensor::empty([1, 1, 3], &device),
            opacities: Tensor::empty([1, 1], &device),
            positions: Tensor::empty([1, 3], &device),
            rotations: Tensor::empty([1, 4], &device),
            scalings: Tensor::from_floats(
                [[1.0, -2.0, 3.0], [0.3, 0.0, -0.1]],
                &device,
            ),
        };
        let scalings = scene.scalings();
        let scalings_expected = Tensor::from_floats(
            [
                [
                    1.0 / 3.7416573867739413,
                    -2.0 / 3.7416573867739413,
                    3.0 / 3.7416573867739413,
                ],
                [
                    0.3 / 0.31622776601683794,
                    0.0 / 0.31622776601683794,
                    -0.1 / 0.31622776601683794,
                ],
            ],
            &device,
        );
        assert!(
            scalings
                .to_owned()
                .equal(scalings_expected.to_owned())
                .all()
                .into_scalar(),
            "{:?} != {:?}",
            scalings,
            scalings_expected
        );
    }
}
