use super::*;
use burn::tensor::activation;

impl<B: Backend> Gaussian3dScene<B> {
    /// The colors represented as orthonormalized spherical harmonics
    ///
    /// `[P, 16, 3]`
    pub fn colors_sh(&self) -> Tensor<B, 3> {
        debug_assert_eq!(self.colors_sh.val().dims()[1], 16, "colors_sh.dims()[1] != 16");
        debug_assert_eq!(self.colors_sh.val().dims()[2], 3, "colors_sh.dims()[2] != 3");

        self.colors_sh.val()
    }

    /// The colors represented as orthonormalized spherical harmonics
    ///
    /// `[P, 16, 3]`
    pub fn make_colors_sh(
        colors_sh: Tensor<B, 3>,
    ) -> Tensor<B, 3> {
        debug_assert_eq!(colors_sh.dims()[1], 16, "colors_sh.dims()[1] != 16");
        debug_assert_eq!(colors_sh.dims()[2], 3, "colors_sh.dims()[2] != 3");

        colors_sh
    }

    /// The output is applied to the sigmoid function
    ///
    /// `[P, 1]`
    pub fn opacities(&self) -> Tensor<B, 2> {
        debug_assert_eq!(self.opacities.val().dims()[1], 1, "opacities.dims()[1] != 1");

        activation::sigmoid(self.opacities.val())
    }

    /// The input `opacities` is applied to the inverse sigmoid function
    ///
    /// `[P, 1]`
    pub fn make_opacities(opacities: Tensor<B, 2>) -> Tensor<B, 2> {
        debug_assert_eq!(opacities.dims()[1], 1, "opacities.dims()[1] != 1");

        opacities
            .to_owned()
            .div(opacities.to_owned().neg().add_scalar(1.0))
            .log()
    }

    /// `[P, 3]`
    pub fn positions(&self) -> Tensor<B, 2> {
        debug_assert_eq!(self.positions.val().dims()[1], 3, "positions.dims()[1] != 3");

        self.positions.val()
    }

    /// `[P, 3]`
    pub fn make_positions(positions: Tensor<B, 2>) -> Tensor<B, 2> {
        debug_assert_eq!(positions.dims()[1], 3, "positions.dims()[1] != 3");

        positions
    }

    /// The output is normalized
    ///
    /// `[P, 4]`
    pub fn rotations(&self) -> Tensor<B, 2> {
        debug_assert_eq!(self.rotations.val().dims()[1], 4, "rotations.dims()[1] != 4");

        let rotations = self.rotations.val();
        let norms = rotations.to_owned().mul(rotations).sum_dim(1).sqrt();
        self.rotations.val().div(norms)
    }

    /// `[P, 4]`
    pub fn make_rotations(rotations: Tensor<B, 2>) -> Tensor<B, 2> {
        debug_assert_eq!(rotations.dims()[1], 4, "rotations.dims()[1] != 4");

        rotations
    }

    /// The output is applied to the exponential function
    ///
    /// `[P, 3]`
    pub fn scalings(&self) -> Tensor<B, 2> {
        debug_assert_eq!(self.scalings.val().dims()[1], 3, "scalings.dims()[1] != 3");

        self.scalings.val().exp()
    }

    /// The input `scalings` is applied to the logarithm function
    ///
    /// `[P, 3]`
    pub fn make_scalings(scalings: Tensor<B, 2>) -> Tensor<B, 2> {
        debug_assert_eq!(scalings.dims()[1], 3, "scalings.dims()[1] != 3");

        scalings.log()
    }
}
