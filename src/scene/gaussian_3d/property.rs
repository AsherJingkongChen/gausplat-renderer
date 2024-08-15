use super::*;
use burn::tensor::activation;

impl<B: backend::Backend> Gaussian3dScene<B> {
    /// `[P, 16, 3]`
    ///
    /// The colors represented as orthonormalized spherical harmonics
    pub fn colors_sh(&self) -> Tensor<B, 3> {
        self.colors_sh.to_owned()
    }

    /// `[P, 16, 3]`
    ///
    /// The colors represented as orthonormalized spherical harmonics
    pub fn set_colors_sh(
        &mut self,
        colors_sh: Tensor<B, 3>,
    ) -> &mut Self {
        debug_assert_eq!(colors_sh.dims()[1], 16, "colors_sh.dims()[1] != 16");
        debug_assert_eq!(colors_sh.dims()[2], 3, "colors_sh.dims()[2] != 3");
        self.colors_sh = colors_sh;
        self
    }

    /// `[P, 1]`
    pub fn opacities(&self) -> Tensor<B, 2> {
        activation::sigmoid(self.opacities.to_owned())
    }

    /// `[P, 1]`
    pub fn set_opacities(
        &mut self,
        opacities: Tensor<B, 2>,
    ) -> &mut Self {
        debug_assert_eq!(opacities.dims()[1], 1, "opacities.dims()[1] != 1");
        self.opacities = (opacities.to_owned() / (-opacities + 1.0)).log();
        self
    }

    /// `[P, 3]`
    pub fn positions(&self) -> Tensor<B, 2> {
        self.positions.to_owned()
    }

    /// `[P, 3]`
    pub fn set_positions(
        &mut self,
        positions: Tensor<B, 2>,
    ) -> &mut Self {
        debug_assert_eq!(positions.dims()[1], 3, "positions.dims()[1] != 3");
        self.positions = positions;
        self
    }

    /// `[P, 4]`
    pub fn rotations(&self) -> Tensor<B, 2> {
        let norm = (self.rotations.to_owned() * self.rotations.to_owned())
            .sum_dim(1)
            .sqrt();
        self.rotations.to_owned().div(norm)
    }

    /// `[P, 4]`
    pub fn set_rotations(
        &mut self,
        rotations: Tensor<B, 2>,
    ) -> &mut Self {
        debug_assert_eq!(rotations.dims()[1], 4, "rotations.dims()[1] != 4");
        self.rotations = rotations;
        self
    }

    /// `[P, 3]`
    pub fn scalings(&self) -> Tensor<B, 2> {
        self.scalings.to_owned().exp()
    }

    /// `[P, 4]`
    pub fn set_scalings(
        &mut self,
        scalings: Tensor<B, 2>,
    ) -> &mut Self {
        debug_assert_eq!(scalings.dims()[1], 3, "scalings.dims()[1] != 3");
        self.scalings = scalings.log();
        self
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn set_opacities() {
        use super::*;

        let device = Default::default();

        let mut scene = Gaussian3dScene::<burn::backend::NdArray>::default();
        scene
            .set_opacities(Tensor::from_floats([[0.0], [1.0], [0.5]], &device));

        let opacities = scene.opacities();
        let opacities_expected =
            Tensor::from_floats([[0.0], [1.0], [0.5]], &device);
        assert!(
            opacities
                .to_owned()
                .equal(opacities_expected.to_owned())
                .all()
                .into_scalar(),
            "{:?} != {:?}",
            opacities,
            opacities_expected
        );
    }

    #[test]
    fn set_rotations() {
        use super::*;

        let device = Default::default();

        let mut scene = Gaussian3dScene::<burn::backend::NdArray>::default();
        scene.set_rotations(Tensor::from_floats(
            [[1.0, -2.0, 3.0, 0.0], [0.3, 0.0, -0.1, 0.0]],
            &device,
        ));

        let rotations = scene.rotations();
        let rotations_expected = Tensor::from_floats(
            [
                [
                    1.0 / 3.7416573867739413,
                    -2.0 / 3.7416573867739413,
                    3.0 / 3.7416573867739413,
                    0.0 / 3.7416573867739413,
                ],
                [
                    0.3 / 0.31622776601683794,
                    0.0 / 0.31622776601683794,
                    -0.1 / 0.31622776601683794,
                    0.0 / 0.31622776601683794,
                ],
            ],
            &device,
        );
        assert!(
            rotations
                .to_owned()
                .equal(rotations_expected.to_owned())
                .all()
                .into_scalar(),
            "{:?} != {:?}",
            rotations,
            rotations_expected
        );
    }

    #[test]
    fn set_scalings() {
        use super::*;

        let device = Default::default();

        let mut scene = Gaussian3dScene::<burn::backend::NdArray>::default();
        scene.set_scalings(Tensor::from_floats(
            [[1.0, 2.0, 2.0], [0.4, 0.0, 2.7]],
            &device,
        ));

        let scalings = scene.scalings();
        let scalings_expected =
            Tensor::from_floats([[1.0, 2.0, 2.0], [0.4, 0.0, 2.7]], &device);
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
