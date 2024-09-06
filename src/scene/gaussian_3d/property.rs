pub use super::*;

use burn::tensor::activation;

/// Outer value getters
impl<B: Backend> Gaussian3dScene<B> {
    /// The colors represented as orthonormalized spherical harmonics
    ///
    /// `[P, 16 * 3]`
    #[inline]
    pub fn colors_sh(&self) -> Tensor<B, 2> {
        Self::make_colors_sh(self.colors_sh.val())
    }

    /// The opacities are applied to the sigmoid function
    ///
    /// `[P, 1]`
    #[inline]
    pub fn opacities(&self) -> Tensor<B, 2> {
        Self::make_opacities(self.opacities.val())
    }

    /// The 3D positions in world space
    ///
    /// `[P, 3]`
    #[inline]
    pub fn positions(&self) -> Tensor<B, 2> {
        Self::make_positions(self.positions.val())
    }

    /// The rotations are normalized quaternions
    ///
    /// `[P, 4]`
    #[inline]
    pub fn rotations(&self) -> Tensor<B, 2> {
        Self::make_rotations(self.rotations.val())
    }

    /// The scalings are applied to the exponential function
    ///
    /// `[P, 3]`
    #[inline]
    pub fn scalings(&self) -> Tensor<B, 2> {
        Self::make_scalings(self.scalings.val())
    }
}

/// Outer value makers
impl<B: Backend> Gaussian3dScene<B> {
    /// Making values for [`Gaussian3dScene::colors_sh`]
    #[inline]
    pub fn make_colors_sh(colors_sh: Tensor<B, 2>) -> Tensor<B, 2> {
        colors_sh
    }

    /// Making values for [`Gaussian3dScene::opacities`]
    #[inline]
    pub fn make_opacities(opacities: Tensor<B, 2>) -> Tensor<B, 2> {
        activation::sigmoid(opacities)
    }

    /// Making values for [`Gaussian3dScene::positions`]
    #[inline]
    pub fn make_positions(positions: Tensor<B, 2>) -> Tensor<B, 2> {
        positions
    }

    /// Making values for [`Gaussian3dScene::rotations`]
    #[inline]
    pub fn make_rotations(rotations: Tensor<B, 2>) -> Tensor<B, 2> {
        rotations
            .to_owned()
            .div(rotations.powf_scalar(2.0).sum_dim(1).sqrt())
    }

    /// Making values for [`Gaussian3dScene::scalings`]
    #[inline]
    pub fn make_scalings(scalings: Tensor<B, 2>) -> Tensor<B, 2> {
        scalings.exp()
    }
}

/// Outer value setters
impl<B: Backend> Gaussian3dScene<B> {
    /// Setting values for [`Gaussian3dScene::colors_sh`]
    pub fn set_colors_sh(
        &mut self,
        colors_sh: Tensor<B, 2>,
    ) -> &mut Self {
        self.set_inner_colors_sh(Self::make_inner_colors_sh(colors_sh));
        self
    }

    /// Setting values for [`Gaussian3dScene::opacities`]
    pub fn set_opacities(
        &mut self,
        opacities: Tensor<B, 2>,
    ) -> &mut Self {
        self.set_inner_opacities(Self::make_inner_opacities(opacities))
    }

    /// Setting values for [`Gaussian3dScene::positions`]
    pub fn set_positions(
        &mut self,
        positions: Tensor<B, 2>,
    ) -> &mut Self {
        self.set_inner_positions(Self::make_inner_positions(positions))
    }

    /// Setting values for [`Gaussian3dScene::rotations`]
    pub fn set_rotations(
        &mut self,
        rotations: Tensor<B, 2>,
    ) -> &mut Self {
        self.set_inner_rotations(Self::make_inner_rotations(rotations))
    }

    /// Setting values for [`Gaussian3dScene::scalings`]
    pub fn set_scalings(
        &mut self,
        scalings: Tensor<B, 2>,
    ) -> &mut Self {
        self.set_inner_scalings(Self::make_inner_scalings(scalings))
    }
}

/// Inner value makers
impl<B: Backend> Gaussian3dScene<B> {
    /// Making inner values for [`Gaussian3dScene::colors_sh`]
    #[inline]
    pub fn make_inner_colors_sh(colors_sh: Tensor<B, 2>) -> Tensor<B, 2> {
        colors_sh
    }

    /// Making inner values for [`Gaussian3dScene::opacities`]
    #[inline]
    pub fn make_inner_opacities(opacities: Tensor<B, 2>) -> Tensor<B, 2> {
        opacities.to_owned().div(-opacities + 1.0).log()
    }

    /// Making inner values for [`Gaussian3dScene::positions`]
    #[inline]
    pub fn make_inner_positions(positions: Tensor<B, 2>) -> Tensor<B, 2> {
        positions
    }

    /// Making inner values for [`Gaussian3dScene::rotations`]
    #[inline]
    pub fn make_inner_rotations(rotations: Tensor<B, 2>) -> Tensor<B, 2> {
        rotations
    }

    /// Making inner values for [`Gaussian3dScene::scalings`]
    #[inline]
    pub fn make_inner_scalings(scalings: Tensor<B, 2>) -> Tensor<B, 2> {
        scalings.log()
    }
}

/// Inner value setters
impl<B: Backend> Gaussian3dScene<B> {
    /// Setting inner values for [`Gaussian3dScene::colors_sh`]
    pub fn set_inner_colors_sh(
        &mut self,
        colors_sh: Tensor<B, 2>,
    ) -> &mut Self {
        self.colors_sh =
            Param::initialized(self.colors_sh.id.to_owned(), colors_sh);
        self
    }

    /// Setting inner values for [`Gaussian3dScene::opacities`]
    pub fn set_inner_opacities(
        &mut self,
        opacities: Tensor<B, 2>,
    ) -> &mut Self {
        self.opacities =
            Param::initialized(self.opacities.id.to_owned(), opacities);
        self
    }

    /// Setting inner values for [`Gaussian3dScene::positions`]
    pub fn set_inner_positions(
        &mut self,
        positions: Tensor<B, 2>,
    ) -> &mut Self {
        self.positions =
            Param::initialized(self.positions.id.to_owned(), positions);
        self
    }

    /// Setting inner values for [`Gaussian3dScene::rotations`]
    pub fn set_inner_rotations(
        &mut self,
        rotations: Tensor<B, 2>,
    ) -> &mut Self {
        self.rotations =
            Param::initialized(self.rotations.id.to_owned(), rotations);
        self
    }

    /// Setting inner values for [`Gaussian3dScene::scalings`]
    pub fn set_inner_scalings(
        &mut self,
        scalings: Tensor<B, 2>,
    ) -> &mut Self {
        self.scalings =
            Param::initialized(self.scalings.id.to_owned(), scalings);
        self
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn set() {
        use super::*;
        use burn::{backend::NdArray, tensor::Distribution};

        let device = Default::default();

        let input_colors_sh = Tensor::<NdArray<f32>, 2>::random(
            [10, 16 * 3],
            Distribution::Default,
            &device,
        );
        let input_opacities = Tensor::<NdArray<f32>, 2>::random(
            [10, 1],
            Distribution::Default,
            &device,
        );
        let input_positions = Tensor::<NdArray<f32>, 2>::random(
            [10, 3],
            Distribution::Default,
            &device,
        );
        let input_scalings = Tensor::<NdArray<f32>, 2>::random(
            [10, 3],
            Distribution::Default,
            &device,
        );

        let mut scene = Gaussian3dScene::<NdArray<f32>>::default();

        scene
            .set_colors_sh(input_colors_sh.to_owned())
            .set_opacities(input_opacities.to_owned())
            .set_positions(input_positions.to_owned())
            .set_scalings(input_scalings.to_owned());

        input_colors_sh
            .into_data()
            .assert_approx_eq(&scene.colors_sh().into_data(), 6);
        input_opacities
            .into_data()
            .assert_approx_eq(&scene.opacities().into_data(), 6);
        input_positions
            .into_data()
            .assert_approx_eq(&scene.positions().into_data(), 6);
        input_scalings
            .into_data()
            .assert_approx_eq(&scene.scalings().into_data(), 6);
    }
}
