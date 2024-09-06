pub use super::*;

use burn::tensor::{activation, ElementConversion};

impl<B: Backend> Gaussian3dScene<B> {
    /// The colors represented as orthonormalized spherical harmonics
    ///
    /// `[P, 16 * 3]`
    pub fn colors_sh(&self) -> Tensor<B, 2> {
        self.colors_sh.val()
    }

    /// Making for [`Gaussian3dScene::colors_sh`]
    pub fn make_colors_sh(colors_sh: Tensor<B, 2>) -> Tensor<B, 2> {
        colors_sh
    }

    /// Setting for [`Gaussian3dScene::colors_sh`]
    pub fn set_colors_sh(
        &mut self,
        colors_sh: Tensor<B, 2>,
    ) -> &mut Self {
        self.colors_sh =
            Param::initialized(self.colors_sh.id.to_owned(), colors_sh);
        self
    }

    /// Making and setting for [`Gaussian3dScene::colors_sh`]
    pub fn make_set_colors_sh(
        &mut self,
        colors_sh: Tensor<B, 2>,
    ) -> &mut Self {
        self.set_colors_sh(Self::make_colors_sh(colors_sh));
        self
    }

    /// The opacities are applied to the sigmoid function
    ///
    /// `[P, 1]`
    pub fn opacities(&self) -> Tensor<B, 2> {
        activation::sigmoid(self.opacities.val())
    }

    /// Making for [`Gaussian3dScene::opacities`]
    pub fn make_opacities(opacities: Tensor<B, 2>) -> Tensor<B, 2> {
        opacities.to_owned().div(-opacities + 1.0).log()
    }

    /// Setting for [`Gaussian3dScene::opacities`]
    pub fn set_opacities(
        &mut self,
        opacities: Tensor<B, 2>,
    ) -> &mut Self {
        self.opacities =
            Param::initialized(self.opacities.id.to_owned(), opacities);
        self
    }

    /// Making and setting for [`Gaussian3dScene::opacities`]
    pub fn make_set_opacities(
        &mut self,
        opacities: Tensor<B, 2>,
    ) -> &mut Self {
        self.set_opacities(Self::make_opacities(opacities))
    }

    /// The 3D positions in world space
    ///
    /// `[P, 3]`
    pub fn positions(&self) -> Tensor<B, 2> {
        self.positions.val()
    }

    /// Making for [`Gaussian3dScene::positions`]
    pub fn make_positions(positions: Tensor<B, 2>) -> Tensor<B, 2> {
        positions
    }

    /// Setting for [`Gaussian3dScene::positions`]
    pub fn set_positions(
        &mut self,
        positions: Tensor<B, 2>,
    ) -> &mut Self {
        self.positions =
            Param::initialized(self.positions.id.to_owned(), positions);
        self
    }

    /// Making and setting for [`Gaussian3dScene::positions`]
    pub fn make_set_positions(
        &mut self,
        positions: Tensor<B, 2>,
    ) -> &mut Self {
        self.set_positions(Self::make_positions(positions))
    }

    /// The rotations are normalized quaternions
    ///
    /// `[P, 4]`
    pub fn rotations(&self) -> Tensor<B, 2> {
        let rotations = self.rotations.val();
        let norms = rotations.to_owned().powf_scalar(2.0).sum_dim(1).sqrt();
        rotations.div(norms)
    }

    /// Making for [`Gaussian3dScene::rotations`]
    pub fn make_rotations(rotations: Tensor<B, 2>) -> Tensor<B, 2> {
        rotations
    }

    /// Setting for [`Gaussian3dScene::rotations`]
    pub fn set_rotations(
        &mut self,
        rotations: Tensor<B, 2>,
    ) -> &mut Self {
        self.rotations =
            Param::initialized(self.rotations.id.to_owned(), rotations);
        self
    }

    /// Making and setting for [`Gaussian3dScene::rotations`]
    pub fn make_set_rotations(
        &mut self,
        rotations: Tensor<B, 2>,
    ) -> &mut Self {
        self.set_rotations(Self::make_rotations(rotations))
    }

    /// The scalings are applied to the exponential function
    ///
    /// `[P, 3]`
    pub fn scalings(&self) -> Tensor<B, 2> {
        self.scalings.val().exp()
    }

    /// Making for [`Gaussian3dScene::scalings`]
    pub fn make_scalings(scalings: Tensor<B, 2>) -> Tensor<B, 2> {
        scalings.log()
    }

    /// Setting for [`Gaussian3dScene::scalings`]
    pub fn set_scalings(
        &mut self,
        scalings: Tensor<B, 2>,
    ) -> &mut Self {
        self.scalings =
            Param::initialized(self.scalings.id.to_owned(), scalings);
        self
    }

    /// Making and setting for [`Gaussian3dScene::scalings`]
    pub fn make_set_scalings(
        &mut self,
        scalings: Tensor<B, 2>,
    ) -> &mut Self {
        self.set_scalings(Self::make_scalings(scalings))
    }

    /// Making and dividing [`Gaussian3dScene::scalings`] by scalar
    pub fn make_divs_scalings(
        &mut self,
        scaling: B::FloatElem,
    ) -> &mut Self {
        self.set_scalings(
            self.scalings.val().sub_scalar(scaling.elem::<f64>().ln()),
        )
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn make_set_input_equals_to_output() {
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
            .make_set_colors_sh(input_colors_sh.to_owned())
            .make_set_opacities(input_opacities.to_owned())
            .make_set_positions(input_positions.to_owned())
            .make_set_scalings(input_scalings.to_owned());

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

    #[test]
    fn make_divs_scalings() {
        use super::*;
        use burn::{backend::NdArray, tensor::Distribution};

        let device = Default::default();

        let input_scalings = Tensor::<NdArray<f32>, 2>::random(
            [10, 3],
            Distribution::Default,
            &device,
        );
        let scaling = 0.25;

        let output_scalings = Gaussian3dScene::<NdArray<f32>>::default()
            .make_set_scalings(input_scalings.to_owned())
            .make_divs_scalings(scaling)
            .scalings();

        let expected_scalings = input_scalings.to_owned().div_scalar(scaling);

        output_scalings
            .into_data()
            .assert_approx_eq(&expected_scalings.into_data(), 6);
    }
}
