use super::*;
use burn::tensor::activation;

impl<B: Backend> Gaussian3dScene<B> {
    /// The colors represented as orthonormalized spherical harmonics
    ///
    /// `[P, 16, 3]`
    pub fn colors_sh(&self) -> Tensor<B, 3> {
        self.colors_sh.val()
    }

    /// Making for [`Gaussian3dScene::colors_sh`]
    pub fn make_colors_sh(colors_sh: Tensor<B, 3>) -> Tensor<B, 3> {
        colors_sh
    }

    /// Setting for [`Gaussian3dScene::colors_sh`]
    pub fn set_colors_sh(
        &mut self,
        colors_sh: Tensor<B, 3>,
    ) -> &mut Self {
        self.colors_sh =
            Param::initialized(self.colors_sh.id.to_owned(), colors_sh);
        self
    }

    /// Making and setting for [`Gaussian3dScene::colors_sh`]
    pub fn make_and_set_colors_sh(
        &mut self,
        colors_sh: Tensor<B, 3>,
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
    pub fn make_and_set_opacities(
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
    pub fn make_and_set_positions(
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
    pub fn make_and_set_rotations(
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
    pub fn make_and_set_scalings(
        &mut self,
        scalings: Tensor<B, 2>,
    ) -> &mut Self {
        self.set_scalings(Self::make_scalings(scalings))
    }
}
