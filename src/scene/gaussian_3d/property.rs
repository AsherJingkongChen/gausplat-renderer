//! 3DGS scene property implementation.

pub use super::*;

use burn::tensor::activation;
use humansize::{format_size, BINARY};

/// Outer property value getters
impl<B: Backend> Gaussian3dScene<B> {
    /// Colors in SH space. (Outer value)
    ///
    /// The shape is `[P, M * 3]`, which derives from `[P, M, 3]`.
    /// - `P` is [`Self::point_count`].
    /// - `M` is [`SH_COUNT_MAX`].
    ///
    /// It is represented as orthonormalized spherical harmonic with RGB channels.
    #[inline]
    pub fn get_colors_sh(&self) -> Tensor<B, 2> {
        Self::make_colors_sh(self.colors_sh.val())
    }

    /// Opacities. (Outer value)
    ///
    /// The shape is `[P, 1]`.
    ///
    /// They range from `0.0` to `1.0`.
    #[inline]
    pub fn get_opacities(&self) -> Tensor<B, 2> {
        Self::make_opacities(self.opacities.val())
    }

    /// 3D Positions. (Outer value)
    ///
    /// The shape is `[P, 3]`.
    #[inline]
    pub fn get_positions(&self) -> Tensor<B, 2> {
        Self::make_positions(self.positions.val())
    }

    /// Rotations. (Outer value)
    ///
    /// The shape is `[P, 4]`.
    ///
    /// They are represented as normalized Hamilton quaternions in scalar-last order,
    /// i.e., `[x, y, z, w]`.
    #[inline]
    pub fn get_rotations(&self) -> Tensor<B, 2> {
        Self::make_rotations(self.rotations.val())
    }

    /// 3D scalings. (Outer value)
    ///
    /// The shape is `[P, 3]`.
    #[inline]
    pub fn get_scalings(&self) -> Tensor<B, 2> {
        Self::make_scalings(self.scalings.val())
    }
}

/// Outer property value makers
impl<B: Backend> Gaussian3dScene<B> {
    /// Making values for [`Gaussian3dScene::get_colors_sh`]
    #[inline]
    pub fn make_colors_sh(colors_sh: Tensor<B, 2>) -> Tensor<B, 2> {
        colors_sh
    }

    /// Making values for [`Gaussian3dScene::get_opacities`]
    #[inline]
    pub fn make_opacities(opacities: Tensor<B, 2>) -> Tensor<B, 2> {
        activation::sigmoid(opacities)
    }

    /// Making values for [`Gaussian3dScene::get_positions`]
    #[inline]
    pub fn make_positions(positions: Tensor<B, 2>) -> Tensor<B, 2> {
        positions
    }

    /// Making values for [`Gaussian3dScene::get_rotations`]
    #[inline]
    pub fn make_rotations(rotations: Tensor<B, 2>) -> Tensor<B, 2> {
        rotations
            .to_owned()
            .div(rotations.powf_scalar(2.0).sum_dim(1).sqrt())
    }

    /// Making values for [`Gaussian3dScene::get_scalings`]
    #[inline]
    pub fn make_scalings(scalings: Tensor<B, 2>) -> Tensor<B, 2> {
        scalings.exp()
    }
}

/// Outer property value setters
impl<B: Backend> Gaussian3dScene<B> {
    /// Setting values for [`Gaussian3dScene::get_colors_sh`]
    pub fn set_colors_sh(
        &mut self,
        colors_sh: Tensor<B, 2>,
    ) -> &mut Self {
        self.set_inner_colors_sh(Self::make_inner_colors_sh(colors_sh));
        self
    }

    /// Setting values for [`Gaussian3dScene::get_opacities`]
    pub fn set_opacities(
        &mut self,
        opacities: Tensor<B, 2>,
    ) -> &mut Self {
        self.set_inner_opacities(Self::make_inner_opacities(opacities))
    }

    /// Setting values for [`Gaussian3dScene::get_positions`]
    pub fn set_positions(
        &mut self,
        positions: Tensor<B, 2>,
    ) -> &mut Self {
        self.set_inner_positions(Self::make_inner_positions(positions))
    }

    /// Setting values for [`Gaussian3dScene::get_rotations`]
    pub fn set_rotations(
        &mut self,
        rotations: Tensor<B, 2>,
    ) -> &mut Self {
        self.set_inner_rotations(Self::make_inner_rotations(rotations))
    }

    /// Setting values for [`Gaussian3dScene::get_scalings`]
    pub fn set_scalings(
        &mut self,
        scalings: Tensor<B, 2>,
    ) -> &mut Self {
        self.set_inner_scalings(Self::make_inner_scalings(scalings))
    }
}

/// Inner property value makers
impl<B: Backend> Gaussian3dScene<B> {
    /// Making values for [`Gaussian3dScene::colors_sh`]
    #[inline]
    pub fn make_inner_colors_sh(colors_sh: Tensor<B, 2>) -> Tensor<B, 2> {
        colors_sh
    }

    /// Making values for [`Gaussian3dScene::opacities`]
    #[inline]
    pub fn make_inner_opacities(opacities: Tensor<B, 2>) -> Tensor<B, 2> {
        opacities.to_owned().div(-opacities + 1.0).log()
    }

    /// Making values for [`Gaussian3dScene::positions`]
    #[inline]
    pub fn make_inner_positions(positions: Tensor<B, 2>) -> Tensor<B, 2> {
        positions
    }

    /// Making values for [`Gaussian3dScene::rotations`]
    #[inline]
    pub fn make_inner_rotations(rotations: Tensor<B, 2>) -> Tensor<B, 2> {
        rotations
    }

    /// Making values for [`Gaussian3dScene::scalings`]
    #[inline]
    pub fn make_inner_scalings(scalings: Tensor<B, 2>) -> Tensor<B, 2> {
        scalings.log()
    }
}

/// Inner property value setters
impl<B: Backend> Gaussian3dScene<B> {
    /// Setting inner values for [`Gaussian3dScene::colors_sh`]
    #[inline]
    pub fn set_inner_colors_sh(
        &mut self,
        colors_sh: Tensor<B, 2>,
    ) -> &mut Self {
        self.colors_sh = Param::initialized(self.colors_sh.id.to_owned(), colors_sh);
        self
    }

    /// Setting inner values for [`Gaussian3dScene::opacities`]
    #[inline]
    pub fn set_inner_opacities(
        &mut self,
        opacities: Tensor<B, 2>,
    ) -> &mut Self {
        self.opacities = Param::initialized(self.opacities.id.to_owned(), opacities);
        self
    }

    /// Setting inner values for [`Gaussian3dScene::positions`]
    #[inline]
    pub fn set_inner_positions(
        &mut self,
        positions: Tensor<B, 2>,
    ) -> &mut Self {
        self.positions = Param::initialized(self.positions.id.to_owned(), positions);
        self
    }

    /// Setting inner values for [`Gaussian3dScene::rotations`]
    #[inline]
    pub fn set_inner_rotations(
        &mut self,
        rotations: Tensor<B, 2>,
    ) -> &mut Self {
        self.rotations = Param::initialized(self.rotations.id.to_owned(), rotations);
        self
    }

    /// Setting inner values for [`Gaussian3dScene::scalings`]
    #[inline]
    pub fn set_inner_scalings(
        &mut self,
        scalings: Tensor<B, 2>,
    ) -> &mut Self {
        self.scalings = Param::initialized(self.scalings.id.to_owned(), scalings);
        self
    }
}

/// Attribute getters
impl<B: Backend> Gaussian3dScene<B> {
    /// The device.
    #[inline]
    pub fn device(&self) -> B::Device {
        self.devices().first().expect("A device").to_owned()
    }

    /// Number of points.
    #[inline]
    pub fn point_count(&self) -> usize {
        let point_count_target = self.colors_sh.dims()[0];
        let point_count_other = self.opacities.dims()[0];
        debug_assert_eq!(point_count_other, point_count_target);
        let point_count_other = self.positions.dims()[0];
        debug_assert_eq!(point_count_other, point_count_target);
        let point_count_other = self.rotations.dims()[0];
        debug_assert_eq!(point_count_other, point_count_target);
        let point_count_other = self.scalings.dims()[0];
        debug_assert_eq!(point_count_other, point_count_target);

        point_count_target
    }

    /// Size of the parameters in bytes.
    #[inline]
    pub fn size(&self) -> usize {
        self.num_params() * size_of::<B::FloatElem>()
    }

    /// Readable size of the parameters.
    #[inline]
    pub fn size_readable(&self) -> String {
        format_size(self.size(), BINARY.decimal_places(1))
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn set_outer_property() {
        use super::*;
        use burn::{backend::NdArray, tensor::Distribution};

        let device = Default::default();

        let input_colors_sh =
            Tensor::<NdArray<f32>, 2>::random([10, 48], Distribution::Default, &device);
        let input_rotations = Tensor::<NdArray<f32>, 2>::ones([10, 4], &device);
        let input_opacities =
            Tensor::<NdArray<f32>, 2>::random([10, 1], Distribution::Default, &device);
        let input_positions =
            Tensor::<NdArray<f32>, 2>::random([10, 3], Distribution::Default, &device);
        let input_scalings =
            Tensor::<NdArray<f32>, 2>::random([10, 3], Distribution::Default, &device)
                .add_scalar(1.0);

        let mut scene = Gaussian3dScene::<NdArray<f32>>::default();

        scene
            .set_colors_sh(input_colors_sh.to_owned())
            .set_opacities(input_opacities.to_owned())
            .set_positions(input_positions.to_owned())
            .set_rotations(input_rotations.to_owned())
            .set_scalings(input_scalings.to_owned());

        assert_eq!(scene.point_count(), 10);

        input_colors_sh
            .into_data()
            .assert_approx_eq(&scene.get_colors_sh().into_data(), 6);
        input_opacities
            .into_data()
            .assert_approx_eq(&scene.get_opacities().into_data(), 6);
        input_positions
            .into_data()
            .assert_approx_eq(&scene.get_positions().into_data(), 6);
        assert!(
            input_rotations
                .not_equal(scene.get_rotations())
                .all()
                .into_scalar(),
            "Rotations should be not equal to unnormalized ones"
        );
        input_scalings
            .into_data()
            .assert_approx_eq(&scene.get_scalings().into_data(), 6);
    }
}
