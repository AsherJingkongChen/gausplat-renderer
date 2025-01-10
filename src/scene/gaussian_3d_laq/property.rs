//! 3DGS-LAQ Property implementation.

pub use super::*;

use humansize::{format_size, BINARY};
use std::ops::Add;

/// Inner property value estimator.
impl<B: Backend> Gaussian3dSceneLAQ<B> {
    /// Estimating values for [`Gaussian3dScene::colors_sh`].
    pub fn get_estimated_colors_sh(&self) -> Tensor<B, 2> {
        let colors_sh_dc_h = self
            .dequantizer_colors_sh_dc
            .forward(self.colors_sh_dc_q_h.val(), false);
        let colors_sh_rest_h = self
            .dequantizer_colors_sh_rest
            .forward(self.colors_sh_rest_q_h.val(), false);
        let colors_sh_h = Tensor::cat([colors_sh_dc_h, colors_sh_rest_h].into(), 1);
        colors_sh_h
    }

    /// Estimating values for [`Gaussian3dScene::opacities`].
    pub fn get_estimated_opacities(&self) -> Tensor<B, 2> {
        self.dequantizer_opacities
            .forward(self.opacities_q_h.val(), false)
    }

    /// Estimating values for [`Gaussian3dScene::rotations`].
    pub fn get_estimated_rotations(&self) -> Tensor<B, 2> {
        self.dequantizer_rotations
            .forward(self.rotations_q_h.val(), false)
    }

    /// Estimating values for [`Gaussian3dScene::scalings`].
    pub fn get_estimated_scalings(&self) -> Tensor<B, 2> {
        self.dequantizer_scalings
            .forward(self.scalings_q_h.val(), false)
    }
}

/// Attribute getters
impl<B: Backend> Gaussian3dSceneLAQ<B> {
    /// The device.
    #[inline]
    pub fn device(&self) -> B::Device {
        self.devices().first().expect("A device").to_owned()
    }

    /// Number of points.
    #[inline]
    pub fn point_count(&self) -> usize {
        let point_count_target = self.colors_sh_dc_q_h.dims()[0];
        let point_count_other = self.colors_sh_rest_q_h.dims()[0];
        debug_assert_eq!(point_count_other, point_count_target);
        let point_count_other = self.opacities_q_h.dims()[0];
        debug_assert_eq!(point_count_other, point_count_target);
        let point_count_other = self.rotations_q_h.dims()[0];
        debug_assert_eq!(point_count_other, point_count_target);
        let point_count_other = self.scalings_q_h.dims()[0];
        debug_assert_eq!(point_count_other, point_count_target);

        point_count_target
    }

    /// Size of the parameters in bytes.
    #[inline]
    pub fn size(&self) -> usize {
        let quantized_latent_size = (self
            .colors_sh_dc_q_h
            .num_params()
            .add(self.colors_sh_rest_q_h.num_params())
            .add(self.opacities_q_h.num_params())
            .add(self.rotations_q_h.num_params())
            .add(self.scalings_q_h.num_params()))
            * size_of::<u8>();
        let dequantizer_size = (self
            .dequantizer_colors_sh_dc
            .num_params()
            .add(self.dequantizer_colors_sh_rest.num_params())
            .add(self.dequantizer_opacities.num_params())
            .add(self.dequantizer_rotations.num_params())
            .add(self.dequantizer_scalings.num_params()))
            * size_of::<B::FloatElem>();
        let size = quantized_latent_size + dequantizer_size;
        size
    }

    /// Readable size of the parameters.
    #[inline]
    pub fn size_readable(&self) -> String {
        format_size(self.size(), BINARY.decimal_places(1))
    }
}
