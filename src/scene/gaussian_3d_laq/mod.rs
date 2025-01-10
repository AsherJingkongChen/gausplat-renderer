//! 3DGS-LAQ, 3DGS scene representation using Learned Attribute Quantization (LAQ).

pub mod export;
pub mod import;
pub mod property;

pub use super::*;
pub use burn::{
    module::{AutodiffModule, Module, Param},
    tensor::{
        backend::{AutodiffBackend, Backend},
        Tensor, TensorData,
    },
};
pub use gaussian_3d::Gaussian3dScene;
pub use laq::Dequantizer;

use std::fmt;

/// 3DGS scene representation using Learned Attribute Quantization (LAQ).
///
/// ## Details
///
/// Some of the attributes in 3DGS scene are not quantized, including:
/// - [`positions`](Gaussian3dScene::positions)
#[derive(Module)]
pub struct Gaussian3dSceneLAQ<B: Backend> {
    /// Quantized latents estimation of [colors in SH space (only degree 0)](Gaussian3dScene::colors_sh).
    ///
    /// The shape is `[P, 3]`.
    /// - `P` is [`Self::point_count`].
    pub colors_sh_dc_q_h: Param<Tensor<B, 2>>,
    /// Quantized latents estimation of [colors in SH space (except degree 0)](Gaussian3dScene::colors_sh).
    ///
    /// The shape is `[P, (M - 1) * 3]`.
    /// - `P` is [`Self::point_count`].
    /// - `M` is [`SH_COUNT_MAX`](gaussian_3d::SH_COUNT_MAX).
    pub colors_sh_rest_q_h: Param<Tensor<B, 2>>,
    /// Quantized latents estimation of [opacities](Gaussian3dScene::opacities).
    pub opacities_q_h: Param<Tensor<B, 2>>,
    /// Quantized latents estimation of [rotations as quaternions](Gaussian3dScene::rotations).
    pub rotations_q_h: Param<Tensor<B, 2>>,
    /// Quantized latents estimation of [3D scalings](Gaussian3dScene::scalings).
    pub scalings_q_h: Param<Tensor<B, 2>>,

    /// De-quantize from [`Self::colors_sh_dc_q_h`] to [`Gaussian3dScene::colors_sh`] (only degree 0).
    pub dequantizer_colors_sh_dc: Dequantizer<B>,
    /// De-quantize from [`Self::colors_sh_rest_q_h`] to [`Gaussian3dScene::colors_sh`] (except degree 0).
    pub dequantizer_colors_sh_rest: Dequantizer<B>,
    /// De-quantize from [`Self::opacities_q_h`] to [`Gaussian3dScene::opacities`].
    pub dequantizer_opacities: Dequantizer<B>,
    /// De-quantize from [`Self::rotations_q_h`] to [`Gaussian3dScene::rotations`].
    pub dequantizer_rotations: Dequantizer<B>,
    /// De-quantize from [`Self::scalings_q_h`] to [`Gaussian3dScene::scalings`].
    pub dequantizer_scalings: Dequantizer<B>,
}

impl<B: Backend> fmt::Debug for Gaussian3dSceneLAQ<B> {
    fn fmt(
        &self,
        f: &mut fmt::Formatter,
    ) -> fmt::Result {
        f.debug_struct(&format!("Gaussian3dSceneLAQ<{}>", B::name()))
            .field("device", &self.device())
            .field("point_count", &self.point_count())
            .field("size", &self.size_readable())
            .field("colors_sh_dc_q_h.dims()", &self.colors_sh_dc_q_h.dims())
            .field("colors_sh_rest_q_h.dims()", &self.colors_sh_rest_q_h.dims())
            .field("opacities_q_h.dims()", &self.opacities_q_h.dims())
            .field("rotations_q_h.dims()", &self.rotations_q_h.dims())
            .field("scalings_q_h.dims()", &self.scalings_q_h.dims())
            .finish()
    }
}

impl<B: Backend> Default for Gaussian3dSceneLAQ<B> {
    fn default() -> Self {
        let colors_sh_dc_q_h = Param::uninitialized(
            Default::default(),
            |_, _| unimplemented!(),
            Default::default(),
            true,
        );
        let colors_sh_rest_q_h = Param::uninitialized(
            Default::default(),
            |_, _| unimplemented!(),
            Default::default(),
            true,
        );
        let opacities_q_h = Param::uninitialized(
            Default::default(),
            |_, _| unimplemented!(),
            Default::default(),
            true,
        );
        let rotations_q_h = Param::uninitialized(
            Default::default(),
            |_, _| unimplemented!(),
            Default::default(),
            true,
        );
        let scalings_q_h = Param::uninitialized(
            Default::default(),
            |_, _| unimplemented!(),
            Default::default(),
            true,
        );

        let dequantizer_colors_sh_dc = Default::default();
        let dequantizer_colors_sh_rest = Default::default();
        let dequantizer_opacities = Default::default();
        let dequantizer_rotations = Default::default();
        let dequantizer_scalings = Default::default();

        Self {
            colors_sh_dc_q_h,
            colors_sh_rest_q_h,
            opacities_q_h,
            rotations_q_h,
            scalings_q_h,
            dequantizer_colors_sh_dc,
            dequantizer_colors_sh_rest,
            dequantizer_opacities,
            dequantizer_rotations,
            dequantizer_scalings,
        }
    }
}
