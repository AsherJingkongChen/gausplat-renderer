//! 3DGS rendering implementation.
//!
//! For more information, see:
//! 1. [3DGS survey](https://arxiv.org/abs/2401.03890).
//! 2. [JIT kernel API](jit::kernel).

pub mod backward;
pub mod forward;
pub mod jit;

pub use super::view::*;
pub use crate::{
    backend::{autodiff, Autodiff, AutodiffBackend, Backend},
    error::Error,
    spherical_harmonics::SH_DEGREE_MAX,
};
pub use burn::{
    config::Config,
    record::Record,
    tensor::{Int, Tensor},
};

use std::fmt;

/// 3DGS scene renderer.
pub trait Gaussian3dRenderer<B: Backend>: 'static + Send + Sized + fmt::Debug {
    /// Render the 3DGS scene (forward).
    fn render_forward(
        input: forward::RenderInput<B>,
        view: &View,
        options: &Gaussian3dRenderOptions,
    ) -> Result<forward::RenderOutput<B>, Error>;

    /// Render the 3DGS scene (backward).
    ///
    /// It computes the gradients from
    /// the [output in forward pass](forward::RenderOutput).
    fn render_backward(
        state: backward::RenderInput<B>,
        colors_rgb_2d_grad: B::FloatTensorPrimitive,
    ) -> backward::RenderOutput<B>;
}

/// 3DGS rendering options.
#[derive(Config, Copy, Debug, PartialEq, Record)]
pub struct Gaussian3dRenderOptions {
    #[config(default = "SH_DEGREE_MAX")]
    /// The maximum degree of color in SH space.
    ///
    /// It should be no more than [`SH_DEGREE_MAX`].
    pub colors_sh_degree_max: u32,
}

/// 3DGS rendering output.
#[derive(Clone)]
pub struct Gaussian3dRenderOutput<B: Backend> {
    /// `[I_y, I_x, 3]`
    pub colors_rgb_2d: Tensor<B, 3>,
    // TODO: THM
}

/// 3DGS rendering output (autodiff enabled).
#[derive(Clone)]
pub struct Gaussian3dRenderOutputAutodiff<AB: AutodiffBackend> {
    /// 2D Colors in RGB space.
    ///
    /// The shape is `[I_y, I_x, 3]`.
    /// - `I_y`: Image height.
    /// - `I_x`: Image width.
    ///
    /// It is the rendered image.
    pub colors_rgb_2d: Tensor<AB, 3>,
    /// Its gradient is the gradient norm of the 2D positions.
    ///
    /// The gradient shape is `[P]`.
    /// - `P`: Point count.
    ///
    /// ## Usage
    ///
    /// ```ignore
    /// use burn::backend::autodiff::grads::Gradients;
    ///
    /// let mut grads: Gradients = todo!();
    ///
    /// let positions_2d_grad_norm =
    ///     positions_2d_grad_norm_ref.grad_remove(&mut grads);
    /// ```
    pub positions_2d_grad_norm_ref: Tensor<AB, 1>,
    /// Visible radii of 3D Gaussians.
    ///
    /// The shape is `[P]`.
    /// - `P`: Point count.
    pub radii: Tensor<AB::InnerBackend, 1, Int>,
}

impl Default for Gaussian3dRenderOptions {
    #[inline]
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(not(test))]
impl<B: Backend> fmt::Debug for Gaussian3dRenderOutput<B> {
    fn fmt(
        &self,
        f: &mut fmt::Formatter<'_>,
    ) -> fmt::Result {
        f.debug_struct(&format!("RenderOutput<{}>", B::name()))
            .field("colors_rgb_2d.dims()", &self.colors_rgb_2d.dims())
            .finish()
    }
}

#[cfg(not(test))]
impl<AB: AutodiffBackend> fmt::Debug for Gaussian3dRenderOutputAutodiff<AB> {
    fn fmt(
        &self,
        f: &mut fmt::Formatter<'_>,
    ) -> fmt::Result {
        let radii_dims = self.radii.dims();
        let positions_2d_grad_norm_dims = &radii_dims;

        f.debug_struct(&format!("RenderOutputAutodiff<{}>", AB::name()))
            .field("colors_rgb_2d.dims()", &self.colors_rgb_2d.dims())
            .field(
                "positions_2d_grad_norm.dims()",
                &positions_2d_grad_norm_dims,
            )
            .field("radii.dims()", &radii_dims)
            .finish()
    }
}
