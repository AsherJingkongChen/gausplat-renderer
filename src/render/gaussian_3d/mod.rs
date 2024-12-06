//! 3DGS rendering implementation.
//!
//! For more information, see the [survey](https://arxiv.org/abs/2401.03890).

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

/// Renderer for 3DGS.
pub trait Gaussian3dRenderer<B: Backend>: 'static + Send + Sized + fmt::Debug {
    /// Rendering (forward).
    fn render_forward(
        input: forward::RenderInput<B>,
        view: &View,
        options: &Gaussian3dRenderOptions,
    ) -> Result<forward::RenderOutput<B>, Error>;

    /// Rendering (backward).
    fn render_backward(
        state: backward::RenderInput<B>,
        colors_rgb_2d_grad: B::FloatTensorPrimitive,
    ) -> backward::RenderOutput<B>;
}

/// Options for rendering 3DGS.
#[derive(Config, Copy, Debug, PartialEq, Record)]
pub struct Gaussian3dRenderOptions {
    #[config(default = "SH_DEGREE_MAX")]
    /// It should be no more than [`SH_DEGREE_MAX`].
    pub colors_sh_degree_max: u32,
}

/// Outputs for rendering 3DGS.
#[derive(Clone)]
pub struct Gaussian3dRenderOutput<B: Backend> {
    /// `[I_y, I_x, 3]`
    pub colors_rgb_2d: Tensor<B, 3>,
    // TODO: THM
}

/// Outputs for rendering 3DGS (autodiff enabled).
#[derive(Clone)]
pub struct Gaussian3dRenderOutputAutodiff<AB: AutodiffBackend> {
    /// `[I_y, I_x, 3]`
    pub colors_rgb_2d: Tensor<AB, 3>,

    /// The shape of gradient is `[P]`
    ///
    /// ## Usage
    ///
    /// ```plaintext
    /// use burn::backend::autodiff::grads::Gradients;
    ///
    /// let mut grads: Gradients = todo!();
    ///
    /// let positions_2d_grad_norm =
    ///     positions_2d_grad_norm_ref.grad_remove(&mut grads);
    /// ```
    pub positions_2d_grad_norm_ref: Tensor<AB, 1>,

    /// `[P]`
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
