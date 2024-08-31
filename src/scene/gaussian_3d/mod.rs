pub mod config;
pub mod property;
pub mod render;

pub use crate::preset::backend;
pub use burn::{
    module::{Module, Param},
    tensor::{backend::Backend, Tensor, TensorData},
};
pub use config::*;

use backend::*;
use std::fmt;

#[derive(Module)]
pub struct Gaussian3dScene<B: Backend> {
    /// `[P, 16, 3]`
    pub colors_sh: Param<Tensor<B, 3>>,
    /// `[P, 1]`
    pub opacities: Param<Tensor<B, 2>>,
    /// `[P, 3]`
    pub positions: Param<Tensor<B, 2>>,
    /// `[P, 4]`
    pub rotations: Param<Tensor<B, 2>>,
    /// `[P, 3]`
    pub scalings: Param<Tensor<B, 2>>,
}

impl render::Gaussian3dRenderer for Gaussian3dScene<Wgpu> {
    type Backend = Wgpu;

    fn render_forward(
        input: render::forward::RenderInput<Self::Backend>,
        view: &sparse_view::View,
        options: render::RenderOptions,
    ) -> render::forward::RenderOutput<Self::Backend> {
        render::forward::wgpu::render_gaussian_3d_scene(input, view, options)
    }

    fn render_backward(
        state: render::backward::RenderInput<Self::Backend>,
        colors_rgb_2d_grad: <Self::Backend as Backend>::FloatTensorPrimitive<3>,
    ) -> render::backward::RenderOutput<Self::Backend> {
        render::backward::wgpu::render_gaussian_3d_scene(
            state,
            colors_rgb_2d_grad,
        )
    }
}

impl render::Gaussian3dRenderer for Gaussian3dScene<Autodiff<Wgpu>> {
    type Backend = Wgpu;

    fn render_forward(
        input: render::forward::RenderInput<Self::Backend>,
        view: &sparse_view::View,
        options: render::RenderOptions,
    ) -> render::forward::RenderOutput<Self::Backend> {
        render::forward::wgpu::render_gaussian_3d_scene(input, view, options)
    }

    fn render_backward(
        state: render::backward::RenderInput<Self::Backend>,
        colors_rgb_2d_grad: <Self::Backend as Backend>::FloatTensorPrimitive<3>,
    ) -> render::backward::RenderOutput<Self::Backend> {
        render::backward::wgpu::render_gaussian_3d_scene(
            state,
            colors_rgb_2d_grad,
        )
    }
}

impl<B: Backend> fmt::Debug for Gaussian3dScene<B> {
    fn fmt(
        &self,
        f: &mut fmt::Formatter,
    ) -> fmt::Result {
        f.debug_struct("Gaussian3dScene")
            .field("devices", &self.devices())
            .field("colors_sh.dims()", &self.colors_sh.dims())
            .field("opacities.dims()", &self.opacities.dims())
            .field("positions.dims()", &self.positions.dims())
            .field("rotations.dims()", &self.rotations.dims())
            .field("scalings.dims()", &self.scalings.dims())
            .finish()
    }
}

impl<B: Backend> Default for Gaussian3dScene<B> {
    fn default() -> Self {
        Gaussian3dSceneConfig::default().into()
    }
}
