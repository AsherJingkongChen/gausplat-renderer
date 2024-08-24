mod kernel;

pub use super::*;

use crate::consts::render::*;
use burn::{
    backend::wgpu::{
        into_contiguous, Kernel, SourceKernel, WorkGroup, WorkgroupSize,
    },
    tensor::ops::{ActivationOps, FloatTensor, IntTensor, ModuleOps},
};
use bytemuck::{bytes_of, cast_slice, cast_slice_mut, from_bytes};

pub fn render_gaussian_3d_scene(
    output: forward::RendererOutput<Wgpu>,
    view: &sparse_view::View,
    options: &RendererOptions,
) -> backward::RendererOutput<Wgpu> {
    // I_X
    let image_size_x = view.image_width as usize;
    // I_Y
    let image_size_y = view.image_height as usize;
    let focal_length_x =
        (image_size_x as f64 / (view.field_of_view_x / 2.0).tan() / 2.0) as f32;
    let focal_length_y =
        (image_size_y as f64 / (view.field_of_view_y / 2.0).tan() / 2.0) as f32;
    // I_X / 2.0
    let image_size_half_x = (image_size_x as f64 / 2.0) as f32;
    // I_Y / 2.0
    let image_size_half_y = (image_size_y as f64 / 2.0) as f32;
    

    // <Wgpu as ActivationOps<Wgpu>>::relu_backward::<2>;
    // <Wgpu as ModuleOps<Wgpu>>::conv2d;
    // <Autodiff<Wgpu> as ModuleOps<Autodiff<Wgpu>>>::conv2d;
    backward::RendererOutput {
        _b: std::marker::PhantomData,
    }
}
