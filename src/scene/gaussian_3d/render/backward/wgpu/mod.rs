mod kernel;

pub use super::*;

use crate::consts::render::*;
use burn::{
    backend::wgpu::{
        into_contiguous, Kernel, SourceKernel, WorkGroup, WorkgroupSize,
    },
    tensor::ops::{ActivationOps, IntTensor, ModuleOps},
};
use bytemuck::{bytes_of, cast_slice, cast_slice_mut, from_bytes};

pub fn render_gaussian_3d_scene(
    grad: FloatTensor<Wgpu, 3>,
    state: forward::RendererState<Wgpu>,
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
    // ([P, 16, 3], P)
    let (client, colors_sh, device, point_count) = {
        let c = state.colors_sh;
        (c.client, c.handle, c.device, c.shape.dims[0])
    };

    // /// `[P, 16, 3]`
    // colors_sh_grad: B::FloatTensorPrimitive<3>,
    // /// `[P, 1]`
    // opacities_3d_grad: B::FloatTensorPrimitive<1>,
    // /// `[P]`
    // positions_2d_grad_norm: B::FloatTensorPrimitive<1>,
    // /// `[P, 3]`
    // positions_3d_grad: B::FloatTensorPrimitive<2>,
    // /// `[P, 4]`
    // rotations_grad: B::FloatTensorPrimitive<2>,
    // /// `[P, 3]`
    // scalings_grad: B::FloatTensorPrimitive<2>,

    // [P, 16, 3]
    let colors_sh_grad = Tensor::<Wgpu, 3>::zeros([point_count, 16, 3], &device).into_primitive();
    // [P, 1]
    let opacities_3d_grad = Tensor::<Wgpu, 2>::zeros([point_count, 1], &device).into_primitive();
    // [P]
    let positions_2d_grad_norm = Tensor::<Wgpu, 1>::zeros([point_count], &device).into_primitive();
    // [P, 3]
    let positions_3d_grad = Tensor::<Wgpu, 2>::zeros([point_count, 3], &device).into_primitive();
    // [P, 4]
    let rotations_grad = Tensor::<Wgpu, 2>::zeros([point_count, 4], &device).into_primitive();
    // [P, 3]
    let scalings_grad = Tensor::<Wgpu, 2>::zeros([point_count, 3], &device).into_primitive();

    backward::RendererOutput {
        colors_sh_grad,
        opacities_3d_grad,
        positions_2d_grad_norm,
        positions_3d_grad,
        rotations_grad,
        scalings_grad,
    }
}
