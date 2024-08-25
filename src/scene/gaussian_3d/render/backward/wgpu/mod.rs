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
    state: backward::RendererInput<Wgpu>,
    grad: FloatTensor<Wgpu, 3>,
) -> backward::RendererOutput<Wgpu> {
    // P
    let point_count = state.point_count;
    // [P, 16, 3]
    let (client, colors_sh, device) = {
        let c = state.colors_sh;
        (c.client, c.handle, c.device)
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
    let opacities_grad = Tensor::<Wgpu, 2>::zeros([point_count, 1], &device).into_primitive();
    // [P, 3]
    let positions_grad = Tensor::<Wgpu, 2>::zeros([point_count, 3], &device).into_primitive();
    // [P]
    let positions_2d_grad_norm = Tensor::<Wgpu, 1>::zeros([point_count], &device).into_primitive();
    // [P, 4]
    let rotations_grad = Tensor::<Wgpu, 2>::zeros([point_count, 4], &device).into_primitive();
    // [P, 3]
    let scalings_grad = Tensor::<Wgpu, 2>::zeros([point_count, 3], &device).into_primitive();

    backward::RendererOutput {
        colors_sh_grad,
        opacities_grad,
        positions_grad,
        positions_2d_grad_norm,
        rotations_grad,
        scalings_grad,
    }
}
