mod wgpu;

pub use super::*;

use crate::consts::render::*;
use burn::{
    backend::wgpu::{
        into_contiguous, Kernel, SourceKernel, WorkGroup, WorkgroupSize,
    },
    tensor::ops::{ActivationOps, FloatTensor, IntTensor, ModuleOps},
};
use bytemuck::{bytes_of, cast_slice, cast_slice_mut, from_bytes};

#[derive(Clone, Debug)]
pub struct RendererOutput<B: Backend> {
    pub _b: std::marker::PhantomData<B>,
}

pub(super) fn render_gaussian_3d_scene_wgpu(
    output: forward::RendererOutput<Wgpu>,
    options: &RenderOptions,
) -> backward::RendererOutput<Wgpu> {
    use wgpu::*;

    <Wgpu as ActivationOps<Wgpu>>::relu_backward::<2>;
    <Wgpu as ModuleOps<Wgpu>>::conv2d;
    <Autodiff<Wgpu> as ModuleOps<Autodiff<Wgpu>>>::conv2d;
    RendererOutput { _b: std::marker::PhantomData }
}
