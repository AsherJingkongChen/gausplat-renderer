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

#[derive(Clone, Debug)]
pub struct RendererOutput<B: Backend> {
    pub _b: std::marker::PhantomData<B>,
}

pub fn render_gaussian_3d_scene(
    output: forward::RendererOutput<Wgpu>,
    options: &RendererOptions,
) -> backward::RendererOutput<Wgpu> {
    <Wgpu as ActivationOps<Wgpu>>::relu_backward::<2>;
    <Wgpu as ModuleOps<Wgpu>>::conv2d;
    <Autodiff<Wgpu> as ModuleOps<Autodiff<Wgpu>>>::conv2d;
    backward::RendererOutput {
        _b: std::marker::PhantomData,
    }
}