use burn::backend::{
    autodiff::Autodiff,
    wgpu::{AutoGraphicsApi, JitBackend, WgpuRuntime},
};

pub type Wgpu = JitBackend<WgpuRuntime<AutoGraphicsApi, f32, i32>>;

pub type WgpuAutodiff = Autodiff<Wgpu>;
