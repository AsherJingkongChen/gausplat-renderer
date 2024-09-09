pub use burn::backend::wgpu::WgpuDevice;

use burn::backend::{autodiff, wgpu};

pub type Autodiff<B> = autodiff::Autodiff<B>;
pub type Wgpu = burn_jit::JitBackend<wgpu::WgpuRuntime, f32, i32>;
