pub use burn::backend::wgpu::{WgpuDevice, WgpuRuntime};

use burn::backend::autodiff;

pub type Autodiff<B> = autodiff::Autodiff<B>;
pub type Wgpu = burn_jit::JitBackend<WgpuRuntime, f32, i32>;
