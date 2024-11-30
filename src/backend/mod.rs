/// Types and traits of JIT backend.
pub mod jit {
    pub use burn_jit::{BoolElement, FloatElement, IntElement, JitBackend, JitRuntime};
}

pub use burn::backend::wgpu::{self, WgpuDevice, WgpuRuntime};
pub use burn::{
    backend::autodiff,
    tensor::backend::{AutodiffBackend, Backend},
};
pub use jit::JitBackend;

/// The backend marker to enable autodiff.
pub type Autodiff<B> = autodiff::Autodiff<B>;
/// The JIT backend using [`WgpuRuntime`].
pub type Wgpu = JitBackend<WgpuRuntime, f32, i32, u32>;
