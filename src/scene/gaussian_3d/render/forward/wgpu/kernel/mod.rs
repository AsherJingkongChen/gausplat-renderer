pub mod rank;
pub mod rasterize;
pub mod scan;
pub mod segment;
pub mod sort;
pub mod transform;

pub use burn_jit::{
    cubecl::KernelId,
    element::{FloatElement, IntElement},
    template::{KernelSource, SourceTemplate},
    tensor::JitTensor,
    JitRuntime,
};

use burn_jit::{
    cubecl::{CubeCount, CubeDim},
    template::SourceKernel,
    JitBackend,
};

macro_rules! impl_kernel_source {
    ($kernel: ident, $source_path: expr) => {
        #[derive(Clone, Copy, Debug, Default)]
        pub struct $kernel;

        impl KernelSource for $kernel {
            fn source(&self) -> SourceTemplate {
                SourceTemplate::new(include_str!($source_path))
            }

            fn id(&self) -> KernelId {
                KernelId::new::<Self>()
            }
        }
    };
}

pub(crate) use impl_kernel_source;
