use burn::backend::wgpu::{KernelSource, SourceTemplate};
use bytemuck::{Pod, Zeroable};

pub struct Kernel1WgslSource;

#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct Kernel1Arguments {}

impl KernelSource for Kernel1WgslSource {
    fn source(&self) -> SourceTemplate {
        SourceTemplate::new(include_str!("./1.wgsl"))
    }
}
