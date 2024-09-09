use burn::backend::wgpu::{KernelSource, SourceTemplate};
use burn_jit::cubecl::KernelId;
use bytemuck::{Pod, Zeroable};

pub struct Kernel1WgslSource;
pub struct Kernel2WgslSource;

#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct Kernel1Arguments {
    /// `I_x`
    pub image_size_x: u32,
    /// `I_y`
    pub image_size_y: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct Kernel2Arguments {
    pub colors_sh_degree_max: u32,
    pub focal_length_x: f32,
    pub focal_length_y: f32,
    /// `I_x / 2`
    pub image_size_half_x: f32,
    /// `I_y / 2`
    pub image_size_half_y: f32,
    /// `P`
    pub point_count: u32,
}

impl KernelSource for Kernel1WgslSource {
    fn source(&self) -> SourceTemplate {
        SourceTemplate::new(include_str!("./1.wgsl"))
    }

    fn id(&self) -> KernelId {
        KernelId::new::<Self>()
    }
}

impl KernelSource for Kernel2WgslSource {
    fn source(&self) -> SourceTemplate {
        SourceTemplate::new(include_str!("./2.wgsl"))
    }

    fn id(&self) -> KernelId {
        KernelId::new::<Self>()
    }
}
