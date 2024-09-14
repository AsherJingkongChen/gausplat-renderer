use burn_jit::{
    cubecl::KernelId,
    template::{KernelSource, SourceTemplate},
};
use bytemuck::{Pod, Zeroable};

pub struct Kernel1WgslSource;
pub struct Kernel3WgslSource;
pub struct Kernel5WgslSource;
pub struct Kernel6WgslSource;

#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct Kernel1Arguments {
    pub colors_sh_degree_max: u32,
    pub filter_low_pass: f32,
    pub focal_length_x: f32,
    pub focal_length_y: f32,
    /// `I_x`
    pub image_size_x: u32,
    /// `I_y`
    pub image_size_y: u32,
    /// `I_x * 0.5 - 0.5`
    pub image_size_half_x: f32,
    /// `I_y * 0.5 - 0.5`
    pub image_size_half_y: f32,
    /// `P`
    pub point_count: u32,
    /// `I_x / T_x`
    pub tile_count_x: u32,
    /// `I_y / T_y`
    pub tile_count_y: u32,
    /// `T_x`
    pub tile_size_x: u32,
    /// `T_y`
    pub tile_size_y: u32,
    pub view_bound_x: f32,
    pub view_bound_y: f32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct Kernel3Arguments {
    /// `P`
    pub point_count: u32,
    /// `I_x / T_x`
    pub tile_count_x: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct Kernel5Arguments {
    /// `T`
    pub tile_touched_count: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct Kernel6Arguments {
    /// `I_x`
    pub image_size_x: u32,
    /// `I_y`
    pub image_size_y: u32,
}

impl KernelSource for Kernel1WgslSource {
    fn source(&self) -> SourceTemplate {
        SourceTemplate::new(include_str!("./1.wgsl"))
    }

    fn id(&self) -> KernelId {
        KernelId::new::<Self>()
    }
}

impl KernelSource for Kernel3WgslSource {
    fn source(&self) -> SourceTemplate {
        SourceTemplate::new(include_str!("./3.wgsl"))
    }

    fn id(&self) -> KernelId {
        KernelId::new::<Self>()
    }
}

impl KernelSource for Kernel5WgslSource {
    fn source(&self) -> SourceTemplate {
        SourceTemplate::new(include_str!("./5.wgsl"))
    }

    fn id(&self) -> KernelId {
        KernelId::new::<Self>()
    }
}

impl KernelSource for Kernel6WgslSource {
    fn source(&self) -> SourceTemplate {
        SourceTemplate::new(include_str!("./6.wgsl"))
    }

    fn id(&self) -> KernelId {
        KernelId::new::<Self>()
    }
}
