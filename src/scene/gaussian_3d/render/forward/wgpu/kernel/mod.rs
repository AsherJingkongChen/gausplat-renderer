pub use burn_jit::{
    cubecl::KernelId,
    template::{KernelSource, SourceTemplate},
};
pub use bytemuck::{Pod, Zeroable};

impl_kernel_source!(Kernel1WgslSource, "./1.wgsl");
impl_kernel_source!(Kernel3WgslSource, "./3.wgsl");
impl_kernel_source!(Kernel5WgslSource, "./5.wgsl");
impl_kernel_source!(Kernel6WgslSource, "./6.wgsl");
impl_kernel_source!(KernelScanAddAdd, "./scan_add_add.wgsl");
impl_kernel_source!(KernelScanAddScan, "./scan_add_scan.wgsl");
impl_kernel_source!(KernelRadixSortScanLocal, "./radix_sort_scan_local.wgsl");
impl_kernel_source!(KernelRadixSortScatterKey, "./radix_sort_scatter_key.wgsl");
impl_kernel_source!(KernelRadixSort1, "./radix_sort_1.wgsl");

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
pub struct Kernel6Arguments {
    /// `I_x`
    pub image_size_x: u32,
    /// `I_y`
    pub image_size_y: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct KernelRadixSortArguments {
    /// `(0 ~ .., 2^R)`
    pub radix_bit_offset: u32,
}


macro_rules! impl_kernel_source {
    ($kernel:ident, $source_path:expr) => {
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
use impl_kernel_source;
