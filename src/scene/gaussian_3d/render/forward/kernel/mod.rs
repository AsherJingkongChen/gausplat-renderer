use burn::backend::wgpu::kernel_wgsl;
use bytemuck::{Pod, Zeroable};
use derive_new::new;

kernel_wgsl!(
    RenderGaussian3dForward1,
    "./render_gaussian_3d_forward_1.wgsl"
);
kernel_wgsl!(
    RenderGaussian3dForward2,
    "./render_gaussian_3d_forward_2.wgsl"
);
kernel_wgsl!(
    RenderGaussian3dForward3,
    "./render_gaussian_3d_forward_3.wgsl"
);
kernel_wgsl!(
    RenderGaussian3dForward5,
    "./render_gaussian_3d_forward_5.wgsl"
);
kernel_wgsl!(
    RenderGaussian3dForward6,
    "./render_gaussian_3d_forward_6.wgsl"
);

#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
pub(super) struct RenderGaussian3dForward1Arguments {
    pub colors_sh_degree_max: u32,
    pub filter_low_pass: f32,
    pub focal_length_x: f32,
    pub focal_length_y: f32,
    /// `I_X`
    pub image_size_x: u32,
    /// `I_Y`
    pub image_size_y: u32,
    /// `I_X / 2`
    pub image_size_half_x: f32,
    /// `I_Y / 2`
    pub image_size_half_y: f32,
    /// `P`
    pub point_count: u32,
    /// `I_X / T_X`
    pub tile_count_x: u32,
    /// `I_Y / T_Y`
    pub tile_count_y: u32,
    /// `T_X`
    pub tile_size_x: u32,
    /// `T_Y`
    pub tile_size_y: u32,
    pub view_bound_x: f32,
    pub view_bound_y: f32,
}

#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
pub(super) struct RenderGaussian3dForward2Arguments {
    /// `P`
    pub point_count: u32,
}

#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
pub(super) struct RenderGaussian3dForward3Arguments {
    /// `P`
    pub point_count: u32,
    /// `I_X / T_X`
    pub tile_count_x: u32,
}

#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
pub(super) struct RenderGaussian3dForward5Arguments {
    /// `T`
    pub tile_touched_count: u32,
}

#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
pub(super) struct RenderGaussian3dForward6Arguments {
    /// `I_X`
    pub image_size_x: u32,
    /// `I_Y`
    pub image_size_y: u32,
}
