pub use super::*;
pub use bytemuck::{Pod, Zeroable};

use burn::tensor::ops::{FloatTensorOps, IntTensorOps};
use bytemuck::bytes_of;

#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct Arguments {
    /// `(0 ~ 3)`
    pub colors_sh_degree_max: u32,
    /// `f_x <- I_x / tan(Fov_x / 2) / 2`
    pub focal_length_x: f32,
    /// `f_y <- I_y / tan(Fov_y / 2) / 2`
    pub focal_length_y: f32,
    /// `I_x / 2`
    pub image_size_half_x: f32,
    /// `I_y / 2`
    pub image_size_half_y: f32,
    /// `P`
    pub point_count: u32,
    /// `I_x / T_x`
    pub tile_count_x: u32,
    /// `I_y / T_y`
    pub tile_count_y: u32,
    /// `tan(Fov_x / 2) * (C_f + 1)`
    pub view_bound_x: f32,
    /// `tan(Fov_y / 2) * (C_f + 1)`
    pub view_bound_y: f32,
}

#[derive(Clone, Debug)]
pub struct Inputs<R: JitRuntime, F: FloatElement> {
    /// `[P, 48] <- [P, 16, 3]`
    pub colors_sh: JitTensor<R, F>,
    /// `[P, 3]`
    pub positions_3d: JitTensor<R, F>,
    /// `[P, 4]`
    pub rotations: JitTensor<R, F>,
    /// `[P, 3]`
    pub scalings: JitTensor<R, F>,
    /// `[3 (+ 1), 3 + 1 + 1]`
    pub view_transform: JitTensor<R, F>,
}

#[derive(Clone, Debug)]
pub struct Outputs<R: JitRuntime, F: FloatElement, I: IntElement> {
    /// `[P, 3 (+ 1)]`
    pub colors_rgb_3d: JitTensor<R, F>,
    /// `[P, 2, 2]`
    pub conics: JitTensor<R, F>,
    /// `[P]`
    pub depths: JitTensor<R, F>,
    /// `[P, 3 (+ 1)]`
    pub is_colors_rgb_3d_not_clamped: JitTensor<R, F>,
    /// `[P, 2]`
    pub positions_2d: JitTensor<R, F>,
    /// `[P]`
    pub radii: JitTensor<R, I>,
    /// `[P]`
    pub tile_touched_counts: JitTensor<R, I>,
    /// `[P, 2]`
    pub tiles_touched_max: JitTensor<R, I>,
    /// `[P, 2]`
    pub tiles_touched_min: JitTensor<R, I>,
    /// `[P, 2, 3]`
    pub transforms_2d: JitTensor<R, F>,
    /// `[P, 3 (+ 1)]`
    pub view_directions: JitTensor<R, F>,
    /// `[P, 3 (+ 1)]`
    pub view_offsets: JitTensor<R, F>,
}

/// `C_f`
pub const FILTER_LOW_PASS: f64 = 0.3;
pub const GROUP_SIZE: u32 = 256;

/// Transforming the points.
pub fn main<R: JitRuntime, F: FloatElement, I: IntElement>(
    arguments: Arguments,
    inputs: Inputs<R, F>,
) -> Outputs<R, F, I> {
    impl_kernel_source!(Kernel, "kernel.wgsl");

    // Specifying the parameters

    let client = &inputs.colors_sh.client;
    let device = &inputs.colors_sh.device;
    // P
    let point_count = arguments.point_count as usize;

    let colors_rgb_3d =
        JitBackend::<R, F, I>::float_empty([point_count, 3 + 1].into(), device);
    let conics =
        JitBackend::<R, F, I>::float_empty([point_count, 2, 2].into(), device);
    let depths =
        JitBackend::<R, F, I>::float_empty([point_count].into(), device);
    let is_colors_rgb_3d_not_clamped =
        JitBackend::<R, F, I>::float_empty([point_count, 3 + 1].into(), device);
    let positions_2d =
        JitBackend::<R, F, I>::float_empty([point_count, 2].into(), device);
    let radii = JitBackend::<R, F, I>::int_empty([point_count].into(), device);
    let tile_touched_counts =
        JitBackend::<R, F, I>::int_empty([point_count].into(), device);
    let tiles_touched_max =
        JitBackend::<R, F, I>::int_empty([point_count, 2].into(), device);
    let tiles_touched_min =
        JitBackend::<R, F, I>::int_empty([point_count, 2].into(), device);
    let transforms_2d =
        JitBackend::<R, F, I>::float_empty([point_count, 2, 3].into(), device);
    let view_directions =
        JitBackend::<R, F, I>::float_empty([point_count, 3 + 1].into(), device);
    let view_offsets =
        JitBackend::<R, F, I>::float_empty([point_count, 3 + 1].into(), device);

    // Launching the kernel

    client.execute(
        Box::new(SourceKernel::new(
            Kernel,
            CubeDim {
                x: GROUP_SIZE,
                y: 1,
                z: 1,
            },
        )),
        CubeCount::Static(arguments.point_count.div_ceil(GROUP_SIZE), 1, 1),
        vec![
            client.create(bytes_of(&arguments)).binding(),
            inputs.colors_sh.handle.binding(),
            inputs.positions_3d.handle.binding(),
            inputs.rotations.handle.binding(),
            inputs.scalings.handle.binding(),
            inputs.view_transform.handle.binding(),
            colors_rgb_3d.handle.to_owned().binding(),
            conics.handle.to_owned().binding(),
            depths.handle.to_owned().binding(),
            is_colors_rgb_3d_not_clamped.handle.to_owned().binding(),
            positions_2d.handle.to_owned().binding(),
            radii.handle.to_owned().binding(),
            tile_touched_counts.handle.to_owned().binding(),
            tiles_touched_max.handle.to_owned().binding(),
            tiles_touched_min.handle.to_owned().binding(),
            transforms_2d.handle.to_owned().binding(),
            view_directions.handle.to_owned().binding(),
            view_offsets.handle.to_owned().binding(),
        ],
    );

    Outputs {
        colors_rgb_3d,
        conics,
        depths,
        is_colors_rgb_3d_not_clamped,
        positions_2d,
        radii,
        tile_touched_counts,
        tiles_touched_max,
        tiles_touched_min,
        transforms_2d,
        view_directions,
        view_offsets,
    }
}
