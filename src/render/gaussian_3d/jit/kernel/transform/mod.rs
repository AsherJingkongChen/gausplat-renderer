//! Transforming the points.

pub use super::*;

use burn::tensor::ops::{FloatTensorOps, IntTensorOps};
use bytemuck::{bytes_of, Pod, Zeroable};

/// Arguments.
#[repr(C, align(16))]
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
    pub tile_count_x: i32,
    /// `I_y / T_y`
    pub tile_count_y: i32,
    /// `tan(Fov_x / 2) * (C_f + 1)`
    pub view_bound_x: f32,
    /// `tan(Fov_y / 2) * (C_f + 1)`
    pub view_bound_y: f32,
    /// Padding
    pub _padding_1: [u32; 2],
    /// `[3]`
    pub view_position: [f32; 3],
    /// Padding
    pub _padding_2: [u32; 1],
    /// `[3 (+ 1), 3 + 1]`
    pub view_transform: [[f32; 4]; 4],
}

/// Inputs.
#[derive(Clone, Debug)]
pub struct Inputs<R: JitRuntime> {
    /// `[P, 48]` <- `[P, 16, 3]`
    pub colors_sh: JitTensor<R>,
    /// `[P, 3]`
    pub positions_3d: JitTensor<R>,
    /// `[P, 4]`
    pub rotations: JitTensor<R>,
    /// `[P, 3]`
    pub scalings: JitTensor<R>,
}

/// Outputs.
#[derive(Clone, Debug)]
pub struct Outputs<R: JitRuntime> {
    /// `[P, 3]`
    pub colors_rgb_3d: JitTensor<R>,
    /// `[P, 3]`
    pub conics: JitTensor<R>,
    /// `[P]`
    pub depths: JitTensor<R>,
    /// `[P, 3]`
    pub is_colors_rgb_3d_not_clamped: JitTensor<R>,
    /// `[P, 4]`
    pub point_tile_bounds: JitTensor<R>,
    /// `[P, 2]`
    pub positions_2d: JitTensor<R>,
    /// `[P, 2]`
    pub positions_3d_in_normalized: JitTensor<R>,
    /// `[P]`
    pub radii: JitTensor<R>,
    /// `[P, 3, 3]`
    pub rotations_matrix: JitTensor<R>,
    /// `[P]`
    pub tile_touched_counts: JitTensor<R>,
}

/// `C_f`
pub const FILTER_LOW_PASS: f64 = 0.3;
/// Group size.
pub const GROUP_SIZE: u32 = 256;

/// Transforming the points.
pub fn main<R: JitRuntime, F: FloatElement, I: IntElement, B: BoolElement>(
    arguments: Arguments,
    inputs: Inputs<R>,
) -> Outputs<R> {
    impl_kernel_source!(Kernel, "kernel.wgsl");

    // Specifying the parameters

    let client = &inputs.colors_sh.client;
    let device = &inputs.colors_sh.device;
    // P
    let point_count = arguments.point_count as usize;

    let colors_rgb_3d =
        JitBackend::<R, F, I, B>::float_empty([point_count, 3].into(), device);
    let conics = JitBackend::<R, F, I, B>::float_empty([point_count, 3].into(), device);
    let depths = JitBackend::<R, F, I, B>::float_empty([point_count].into(), device);
    let is_colors_rgb_3d_not_clamped =
        JitBackend::<R, F, I, B>::float_empty([point_count, 3].into(), device);
    let point_tile_bounds =
        JitBackend::<R, F, I, B>::int_empty([point_count, 4].into(), device);
    let positions_2d =
        JitBackend::<R, F, I, B>::float_empty([point_count, 2].into(), device);
    let positions_3d_in_normalized =
        JitBackend::<R, F, I, B>::float_empty([point_count, 2].into(), device);
    let radii = JitBackend::<R, F, I, B>::int_empty([point_count].into(), device);
    let rotations_matrix =
        JitBackend::<R, F, I, B>::float_empty([point_count, 3, 3].into(), device);
    let tile_touched_counts =
        JitBackend::<R, F, I, B>::int_empty([point_count].into(), device);

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
            colors_rgb_3d.handle.to_owned().binding(),
            conics.handle.to_owned().binding(),
            depths.handle.to_owned().binding(),
            is_colors_rgb_3d_not_clamped.handle.to_owned().binding(),
            point_tile_bounds.handle.to_owned().binding(),
            positions_2d.handle.to_owned().binding(),
            positions_3d_in_normalized.handle.to_owned().binding(),
            radii.handle.to_owned().binding(),
            rotations_matrix.handle.to_owned().binding(),
            tile_touched_counts.handle.to_owned().binding(),
        ],
    );

    Outputs {
        colors_rgb_3d,
        conics,
        depths,
        is_colors_rgb_3d_not_clamped,
        point_tile_bounds,
        positions_2d,
        positions_3d_in_normalized,
        radii,
        rotations_matrix,
        tile_touched_counts,
    }
}
