pub use super::*;
pub use bytemuck::{Pod, Zeroable};

use burn::tensor::ops::{FloatTensorOps, IntTensorOps};
use bytemuck::bytes_of;

#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct Arguments {
    /// `(0 ~ 3)`
    pub colors_sh_degree_max: u32,
    /// `C_f (Constant)`
    pub filter_low_pass: f32,
    /// `f_x <- I_x / tan(fov_x / 2) / 2`
    pub focal_length_x: f32,
    /// `f_y <- I_x / tan(fov_y / 2) / 2`
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
    /// `tan(fov_x / 2) * (1 + C_f)`
    pub view_bound_x: f32,
    /// `tan(fov_y / 2) * (1 + C_f)`
    pub view_bound_y: f32,
}

#[derive(Clone, Debug)]
pub struct Inputs<R: JitRuntime, F: FloatElement> {
    /// `[P, 16, 3]`
    pub colors_sh: JitTensor<R, F>,
    /// `[P, 3]`
    pub positions_3d: JitTensor<R, F>,
    /// `[P, 4]`
    pub rotations: JitTensor<R, F>,
    /// `[P, 3]`
    pub scalings: JitTensor<R, F>,
    /// `[3]`
    pub view_position: JitTensor<R, F>,
    /// `[4, 4]`
    pub view_transform: JitTensor<R, F>,
}

#[derive(Clone, Debug)]
pub struct Outputs<R: JitRuntime, F: FloatElement, I: IntElement> {
    /// `[P, 3 (+ 1)]`
    pub colors_rgb_3d: JitTensor<R, F>,
    /// `[P, 2, 2]`
    pub conics: JitTensor<R, F>,
    /// `[P, 3 (+ 1), 3]`
    pub covariances_3d: JitTensor<R, F>,
    /// `[P]`
    pub depths: JitTensor<R, F>,
    /// `[P, 3 (+ 1)]`
    pub is_colors_rgb_3d_not_clamped: JitTensor<R, F>,
    /// `[P, 2]`
    pub positions_2d: JitTensor<R, F>,
    /// `[P, 2]`
    pub positions_3d_in_normalized: JitTensor<R, F>,
    /// `[P, 2]`
    pub positions_3d_in_normalized_clamped: JitTensor<R, F>,
    /// `[P]`
    pub radii: JitTensor<R, I>,
    /// `[P, 3 (+ 1), 3]`
    pub rotations_matrix: JitTensor<R, F>,
    /// `[P, 3 (+ 1), 3]`
    pub rotation_scalings: JitTensor<R, F>,
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
    let covariances_3d = JitBackend::<R, F, I>::float_empty(
        [point_count, 3 + 1, 3].into(),
        device,
    );
    let depths =
        JitBackend::<R, F, I>::float_empty([point_count].into(), device);
    let is_colors_rgb_3d_not_clamped =
        JitBackend::<R, F, I>::float_empty([point_count, 3 + 1].into(), device);
    let positions_2d =
        JitBackend::<R, F, I>::float_empty([point_count, 2].into(), device);
    let positions_3d_in_normalized =
        JitBackend::<R, F, I>::float_empty([point_count, 2].into(), device);
    let positions_3d_in_normalized_clamped =
        JitBackend::<R, F, I>::float_empty([point_count, 2].into(), device);
    let radii = JitBackend::<R, F, I>::int_empty([point_count].into(), device);
    let rotations_matrix = JitBackend::<R, F, I>::float_empty(
        [point_count, 3 + 1, 3].into(),
        device,
    );
    let rotation_scalings = JitBackend::<R, F, I>::float_empty(
        [point_count, 3 + 1, 3].into(),
        device,
    );
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
        CubeCount::Static(
            (arguments.point_count + GROUP_SIZE - 1) / GROUP_SIZE,
            1,
            1,
        ),
        vec![
            client.create(bytes_of(&arguments)).binding(),
            inputs.colors_sh.handle.binding(),
            inputs.positions_3d.handle.binding(),
            inputs.rotations.handle.binding(),
            inputs.scalings.handle.binding(),
            inputs.view_position.handle.binding(),
            inputs.view_transform.handle.binding(),
            colors_rgb_3d.handle.to_owned().binding(),
            conics.handle.to_owned().binding(),
            covariances_3d.handle.to_owned().binding(),
            depths.handle.to_owned().binding(),
            is_colors_rgb_3d_not_clamped.handle.to_owned().binding(),
            positions_2d.handle.to_owned().binding(),
            positions_3d_in_normalized.handle.to_owned().binding(),
            positions_3d_in_normalized_clamped
                .handle
                .to_owned()
                .binding(),
            radii.handle.to_owned().binding(),
            rotations_matrix.handle.to_owned().binding(),
            rotation_scalings.handle.to_owned().binding(),
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
        covariances_3d,
        depths,
        is_colors_rgb_3d_not_clamped,
        positions_2d,
        positions_3d_in_normalized,
        positions_3d_in_normalized_clamped,
        radii,
        rotations_matrix,
        rotation_scalings,
        tile_touched_counts,
        tiles_touched_max,
        tiles_touched_min,
        transforms_2d,
        view_directions,
        view_offsets,
    }
}
