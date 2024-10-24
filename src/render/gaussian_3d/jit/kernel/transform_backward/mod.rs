pub use super::*;
pub use bytemuck::{Pod, Zeroable};

use burn::tensor::ops::FloatTensorOps;
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
}

#[derive(Clone, Debug)]
pub struct Inputs<R: JitRuntime, F: FloatElement, I: IntElement> {
    /// `[P, 3 (+ 1)]`
    pub colors_rgb_3d_grad: JitTensor<R, F>,
    /// `[P, 48] <- [P, 16, 3]`
    pub colors_sh: JitTensor<R, F>,
    /// `[P, 2, 2]`
    pub conics: JitTensor<R, F>,
    /// `[P, 2, 2]`
    pub conics_grad: JitTensor<R, F>,
    /// `[P, 3 (+ 1)]`
    pub covariances_3d: JitTensor<R, F>,
    /// `[P]`
    pub depths: JitTensor<R, F>,
    /// `[P, 3 (+ 1)]`
    pub is_colors_rgb_3d_not_clamped: JitTensor<R, F>,
    /// `[P, 2]`
    pub positions_2d_grad: JitTensor<R, F>,
    /// `[P, 2]`
    pub positions_3d_in_normalized: JitTensor<R, F>,
    /// `[P, 2]`
    pub positions_3d_in_normalized_clamped: JitTensor<R, F>,
    /// `[P]`
    pub radii: JitTensor<R, I>,
    /// `[P, 4]`
    pub rotations: JitTensor<R, F>,
    /// `[P, 3 (+ 1), 3]`
    pub rotations_matrix: JitTensor<R, F>,
    /// `[P, 3 (+ 1), 3]`
    pub rotation_scalings: JitTensor<R, F>,
    /// `[P, 3]`
    pub scalings: JitTensor<R, F>,
    /// `[P, 2, 3]`
    pub transforms_2d: JitTensor<R, F>,
    /// `[P, 3 (+ 1)]`
    pub view_directions: JitTensor<R, F>,
    /// `[P, 3 (+ 1)]`
    pub view_offsets: JitTensor<R, F>,
    /// `[3 (+ 1), 3]`
    pub view_rotation: JitTensor<R, F>,
}

#[derive(Clone, Debug)]
pub struct Outputs<R: JitRuntime, F: FloatElement> {
    /// `[P, 48] <- [P, 16, 3]`
    pub colors_sh_grad: JitTensor<R, F>,
    /// `[P]`
    pub positions_2d_grad_norm: JitTensor<R, F>,
    /// `[P, 3]`
    pub positions_3d_grad: JitTensor<R, F>,
    /// `[P, 4]`
    pub rotations_grad: JitTensor<R, F>,
    /// `[P, 3]`
    pub scalings_grad: JitTensor<R, F>,
}

/// `C_f`
pub const FILTER_LOW_PASS: f64 = 0.3;
pub const GROUP_SIZE: u32 = 256;

/// Transforming the points.
pub fn main<R: JitRuntime, F: FloatElement, I: IntElement>(
    arguments: Arguments,
    inputs: Inputs<R, F, I>,
) -> Outputs<R, F> {
    impl_kernel_source!(Kernel, "kernel.wgsl");

    // Specifying the parameters

    let client = &inputs.colors_rgb_3d_grad.client;
    let device = &inputs.colors_rgb_3d_grad.device;
    // P
    let point_count = arguments.point_count as usize;

    // [P, 48] <- [P, 16, 3]
    let colors_sh_grad =
        JitBackend::<R, F, I>::float_zeros([point_count, 48].into(), device);
    let positions_2d_grad_norm =
        JitBackend::<R, F, I>::float_zeros([point_count].into(), device);
    let positions_3d_grad =
        JitBackend::<R, F, I>::float_zeros([point_count, 3].into(), device);
    let rotations_grad =
        JitBackend::<R, F, I>::float_zeros([point_count, 4].into(), device);
    let scalings_grad =
        JitBackend::<R, F, I>::float_zeros([point_count, 3].into(), device);

    client.execute(
        Box::new(SourceKernel::new(
            Kernel,
            CubeDim {
                x: GROUP_SIZE,
                y: 1,
                z: 1,
            },
        )),
        CubeCount::Static((point_count as u32).div_ceil(GROUP_SIZE), 1, 1),
        vec![
            client.create(bytes_of(&arguments)).binding(),
            inputs.colors_rgb_3d_grad.handle.binding(),
            inputs.colors_sh.handle.binding(),
            inputs.conics.handle.binding(),
            inputs.conics_grad.handle.binding(),
            inputs.covariances_3d.handle.binding(),
            inputs.depths.handle.binding(),
            inputs.is_colors_rgb_3d_not_clamped.handle.binding(),
            inputs.positions_2d_grad.handle.binding(),
            inputs.positions_3d_in_normalized.handle.binding(),
            inputs.positions_3d_in_normalized_clamped.handle.binding(),
            inputs.radii.handle.binding(),
            inputs.rotations.handle.binding(),
            inputs.rotations_matrix.handle.binding(),
            inputs.rotation_scalings.handle.binding(),
            inputs.scalings.handle.binding(),
            inputs.transforms_2d.handle.binding(),
            inputs.view_directions.handle.binding(),
            inputs.view_offsets.handle.binding(),
            inputs.view_rotation.handle.binding(),
            colors_sh_grad.handle.to_owned().binding(),
            positions_2d_grad_norm.handle.to_owned().binding(),
            positions_3d_grad.handle.to_owned().binding(),
            rotations_grad.handle.to_owned().binding(),
            scalings_grad.handle.to_owned().binding(),
        ],
    );

    Outputs {
        colors_sh_grad,
        positions_2d_grad_norm,
        positions_3d_grad,
        rotations_grad,
        scalings_grad,
    }
}
