//! 3DGS rendering context (backward).

pub use super::*;

/// Rendering inputs (backward).
#[derive(Clone, Debug)]
pub struct RenderInput<B: Backend> {
    /// The shape is `[P, 3]`
    pub colors_rgb_3d: B::FloatTensorPrimitive,
    /// The shape is `[P, M * 3]` <- `[P, M, 3]`
    pub colors_sh: B::FloatTensorPrimitive,
    /// `(0 ~ 3)`
    pub colors_sh_degree_max: u32,
    /// The shape is `[P, 3]`
    pub conics: B::FloatTensorPrimitive,
    /// The shape is `[P]`
    pub depths: B::FloatTensorPrimitive,
    /// `f_x <- I_x / tan(Fov_x / 2) / 2`
    pub focal_length_x: f32,
    /// `f_y <- I_y / tan(Fov_y / 2) / 2`
    pub focal_length_y: f32,
    /// `I_x / 2`
    pub image_size_half_x: f32,
    /// `I_y / 2`
    pub image_size_half_y: f32,
    /// `I_x`
    pub image_size_x: u32,
    /// `I_y`
    pub image_size_y: u32,
    /// The shape is `[P, 3]`
    pub is_colors_rgb_3d_not_clamped: B::FloatTensorPrimitive,
    /// The shape is `[P, 1]`
    pub opacities_3d: B::FloatTensorPrimitive,
    /// `P`
    pub point_count: u32,
    /// The shape is `[T]`
    pub point_indices: B::IntTensorPrimitive,
    /// The shape is `[I_y, I_x]`
    pub point_rendered_counts: B::IntTensorPrimitive,
    /// The shape is `[P, 2]`
    pub positions_2d: B::FloatTensorPrimitive,
    /// The shape is `[P, 3]`
    pub positions_3d: B::FloatTensorPrimitive,
    /// The shape is `[P, 2]`
    pub positions_3d_in_normalized: B::FloatTensorPrimitive,
    /// The shape is `[P]`
    pub radii: B::IntTensorPrimitive,
    /// The shape is `[P, 4]`
    pub rotations: B::FloatTensorPrimitive,
    /// The shape is `[P, 3, 3]`
    pub rotations_matrix: B::FloatTensorPrimitive,
    /// The shape is `[P, 3]`
    pub scalings: B::FloatTensorPrimitive,
    /// `I_x / T_x`
    pub tile_count_x: u32,
    /// `I_y / T_y`
    pub tile_count_y: u32,
    /// The shape is `[I_y / T_y, I_x / T_x, 2]`
    pub tile_point_ranges: B::IntTensorPrimitive,
    /// The shape is `[I_y, I_x]`
    pub transmittances: B::FloatTensorPrimitive,
    /// `tan(Fov_x / 2) * (C_f + 1)`
    pub view_bound_x: f32,
    /// `tan(Fov_y / 2) * (C_f + 1)`
    pub view_bound_y: f32,
    /// `[3]`
    pub view_position: [f32; 3],
    /// `[3 (+ 1), 3 + 1]`
    pub view_transform: [[f32; 4]; 4],
}

/// Outputs for rendering (backward).
#[derive(Clone, Debug)]
pub struct RenderOutput<B: Backend> {
    /// The shape is `[P, M * 3]` <- `[P, M, 3]`
    ///
    /// It is the gradient of `colors_rgb_2d` with respect to `colors_sh`.
    pub colors_sh_grad: B::FloatTensorPrimitive,
    /// The shape is `[P, 1]`
    ///
    /// It is the gradient of `colors_rgb_2d` with respect to `colors_sh_degree_max`.
    pub opacities_grad: B::FloatTensorPrimitive,
    /// The shape is `[P]`
    ///
    /// It is the gradient norm of the 2D positions.
    pub positions_2d_grad_norm: B::FloatTensorPrimitive,
    /// The shape is `[P, 3]`
    ///
    /// It is the gradient of `colors_rgb_2d` with respect to `positions_3d`.
    pub positions_grad: B::FloatTensorPrimitive,
    /// The shape is `[P, 4]`
    ///
    /// It is the gradient of `colors_rgb_2d` with respect to `rotations`.
    pub rotations_grad: B::FloatTensorPrimitive,
    /// The shape is `[P, 3]`
    ///
    /// It is the gradient of `colors_rgb_2d` with respect to `scalings`.
    pub scalings_grad: B::FloatTensorPrimitive,
}
