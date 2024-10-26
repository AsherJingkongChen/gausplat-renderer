pub use super::*;

#[derive(Clone, Debug)]
pub struct RenderInput<B: Backend> {
    /// `[P, 3 (+ 1)]`
    pub colors_rgb_3d: B::FloatTensorPrimitive,
    /// `[P, 48] <- [P, 16, 3]`
    pub colors_sh: B::FloatTensorPrimitive,
    /// `(0 ~ 3)`
    pub colors_sh_degree_max: u32,
    /// `[P, 2, 2]`
    pub conics: B::FloatTensorPrimitive,
    /// `[P]`
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
    /// `[P, 3 (+ 1)]`
    pub is_colors_rgb_3d_not_clamped: B::FloatTensorPrimitive,
    /// `[P]`
    pub opacities_3d: B::FloatTensorPrimitive,
    /// `P`
    pub point_count: u32,
    /// `[T]`
    pub point_indices: B::IntTensorPrimitive,
    /// `[I_y, I_x]`
    pub point_rendered_counts: B::IntTensorPrimitive,
    /// `[P, 2]`
    pub positions_2d: B::FloatTensorPrimitive,
    /// `[P, 3]`
    pub positions_3d: B::FloatTensorPrimitive,
    /// `[P]`
    pub radii: B::IntTensorPrimitive,
    /// `[P, 4]`
    pub rotations: B::FloatTensorPrimitive,
    /// `[P, 3]`
    pub scalings: B::FloatTensorPrimitive,
    /// `I_x / T_x`
    pub tile_count_x: u32,
    /// `I_y / T_y`
    pub tile_count_y: u32,
    /// `[I_y / T_y, I_x / T_x, 2]`
    pub tile_point_ranges: B::IntTensorPrimitive,
    /// `[I_y, I_x]`
    pub transmittances: B::FloatTensorPrimitive,
    /// `[3 (+ 1), 3 + 1 + 1]`
    pub view_transform: B::FloatTensorPrimitive,
    /// `tan(Fov_x / 2) * (C_f + 1)`
    pub view_bound_x: f32,
    /// `tan(Fov_y / 2) * (C_f + 1)`
    pub view_bound_y: f32,
}

#[derive(Clone, Debug)]
pub struct RenderOutput<B: Backend> {
    /// `[P, 48] <- [P, 16, 3]`
    pub colors_sh_grad: B::FloatTensorPrimitive,
    /// `[P]`
    pub opacities_grad: B::FloatTensorPrimitive,
    /// `[P]`
    pub positions_2d_grad_norm: B::FloatTensorPrimitive,
    /// `[P, 3]`
    pub positions_grad: B::FloatTensorPrimitive,
    /// `[P, 4]`
    pub rotations_grad: B::FloatTensorPrimitive,
    /// `[P, 3]`
    pub scalings_grad: B::FloatTensorPrimitive,
}
