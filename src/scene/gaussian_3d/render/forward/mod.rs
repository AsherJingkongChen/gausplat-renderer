pub(super) mod wgpu;

pub use super::*;

#[derive(Clone, Debug)]
pub struct RendererOutput<B: Backend> {
    /// `[I_Y, I_X, 3]`
    pub colors_rgb_2d: B::FloatTensorPrimitive<3>,
    /// `[P, 3 (+ 1)]`
    pub colors_rgb_3d: B::FloatTensorPrimitive<2>,
    /// `[P, 16, 3]`
    pub colors_sh: B::FloatTensorPrimitive<3>,
    /// `[P, 2, 2]`
    pub conics: B::FloatTensorPrimitive<3>,
    /// `[P, 3 (+ 1), 3]`
    pub covariances_3d: B::FloatTensorPrimitive<3>,
    /// `I_X`
    pub image_size_x: u32,
    /// `I_Y`
    pub image_size_y: u32,
    /// `[P, 3 (+ 1)]`
    pub is_colors_rgb_3d_clamped: B::FloatTensorPrimitive<2>,
    /// `[P, 1]`
    pub opacities_3d: B::FloatTensorPrimitive<2>,
    /// `P`
    pub point_count: u32,
    /// `[T]`
    pub point_indexes: B::IntTensorPrimitive<1>,
    /// `[I_Y, I_X]`
    pub point_rendered_counts: B::IntTensorPrimitive<2>,
    /// `[P, 2]`
    pub positions_2d: B::FloatTensorPrimitive<2>,
    /// `[P, 3]`
    pub positions_3d: B::FloatTensorPrimitive<2>,
    /// `[P]`
    pub radii: B::IntTensorPrimitive<1>,
    /// `[P, 4]`
    pub rotations: B::FloatTensorPrimitive<2>,
    /// `[P, 3]`
    pub scalings: B::FloatTensorPrimitive<2>,
    /// `[(I_X / T_X) * (I_Y / T_Y), 2]`
    pub tile_point_ranges: B::IntTensorPrimitive<2>,
    /// `[I_Y, I_X]`
    pub transmittances: B::FloatTensorPrimitive<2>,
    pub view_bound_x: f32,
    pub view_bound_y: f32,
    /// `[P, 3]`
    pub view_directions: B::FloatTensorPrimitive<2>,
    /// `[P, 3]`
    pub view_offsets: B::FloatTensorPrimitive<2>,
    /// `[3 (+ 1), 4]`
    pub view_transform: B::FloatTensorPrimitive<2>,
}
