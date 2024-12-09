//! 3DGS rendering context (forward).

pub use super::*;

/// Rendering inputs (forward).
#[derive(Clone, Debug)]
pub struct RenderInput<B: Backend> {
    /// Device for rendering.
    pub device: B::Device,
    /// Point count.
    ///
    /// It is `P`.
    pub point_count: u64,
    /// The shape is `[P, M * 3]`.
    pub colors_sh: B::FloatTensorPrimitive,
    /// The shape is `[P, 1]`.
    pub opacities: B::FloatTensorPrimitive,
    /// The shape is `[P, 3]`.
    pub positions: B::FloatTensorPrimitive,
    /// The shape is `[P, 4]`.
    pub rotations: B::FloatTensorPrimitive,
    /// The shape is `[P, 3]`.
    pub scalings: B::FloatTensorPrimitive,
}

/// Rendering outputs (forward).
#[derive(Clone, Debug)]
pub struct RenderOutput<B: Backend> {
    /// The shape is `[I_y, I_x, 3]`
    pub colors_rgb_2d: B::FloatTensorPrimitive,
    /// Rendering inputs (backward).
    pub state: backward::RenderInput<B>,
}
