//! Types for rendering (forward).

pub use super::*;

/// Inputs for rendering (forward).
#[derive(Clone, Debug)]
pub struct RenderInput<B: Backend> {
    /// Device for rendering.
    pub device: B::Device,
    /// `P`
    pub point_count: u64,
    /// `[P, 48]` <- `[P, 16, 3]`
    pub colors_sh: B::FloatTensorPrimitive,
    /// `[P]`
    pub opacities: B::FloatTensorPrimitive,
    /// `[P, 3]`
    pub positions: B::FloatTensorPrimitive,
    /// `[P, 4]`
    pub rotations: B::FloatTensorPrimitive,
    /// `[P, 3]`
    pub scalings: B::FloatTensorPrimitive,
}

/// Outputs for rendering (forward).
#[derive(Clone, Debug)]
pub struct RenderOutput<B: Backend> {
    /// `[I_y, I_x, 3]`
    pub colors_rgb_2d: B::FloatTensorPrimitive,
    /// Inputs for rendering (backward).
    pub state: backward::RenderInput<B>,
}
