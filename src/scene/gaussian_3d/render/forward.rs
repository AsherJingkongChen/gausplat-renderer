pub use super::*;

#[derive(Clone, Debug)]
pub struct RenderInput<B: Backend> {
    pub device: B::Device,
    /// `P`
    pub point_count: u64,
    /// `[P, 16, 3]`
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

#[derive(Clone, Debug)]
pub struct RenderOutput<B: Backend> {
    /// `[I_y, I_x, 3]`
    pub colors_rgb_2d: B::FloatTensorPrimitive,
    pub state: backward::RenderInput<B>,
}
