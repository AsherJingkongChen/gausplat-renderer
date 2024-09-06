pub mod wgpu;

pub use super::*;

#[derive(Clone, Debug)]
pub struct RenderInput<B: Backend> {
    /// `[P, 16 * 3]`
    pub colors_sh: B::FloatTensorPrimitive<2>,
    /// `[P, 1]`
    pub opacities: B::FloatTensorPrimitive<2>,
    /// `[P, 3]`
    pub positions: B::FloatTensorPrimitive<2>,
    /// `[P, 4]`
    pub rotations: B::FloatTensorPrimitive<2>,
    /// `[P, 3]`
    pub scalings: B::FloatTensorPrimitive<2>,
}

#[derive(Clone, Debug)]
pub struct RenderOutput<B: Backend> {
    /// `[I_y, I_x, 3]`
    pub colors_rgb_2d: B::FloatTensorPrimitive<3>,
    pub state: backward::RenderInput<B>,
}
