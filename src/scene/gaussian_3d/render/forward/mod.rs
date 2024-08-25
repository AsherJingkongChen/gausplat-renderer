pub(super) mod wgpu;

pub use super::*;

#[derive(Clone, Debug)]
pub struct RendererInput<B: Backend> {
    /// `[P, 16, 3]`
    pub colors_sh: B::FloatTensorPrimitive<3>,
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
pub struct RendererOutput<B: Backend> {
    /// `[I_Y, I_X, 3]`
    pub colors_rgb_2d: B::FloatTensorPrimitive<3>,
    pub state: backward::RendererInput<B>,
}
