pub(super) mod wgpu;

pub use super::*;

#[derive(Clone, Debug)]
pub struct RendererOutput<B: Backend> {
    /// `[P, 16, 3]`
    pub colors_sh_grad: B::FloatTensorPrimitive<3>,
    /// `[P, 1]`
    pub opacities_3d_grad: B::FloatTensorPrimitive<2>,
    /// `[P]`
    pub positions_2d_grad_norm: B::FloatTensorPrimitive<1>,
    /// `[P, 3]`
    pub positions_3d_grad: B::FloatTensorPrimitive<2>,
    /// `[P, 4]`
    pub rotations_grad: B::FloatTensorPrimitive<2>,
    /// `[P, 3]`
    pub scalings_grad: B::FloatTensorPrimitive<2>,
}
