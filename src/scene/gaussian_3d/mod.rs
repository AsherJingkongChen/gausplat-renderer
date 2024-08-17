pub mod property;
pub mod render;

pub use burn::{
    module::Module,
    tensor::{backend::Backend, Data, Tensor},
};

use std::fmt;

#[derive(Module)]
pub struct Gaussian3dScene<B: Backend> {
    /// `[P, 16, 3]`
    pub colors_sh: Tensor<B, 3>,
    /// `[P, 1]`
    pub opacities: Tensor<B, 2>,
    /// `[P, 3]`
    pub positions: Tensor<B, 2>,
    /// `[P, 4]`
    pub rotations: Tensor<B, 2>,
    /// `[P, 3]`
    pub scalings: Tensor<B, 2>,
}

impl<B: Backend> Default for Gaussian3dScene<B> {
    fn default() -> Self {
        Self {
            colors_sh: Tensor::empty([0, 0, 0], &Default::default()),
            opacities: Tensor::empty([0, 0], &Default::default()),
            positions: Tensor::empty([0, 0], &Default::default()),
            rotations: Tensor::empty([0, 0], &Default::default()),
            scalings: Tensor::empty([0, 0], &Default::default()),
        }
    }
}

impl<B: Backend> fmt::Debug for Gaussian3dScene<B> {
    fn fmt(
        &self,
        f: &mut fmt::Formatter,
    ) -> fmt::Result {
        f.debug_struct("Gaussian3dScene")
            .field("devices", &self.devices())
            .field("colors_sh.dims()", &self.colors_sh.dims())
            .field("opacities.dims()", &self.opacities.dims())
            .field("positions.dims()", &self.positions.dims())
            .field("rotations.dims()", &self.rotations.dims())
            .field("scalings.dims()", &self.scalings.dims())
            .finish()
    }
}
