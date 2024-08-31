pub mod config;
pub mod property;
pub mod render;

pub use crate::preset::backend;
pub use burn::{
    module::{Module, Param},
    tensor::{backend::Backend, Tensor, TensorData},
};
pub use config::*;

use std::fmt;

#[derive(Module)]
pub struct Gaussian3dScene<B: Backend> {
    /// `[P, 16, 3]`
    pub colors_sh: Param<Tensor<B, 3>>,
    /// `[P, 1]`
    pub opacities: Param<Tensor<B, 2>>,
    /// `[P, 3]`
    pub positions: Param<Tensor<B, 2>>,
    /// `[P, 4]`
    pub rotations: Param<Tensor<B, 2>>,
    /// `[P, 3]`
    pub scalings: Param<Tensor<B, 2>>,
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

impl<B: Backend> Default for Gaussian3dScene<B> {
    fn default() -> Self {
        Gaussian3dSceneConfig::default().into()
    }
}
