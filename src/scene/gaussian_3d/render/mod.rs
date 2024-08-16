pub mod backward;
pub mod forward;

pub use gausplat_importer::scene::sparse_view;

use crate::scene::gaussian_3d::{Gaussian3dScene, Tensor};
use crate::{backend::Wgpu, error::Error};

impl Gaussian3dScene<Wgpu> {
    pub fn render(
        &self,
        view: &sparse_view::View,
        colors_sh_degree_max: u32,
    ) -> Result<Tensor<Wgpu, 3>, Error> {
        Ok(Tensor::new(
            self.render_forward(view, colors_sh_degree_max)?
                .colors_rgb_2d,
        ))
    }
}
