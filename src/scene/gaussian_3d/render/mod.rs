pub mod backward;
pub mod forward;

pub use crate::scene::gaussian_3d::*;
use crate::{backend::Wgpu, error::Error};
pub use gausplat_importer::scene::sparse_view;

impl Gaussian3dScene<Wgpu> {
    pub fn render(
        &self,
        view: &sparse_view::View,
        colors_sh_degree_max: u32,
    ) -> Result<Tensor<Wgpu, 3>, Error> {
        Ok(self.render_forward(view, colors_sh_degree_max)?.colors_rgb_2d)
    }
}
