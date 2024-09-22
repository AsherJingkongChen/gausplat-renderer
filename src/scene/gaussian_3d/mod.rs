pub mod property;
pub mod render;

pub use crate::preset::backend::{self, *};
pub use burn::{
    module::{AutodiffModule, Module, Param},
    tensor::{backend::Backend, Tensor, TensorData},
};
pub use gausplat_importer::dataset::gaussian_3d::{Point, Points, View};
pub use render::{Gaussian3dRenderer, Gaussian3dRendererOptions};

use crate::preset::{gaussian_3d::*, spherical_harmonics::*};
use humansize::{format_size, BINARY};
use rand::{rngs::StdRng, Rng, SeedableRng};
use std::{fmt, mem::size_of};

#[derive(Module)]
pub struct Gaussian3dScene<B: Backend> {
    /// `[P, 16 * 3]`
    pub colors_sh: Param<Tensor<B, 2>>,
    /// `[P, 1]`
    pub opacities: Param<Tensor<B, 2>>,
    /// `[P, 3]`
    pub positions: Param<Tensor<B, 2>>,
    /// `[P, 4]`
    pub rotations: Param<Tensor<B, 2>>,
    /// `[P, 3]`
    pub scalings: Param<Tensor<B, 2>>,
}

impl<B: Backend> Gaussian3dScene<B> {
    pub fn init(
        device: &B::Device,
        priors: Points,
    ) -> Self {
        // P
        let point_count = priors.len();
        let priors = (priors.to_owned(), priors);
        // [P, 3]
        let colors_rgb = priors
            .0
            .into_iter()
            .flat_map(|point| point.color_rgb)
            .collect::<Vec<_>>();
        // [P, 3]
        let positions = priors
            .1
            .into_iter()
            .flat_map(|point| point.position)
            .collect::<Vec<_>>();

        // [P, 16 * 3]
        let colors_sh = Param::uninitialized(
            "Gaussian3dScene::colors_sh".into(),
            move |device, is_require_grad| {
                let mut colors_sh =
                    Tensor::zeros([point_count, 16 * 3], device);
                let colors_rgb = Tensor::from_data(
                    TensorData::new(colors_rgb.to_owned(), [point_count, 3]),
                    device,
                );

                colors_sh = colors_sh.slice_assign(
                    [0..point_count, 0..3],
                    (colors_rgb - 0.5) / SH_C.0[0],
                );

                colors_sh = Self::make_inner_colors_sh(colors_sh)
                    .set_require_grad(is_require_grad);

                #[cfg(debug_assertions)]
                log::debug!(
                    target: "gausplat_renderer::scene",
                    "Gaussian3dScene::init > colors_sh",
                );

                colors_sh
            },
            device.to_owned(),
            true,
        );

        // [P, 1]
        let opacities = Param::uninitialized(
            "Gaussian3dScene::opacities".into(),
            move |device, is_require_grad| {
                let opacities = Self::make_inner_opacities(Tensor::full(
                    [point_count, 1],
                    0.1,
                    device,
                ))
                .set_require_grad(is_require_grad);

                #[cfg(debug_assertions)]
                log::debug!(
                    target: "gausplat_renderer::scene",
                    "Gaussian3dScene::init > opacities",
                );

                opacities
            },
            device.to_owned(),
            true,
        );

        // [P, 3]
        let positions = Param::uninitialized(
            "Gaussian3dScene::positions".into(),
            move |device, is_require_grad| {
                let positions = Self::make_inner_positions(Tensor::from_data(
                    TensorData::new(positions.to_owned(), [point_count, 3]),
                    device,
                ))
                .set_require_grad(is_require_grad);

                #[cfg(debug_assertions)]
                log::debug!(
                    target: "gausplat_renderer::scene",
                    "Gaussian3dScene::init > positions",
                );

                positions
            },
            device.to_owned(),
            true,
        );

        // [P, 4] (x, y, z, w)
        let rotations = Param::uninitialized(
            "Gaussian3dScene::rotations".into(),
            move |device, is_require_grad| {
                let rotations = Self::make_inner_rotations(Tensor::from_data(
                    TensorData::new(
                        [0.0, 0.0, 0.0, 1.0].repeat(point_count),
                        [point_count, 4],
                    ),
                    device,
                ))
                .set_require_grad(is_require_grad);

                #[cfg(debug_assertions)]
                log::debug!(
                    target: "gausplat_renderer::scene",
                    "Gaussian3dScene::init > rotations",
                );

                rotations
            },
            device.to_owned(),
            true,
        );

        // [P, 3]
        let scalings = Param::uninitialized(
            "Gaussian3dScene::scalings".into(),
            move |device, is_require_grad| {
                let mut sample_max = f32::EPSILON;
                let samples = StdRng::seed_from_u64(SEED)
                    .sample_iter(
                        rand_distr::LogNormal::new(0.0, std::f32::consts::E)
                            .expect("Unreachable"),
                    )
                    .take(point_count)
                    .map(|mut sample| {
                        sample = sample.max(f32::EPSILON);
                        sample_max = sample_max.max(sample);
                        sample
                    })
                    .collect();

                let scalings = Self::make_inner_scalings(
                    Tensor::from_data(
                        TensorData::new(samples, [point_count, 1]),
                        device,
                    )
                    .div_scalar(sample_max)
                    .sqrt()
                    .clamp_min(f32::EPSILON)
                    .repeat_dim(1, 3),
                )
                .set_require_grad(is_require_grad);

                #[cfg(debug_assertions)]
                log::debug!(
                    target: "gausplat_renderer::scene",
                    "Gaussian3dScene::init > scalings",
                );

                scalings
            },
            device.to_owned(),
            true,
        );

        Self {
            colors_sh,
            opacities,
            positions,
            rotations,
            scalings,
        }
    }
}

impl Gaussian3dRenderer<Wgpu> for Gaussian3dScene<Wgpu> {
    fn render_forward(
        input: render::forward::RenderInput<Wgpu>,
        view: &View,
        options: &render::Gaussian3dRendererOptions,
    ) -> render::forward::RenderOutput<Wgpu> {
        render::forward::wgpu::render_gaussian_3d_scene(input, view, options)
    }

    fn render_backward(
        state: render::backward::RenderInput<Wgpu>,
        colors_rgb_2d_grad: <Wgpu as Backend>::FloatTensorPrimitive<3>,
    ) -> render::backward::RenderOutput<Wgpu> {
        render::backward::wgpu::render_gaussian_3d_scene(
            state,
            colors_rgb_2d_grad,
        )
    }
}

impl Gaussian3dRenderer<Wgpu> for Gaussian3dScene<Autodiff<Wgpu>> {
    fn render_forward(
        input: render::forward::RenderInput<Wgpu>,
        view: &View,
        options: &render::Gaussian3dRendererOptions,
    ) -> render::forward::RenderOutput<Wgpu> {
        render::forward::wgpu::render_gaussian_3d_scene(input, view, options)
    }

    fn render_backward(
        state: render::backward::RenderInput<Wgpu>,
        colors_rgb_2d_grad: <Wgpu as Backend>::FloatTensorPrimitive<3>,
    ) -> render::backward::RenderOutput<Wgpu> {
        render::backward::wgpu::render_gaussian_3d_scene(
            state,
            colors_rgb_2d_grad,
        )
    }
}

impl<B: Backend> fmt::Debug for Gaussian3dScene<B> {
    fn fmt(
        &self,
        f: &mut fmt::Formatter,
    ) -> fmt::Result {
        f.debug_struct(&format!("Gaussian3dScene<{}>", B::name()))
            .field("devices", &self.devices())
            .field("size", &format_size(self.size(), BINARY))
            .field("colors_sh.dims()", &self.colors_sh.dims())
            .field("opacities.dims()", &self.opacities.dims())
            .field("positions.dims()", &self.positions.dims())
            .field("rotations.dims()", &self.rotations.dims())
            .field("scalings.dims()", &self.scalings.dims())
            .finish()
    }
}

impl<B: Backend> Default for Gaussian3dScene<B> {
    #[inline]
    fn default() -> Self {
        Self::init(&Default::default(), vec![Default::default(); 16])
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn init() {
        use super::*;
        use burn::backend::NdArray;

        let device = Default::default();
        let priors = vec![
            Point {
                color_rgb: [1.0, 0.5, 0.0],
                position: [0.0, -0.5, 0.2],
            },
            Point {
                color_rgb: [0.5, 1.0, 0.2],
                position: [1.0, 0.0, -0.3],
            },
        ];

        let scene = Gaussian3dScene::<NdArray<f32>>::init(&device, priors);

        let colors_sh = scene.colors_sh();
        assert_eq!(colors_sh.dims(), [2, 16 * 3]);

        let opacities = scene.opacities();
        assert_eq!(opacities.dims(), [2, 1]);

        let positions = scene.positions();
        assert_eq!(positions.dims(), [2, 3]);

        let rotations = scene.rotations();
        assert_eq!(rotations.dims(), [2, 4]);

        let scalings = scene.scalings();
        assert_eq!(scalings.dims(), [2, 3]);

        let size = scene.size();
        assert_eq!(size, (2 * 16 * 3 + 2 + 2 * 3 + 2 * 4 + 2 * 3) * 4);
    }
}
