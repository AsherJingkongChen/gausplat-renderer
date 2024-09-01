pub mod property;
pub mod render;

pub use crate::preset::backend;
pub use burn::{
    module::{Module, Param},
    tensor::{backend::Backend, Tensor, TensorData},
};
pub use gausplat_importer::scene::sparse_view;
pub use render::Gaussian3dRenderer;

use crate::preset::spherical_harmonics::*;
use backend::*;
use rand::{rngs::StdRng, SeedableRng};
use rand_distr::Distribution;
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

impl<B: Backend> Gaussian3dScene<B> {
    pub fn init(
        device: &B::Device,
        priors: sparse_view::Points,
    ) -> Self {
        // P
        let point_count = priors.len();

        // ([P, 3], [P, 3])
        let (colors_rgb, positions) = priors.into_iter().fold(
            (
                Vec::with_capacity(point_count * 3),
                Vec::with_capacity(point_count * 3),
            ),
            |(mut colors_rgb, mut positions), point| {
                colors_rgb.extend(point.color_rgb.map(|c| c as f32));
                positions.extend(point.position.map(|c| c as f32));
                (colors_rgb, positions)
            },
        );

        // [P, 16, 3]
        let colors_sh = Param::uninitialized(
            Default::default(),
            move |device, is_require_grad| {
                let mut colors_sh = Tensor::zeros([point_count, 16, 3], device);
                let colors_rgb = Tensor::from_data(
                    TensorData::new(colors_rgb.to_owned(), [point_count, 1, 3]),
                    device,
                );

                colors_sh = colors_sh.slice_assign(
                    [0..point_count, 0..1, 0..3],
                    (colors_rgb - 0.5) / SH_C[0][0],
                );

                colors_sh = Self::make_colors_sh(colors_sh)
                    .set_require_grad(is_require_grad);

                #[cfg(debug_assertions)]
                log::debug!(
                    target: "gausplat_renderer::scene",
                    "Gaussian3dScene::from(Gaussian3dSceneConfig) > colors_sh",
                );

                colors_sh
            },
            device.to_owned(),
            true,
        );

        // [P, 1]
        let opacities = Param::uninitialized(
            Default::default(),
            move |device, is_require_grad| {
                let opacities = Self::make_opacities(Tensor::full(
                    [point_count, 1],
                    0.1,
                    device,
                ))
                .set_require_grad(is_require_grad);

                #[cfg(debug_assertions)]
                log::debug!(
                    target: "gausplat_renderer::scene",
                    "Gaussian3dScene::from(Gaussian3dSceneConfig) > opacities",
                );

                opacities
            },
            device.to_owned(),
            true,
        );

        // [P, 3]
        let positions = Param::uninitialized(
            Default::default(),
            move |device, is_require_grad| {
                let positions = Self::make_positions(Tensor::from_data(
                    TensorData::new(positions.to_owned(), [point_count, 3]),
                    device,
                ))
                .set_require_grad(is_require_grad);

                #[cfg(debug_assertions)]
                log::debug!(
                    target: "gausplat_renderer::scene",
                    "Gaussian3dScene::from(Gaussian3dSceneConfig) > positions",
                );

                positions
            },
            device.to_owned(),
            true,
        );

        // [P, 4] (x, y, z, w)
        let rotations = Param::uninitialized(
            Default::default(),
            move |device, is_require_grad| {
                let rotations = Self::make_rotations(Tensor::from_data(
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
                    "Gaussian3dScene::from(Gaussian3dSceneConfig) > rotations",
                );

                rotations
            },
            device.to_owned(),
            true,
        );

        // [P, 3]
        let scalings = Param::uninitialized(
            Default::default(),
            move |device, is_require_grad| {
                let mut sample_max = f32::EPSILON;
                let samples =
                    rand_distr::LogNormal::new(0.0, std::f32::consts::E)
                        .expect("unreachable")
                        .sample_iter(&mut StdRng::seed_from_u64(0x3D65))
                        .take(point_count)
                        .map(|mut sample| {
                            sample = sample.max(f32::EPSILON);
                            sample_max = sample_max.max(sample);
                            sample
                        })
                        .collect();

                let scalings = Self::make_scalings(
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
                    "Gaussian3dScene::from(Gaussian3dSceneConfig) > scalings",
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
        view: &sparse_view::View,
        options: render::RenderOptions,
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
        view: &sparse_view::View,
        options: render::RenderOptions,
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
        Self::init(&Default::default(), vec![Default::default()])
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn scene_from_config_shapes() {
        use super::*;

        let device = Default::default();
        let priors = vec![
            sparse_view::Point {
                color_rgb: [1.0, 0.5, 0.0],
                position: [0.0, -0.5, 0.2],
            },
            sparse_view::Point {
                color_rgb: [0.5, 1.0, 0.2],
                position: [1.0, 0.0, -0.3],
            },
        ];

        let scene =
            Gaussian3dScene::<burn::backend::NdArray>::init(&device, priors);

        let colors_sh = scene.colors_sh();
        assert_eq!(colors_sh.dims(), [2, 16, 3]);

        let opacities = scene.opacities();
        assert_eq!(opacities.dims(), [2, 1]);

        let positions = scene.positions();
        assert_eq!(positions.dims(), [2, 3]);

        let rotations = scene.rotations();
        assert_eq!(rotations.dims(), [2, 4]);

        let scalings = scene.scalings();
        assert_eq!(scalings.dims(), [2, 3]);
    }
}
