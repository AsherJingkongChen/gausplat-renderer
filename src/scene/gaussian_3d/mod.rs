pub mod property;
pub mod spherical_harmonics;

pub use crate::{
    backend::{self, *},
    render::gaussian_3d as render,
};
pub use burn::{
    module::{AutodiffModule, Module, Param},
    tensor::{Tensor, TensorData},
};
pub use gausplat_importer::dataset::gaussian_3d::{Point, Points, View};
pub use render::{Gaussian3dRenderer, Gaussian3dRendererOptions};
pub use spherical_harmonics::SH_DEGREE_MAX;

use humansize::{format_size, BINARY};
use rand::{rngs::StdRng, Rng, SeedableRng};
use spherical_harmonics::{SH_C, SH_COUNT};
use std::{fmt, mem::size_of};

pub const COLORS_SH_FEATURE_COUNT: usize = SH_COUNT * 3;

#[derive(Module)]
pub struct Gaussian3dScene<B: Backend> {
    /// `[P, 48] <- [P, 16, 3]`
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

pub const SEED_INIT: u64 = 0x3D65;

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

        // [P, 48] <- [P, 16, 3]
        let colors_sh = Param::uninitialized(
            "Gaussian3dScene::colors_sh".into(),
            move |device, is_require_grad| {
                let mut colors_sh = Tensor::zeros(
                    [point_count, COLORS_SH_FEATURE_COUNT],
                    device,
                );
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
                let samples = StdRng::seed_from_u64(SEED_INIT)
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

impl<R: jit::JitRuntime, F: jit::FloatElement, I: jit::IntElement>
    Gaussian3dRenderer<JitBackend<R, F, I>>
    for Gaussian3dScene<JitBackend<R, F, I>>
{
    fn render_forward(
        input: render::forward::RenderInput<JitBackend<R, F, I>>,
        view: &View,
        options: &render::Gaussian3dRendererOptions,
    ) -> render::forward::RenderOutput<JitBackend<R, F, I>> {
        render::jit::forward(input, view, options)
    }

    fn render_backward(
        state: render::backward::RenderInput<JitBackend<R, F, I>>,
        colors_rgb_2d_grad: <JitBackend<R, F, I> as Backend>::FloatTensorPrimitive,
    ) -> render::backward::RenderOutput<JitBackend<R, F, I>> {
        render::jit::backward(state, colors_rgb_2d_grad)
    }
}

impl<R: jit::JitRuntime, F: jit::FloatElement, I: jit::IntElement>
    Gaussian3dRenderer<JitBackend<R, F, I>>
    for Gaussian3dScene<Autodiff<JitBackend<R, F, I>>>
{
    fn render_forward(
        input: render::forward::RenderInput<JitBackend<R, F, I>>,
        view: &View,
        options: &render::Gaussian3dRendererOptions,
    ) -> render::forward::RenderOutput<JitBackend<R, F, I>> {
        render::jit::forward(input, view, options)
    }

    fn render_backward(
        state: render::backward::RenderInput<JitBackend<R, F, I>>,
        colors_rgb_2d_grad: <JitBackend<R, F, I> as Backend>::FloatTensorPrimitive,
    ) -> render::backward::RenderOutput<JitBackend<R, F, I>> {
        render::jit::backward(state, colors_rgb_2d_grad)
    }
}

impl<B: Backend> fmt::Debug for Gaussian3dScene<B> {
    fn fmt(
        &self,
        f: &mut fmt::Formatter,
    ) -> fmt::Result {
        f.debug_struct(&format!("Gaussian3dScene<{}>", B::name()))
            .field("devices", &self.devices())
            .field("point_count", &self.point_count())
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
    use super::*;

    const VIEW: View = View {
        field_of_view_x: 1.39,
        field_of_view_y: 0.88,
        image_height: 600,
        image_width: 900,
        view_id: 0,
        view_position: [1.86, 0.45, 2.92],
        view_transform: [
            [-0.99, 0.08, -0.10, 0.00],
            [0.06, 0.99, 0.05, 0.00],
            [0.10, 0.05, -0.99, 0.00],
            [1.47, -0.69, 3.08, 1.00],
        ],
    };

    #[test]
    fn init() {
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
        assert_eq!(colors_sh.dims(), [2, 48]);

        let opacities = scene.opacities();
        assert_eq!(opacities.dims(), [2, 1]);

        let positions = scene.positions();
        assert_eq!(positions.dims(), [2, 3]);

        let rotations = scene.rotations();
        assert_eq!(rotations.dims(), [2, 4]);

        let scalings = scene.scalings();
        assert_eq!(scalings.dims(), [2, 3]);

        assert_eq!(scene.point_count(), 2);
        assert_eq!(scene.size(), (2 * 48 + 2 + 2 * 3 + 2 * 4 + 2 * 3) * 4);
    }

    #[test]
    fn default_render_wgpu() {
        Gaussian3dScene::<Wgpu>::default().render(&VIEW, &Default::default());
    }

    #[test]
    fn default_render_wgpu_autodiff() {
        Gaussian3dScene::<Autodiff<Wgpu>>::default()
            .render(&VIEW, &Default::default())
            .colors_rgb_2d
            .backward();
    }
}
