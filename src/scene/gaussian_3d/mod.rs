pub mod property;

pub use super::point::*;
pub use crate::spherical_harmonics::SH_DEGREE_MAX;
pub use crate::{
    backend::{self, *},
    error::Error,
    render::gaussian_3d as render,
};
pub use burn::{
    module::{AutodiffModule, Module, Param},
    tensor::{Tensor, TensorData},
};
pub use gausplat_loader::source::polygon;
pub use render::{
    Gaussian3dRenderOptions, Gaussian3dRenderOutput, Gaussian3dRenderOutputAutodiff,
    Gaussian3dRenderer,
};

use crate::spherical_harmonics::{SH_COEF, SH_COUNT_MAX};
use autodiff::{
    checkpoint::{base::Checkpointer, strategy::NoCheckpointing},
    grads::Gradients,
    ops::{Backward, Ops, OpsKind},
    NodeID,
};
use burn::tensor::{DType, TensorPrimitive};
use gausplat_loader::function::{Decoder, DecoderWith};
use humansize::{format_size, BINARY};
use rand::{rngs::StdRng, Rng, SeedableRng};
use std::{
    fmt,
    io::{BufReader, Read},
    marker,
    mem::take,
    sync::LazyLock,
};

/// [`SH_COUNT_MAX * 3`](SH_COUNT_MAX)
pub const COLORS_SH_FEATURE_COUNT: usize = SH_COUNT_MAX * 3;

#[derive(Module)]
pub struct Gaussian3dScene<B: Backend> {
    /// `[P, 48]` <- `[P, 16, 3]`
    pub colors_sh: Param<Tensor<B, 2>>,
    /// `[P, 1]`
    pub opacities: Param<Tensor<B, 2>>,
    /// `[P, 3]`
    pub positions: Param<Tensor<B, 2>>,
    /// `[P, 4]`. **(in scalar-last order, i.e., `[x, y, z, w]`)**
    pub rotations: Param<Tensor<B, 2>>,
    /// `[P, 3]`
    pub scalings: Param<Tensor<B, 2>>,
}

#[derive(Clone, Copy, Debug, Default)]
struct Gaussian3dRenderBackward<B: Backend, R: Gaussian3dRenderer<B>> {
    __: marker::PhantomData<(B, R)>,
}

#[derive(Clone, Debug)]
struct Gaussian3dRenderBackwardState<B: Backend> {
    pub inner: render::backward::RenderInput<B>,
    pub positions_2d_grad_norm_ref_id: NodeID,
}

pub const SEED_INIT: u64 = 0x3D65;

/// Scene initializers
impl<B: Backend> Gaussian3dScene<B> {
    pub fn init_from_points(
        points: Points,
        device: &B::Device,
    ) -> Self {
        // P
        let point_count = points.len();

        // ([P, 3], [P, 3])
        let (colors_rgb, positions) = points.iter().fold(
            (
                Vec::<f32>::with_capacity(point_count * 3),
                Vec::<f64>::with_capacity(point_count * 3),
            ),
            |(mut colors_rgb, mut positions), point| {
                colors_rgb.extend(point.color_rgb);
                positions.extend(point.position);
                (colors_rgb, positions)
            },
        );

        // [P, 48] <- [P, 16, 3]
        let colors_sh = Param::uninitialized(
            Default::default(),
            move |device, is_require_grad| {
                let mut colors_sh =
                    Tensor::zeros([point_count, COLORS_SH_FEATURE_COUNT], device);
                let colors_rgb = Tensor::from_data(
                    TensorData::new(colors_rgb.to_owned(), [point_count, 3]),
                    device,
                );

                colors_sh = colors_sh.slice_assign(
                    [0..point_count, 0..3],
                    (colors_rgb - 0.5) / SH_COEF.0[0],
                );

                colors_sh = Self::make_inner_colors_sh(colors_sh)
                    .set_require_grad(is_require_grad);

                #[cfg(all(debug_assertions, not(test)))]
                log::debug!(
                    target: "gausplat::renderer::gaussian_3d::scene",
                    "init_from_points > colors_sh",
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
                let opacities = Self::make_inner_opacities(Tensor::full(
                    [point_count, 1],
                    0.1,
                    device,
                ))
                .set_require_grad(is_require_grad);

                #[cfg(all(debug_assertions, not(test)))]
                log::debug!(
                    target: "gausplat::renderer::gaussian_3d::scene",
                    "init_from_points > opacities",
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
                let positions = Self::make_inner_positions(Tensor::from_data(
                    TensorData::new(positions.to_owned(), [point_count, 3]),
                    device,
                ))
                .set_require_grad(is_require_grad);

                #[cfg(all(debug_assertions, not(test)))]
                log::debug!(
                    target: "gausplat::renderer::gaussian_3d::scene",
                    "init_from_points > positions",
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
                let rotations = Self::make_inner_rotations(Tensor::from_data(
                    TensorData::new(
                        [0.0, 0.0, 0.0, 1.0].repeat(point_count),
                        [point_count, 4],
                    ),
                    device,
                ))
                .set_require_grad(is_require_grad);

                #[cfg(all(debug_assertions, not(test)))]
                log::debug!(
                    target: "gausplat::renderer::gaussian_3d::scene",
                    "init_from_points > rotations",
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
                let samples = StdRng::seed_from_u64(SEED_INIT)
                    .sample_iter(
                        rand_distr::LogNormal::new(0.0, std::f32::consts::E).unwrap(),
                    )
                    .take(point_count)
                    .map(|mut sample| {
                        sample = sample.max(f32::EPSILON);
                        sample_max = sample_max.max(sample);
                        sample
                    })
                    .collect();

                let scalings = Self::make_inner_scalings(
                    Tensor::from_data(TensorData::new(samples, [point_count, 1]), device)
                        .div_scalar(sample_max)
                        .sqrt()
                        .clamp_min(f32::EPSILON)
                        .repeat_dim(1, 3),
                )
                .set_require_grad(is_require_grad);

                #[cfg(all(debug_assertions, not(test)))]
                log::debug!(
                    target: "gausplat::renderer::gaussian_3d::scene",
                    "init_from_points > scalings",
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

    pub fn init_from_polygon_3dgs(
        reader: &mut impl Read,
        device: &B::Device,
    ) -> Result<Self, Error> {
        let reader = &mut BufReader::new(reader);

        let header = polygon::Header::decode(reader)?;
        if !header.is_same_order(&POLYGON_HEADER_3DGS) {
            return Err(Error::MismatchedPolygonHeader3DGS(header.into()));
        }
        let payload = polygon::Payload::decode_with(reader, &header)?;
        let mut object = polygon::Object { header, payload };

        // NOTE: The header is validated previously.
        let point_count = object.elem("vertex").unwrap().meta.count;

        let mut take_tensor = |names: &[String], device: &B::Device| {
            let dtype = DType::F32;
            let channel_count = names.len();
            let bytes = names.iter().fold(
                Vec::<u8>::with_capacity(channel_count * point_count * dtype.size()),
                |mut bytes, name| {
                    let data = object.elem_prop_mut("vertex", name).unwrap().data;
                    bytes.extend(take(data));
                    bytes
                },
            );
            let data = TensorData {
                bytes,
                shape: [channel_count, point_count].into(),
                dtype,
            };
            Tensor::<B, 2>::from_data(data, device)
                .swap_dims(0, 1)
                .set_require_grad(true)
        };

        // [P, 48] <- [P, 1, 3] + [P, 3, 15]
        let colors_sh = take_tensor(
            &(0..COLORS_SH_FEATURE_COUNT)
                .map(|i| {
                    if i < 3 {
                        format!("f_dc_{i}")
                    } else {
                        let i = i / 3 + (i % 3) * (SH_COUNT_MAX - 1) - 1;
                        format!("f_rest_{i}")
                    }
                    .into()
                })
                .collect::<Vec<_>>(),
            device,
        );

        // [P, 1]
        let opacities = take_tensor(&["opacity"].map(Into::into), device);

        // [P, 3]
        let positions = take_tensor(&["x", "y", "z"].map(Into::into), device);

        // [P, 4] (x, y, z, w) <- (w, x, y, z)
        let rotations =
            take_tensor(&[1, 2, 3, 0].map(|i| format!("rot_{i}").into()), device);

        // [P, 3]
        let scalings = take_tensor(
            &(0..3)
                .map(|i| format!("scale_{i}").into())
                .collect::<Vec<_>>(),
            device,
        );

        let mut scene = Self::default();
        scene
            .set_inner_colors_sh(colors_sh)
            .set_inner_opacities(opacities)
            .set_inner_positions(positions)
            .set_inner_rotations(rotations)
            .set_inner_scalings(scalings);

        #[cfg(all(debug_assertions, not(test)))]
        log::debug!(
            target: "gausplat::renderer::gaussian_3d::scene",
            "init_from_polygon_3dgs",
        );

        Ok(scene)
    }
}

/// Scene exporters
impl<B: Backend> Gaussian3dScene<B> {
    // TODO: It needs a point cloud viewer to validate the function.
    pub fn to_points(&self) -> Points {
        let point_count = self.point_count();

        // NOTE: The data type is converted.
        let colors_rgb = self
            .get_colors_sh()
            .slice([0..point_count, 0..3])
            .mul_scalar(SH_COEF.0[0])
            .add_scalar(0.5)
            .into_data()
            .convert::<f32>()
            .into_vec()
            .unwrap();

        // NOTE: The data type is converted.
        let positions = self
            .get_positions()
            .into_data()
            .convert::<f64>()
            .into_vec()
            .unwrap();

        colors_rgb
            .chunks_exact(3)
            .zip(positions.chunks_exact(3))
            .map(|(color_rgb, position)| Point {
                // NOTE: The slice size is guaranteed to fit.
                color_rgb: color_rgb.try_into().unwrap(),
                position: position.try_into().unwrap(),
            })
            .collect()
    }

    pub fn to_polygon_3dgs(&self) -> polygon::Object {
        // let point_count = self.point_count();

        // TODO: Use tensor operation to combine the tensor before loading to the host.
        // NOTE: The data type is converted.
        // let [colors_sh, opacities, positions, rotations, scalings] = [
        //     self.get_colors_sh(),
        //     self.get_opacities(),
        //     self.get_positions(),
        //     self.get_rotations(),
        //     self.get_scalings(),
        // ]
        // .map(|tensor| tensor.into_data().bytes);

        // // NOTE: The header is determined.
        // let mut object = polygon::Object {
        //     header: POLYGON_HEADER_3DGS.to_owned(),
        //     ..Default::default()
        // };
        // object.elem_mut("vertex").unwrap().meta.count = point_count;

        // *object.elem_prop_mut("vertex", "opacity").unwrap().data = opacities;

        todo!()
    }
}

impl<R: jit::JitRuntime, F: jit::FloatElement, I: jit::IntElement>
    Gaussian3dRenderer<JitBackend<R, F, I>> for Gaussian3dScene<JitBackend<R, F, I>>
{
    fn render_forward(
        input: render::forward::RenderInput<JitBackend<R, F, I>>,
        view: &render::View,
        options: &render::Gaussian3dRenderOptions,
    ) -> Result<render::forward::RenderOutput<JitBackend<R, F, I>>, Error> {
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
        view: &render::View,
        options: &render::Gaussian3dRenderOptions,
    ) -> Result<render::forward::RenderOutput<JitBackend<R, F, I>>, Error> {
        render::jit::forward(input, view, options)
    }

    fn render_backward(
        state: render::backward::RenderInput<JitBackend<R, F, I>>,
        colors_rgb_2d_grad: <JitBackend<R, F, I> as Backend>::FloatTensorPrimitive,
    ) -> render::backward::RenderOutput<JitBackend<R, F, I>> {
        render::jit::backward(state, colors_rgb_2d_grad)
    }
}

impl<B: Backend> Gaussian3dScene<B>
where
    Self: Gaussian3dRenderer<B>,
{
    pub fn render(
        &self,
        view: &render::View,
        options: &Gaussian3dRenderOptions,
    ) -> Result<Gaussian3dRenderOutput<B>, Error> {
        #[cfg(all(debug_assertions, not(test)))]
        log::debug!(
            target: "gausplat::renderer::gaussian_3d::scene",
            "render > autodiff disabled",
        );

        let input = render::forward::RenderInput {
            device: self.device(),
            point_count: self.point_count() as u64,
            colors_sh: self.colors_sh.val().into_primitive().tensor(),
            opacities: self.opacities.val().into_primitive().tensor(),
            positions: self.positions.val().into_primitive().tensor(),
            rotations: self.rotations.val().into_primitive().tensor(),
            scalings: self.scalings.val().into_primitive().tensor(),
        };

        let output = Self::render_forward(input, view, options)?;

        let colors_rgb_2d = Tensor::new(TensorPrimitive::Float(output.colors_rgb_2d));

        Ok(Gaussian3dRenderOutput { colors_rgb_2d })
    }
}

impl<B: Backend> Gaussian3dScene<Autodiff<B>>
where
    Self: Gaussian3dRenderer<B>,
{
    #[must_use = "The gradients should be used"]
    pub fn render(
        &self,
        view: &render::View,
        options: &Gaussian3dRenderOptions,
    ) -> Result<Gaussian3dRenderOutputAutodiff<Autodiff<B>>, Error> {
        let device = self.device();
        let colors_sh = self.colors_sh.val().into_primitive().tensor();
        let opacities = self.opacities.val().into_primitive().tensor();
        let positions = self.positions.val().into_primitive().tensor();
        let rotations = self.rotations.val().into_primitive().tensor();
        let scalings = self.scalings.val().into_primitive().tensor();

        let positions_2d_grad_norm_ref =
            Tensor::<Autodiff<B>, 1>::empty([1], &device).set_require_grad(true);
        let positions_2d_grad_norm_ref_id = positions_2d_grad_norm_ref
            .to_owned()
            .into_primitive()
            .tensor()
            .node
            .id;

        let input = render::forward::RenderInput {
            device,
            point_count: self.point_count() as u64,
            colors_sh: colors_sh.primitive,
            opacities: opacities.primitive,
            positions: positions.primitive,
            rotations: rotations.primitive,
            scalings: scalings.primitive,
        };

        let output = Self::render_forward(input, view, options)?;

        let radii = Tensor::new(output.state.radii.to_owned());
        let colors_rgb_2d = Tensor::new(TensorPrimitive::Float(
            match Gaussian3dRenderBackward::<B, Self>::default()
                .prepare::<NoCheckpointing>([
                    colors_sh.node,
                    opacities.node,
                    positions.node,
                    rotations.node,
                    scalings.node,
                ])
                .compute_bound()
                .stateful()
            {
                OpsKind::Tracked(prep) => {
                    #[cfg(all(debug_assertions, not(test)))]
                    log::debug!(
                        target: "gausplat::renderer::gaussian_3d::scene",
                        "render > autodiff tracked",
                    );

                    prep.finish(
                        Gaussian3dRenderBackwardState {
                            inner: output.state,
                            positions_2d_grad_norm_ref_id,
                        },
                        output.colors_rgb_2d,
                    )
                },
                OpsKind::UnTracked(prep) => {
                    #[cfg(all(debug_assertions, not(test)))]
                    log::debug!(
                        target: "gausplat::renderer::gaussian_3d::scene",
                        "render > autodiff untracked",
                    );

                    prep.finish(output.colors_rgb_2d)
                },
            },
        ));

        Ok(Gaussian3dRenderOutputAutodiff {
            colors_rgb_2d,
            positions_2d_grad_norm_ref,
            radii,
        })
    }
}

impl<B: Backend, R: Gaussian3dRenderer<B>> Backward<B, 5>
    for Gaussian3dRenderBackward<B, R>
{
    type State = Gaussian3dRenderBackwardState<B>;

    fn backward(
        self,
        ops: Ops<Self::State, 5>,
        grads: &mut Gradients,
        _checkpointer: &mut Checkpointer,
    ) {
        #[cfg(all(debug_assertions, not(test)))]
        log::debug!(
            target: "gausplat::renderer::gaussian_3d::scene",
            "render > backward",
        );

        let colors_rgb_2d_grad = grads.consume::<B>(&ops.node);

        if ops.parents.iter().all(Option::is_none) {
            return;
        }

        let output = R::render_backward(ops.state.inner, colors_rgb_2d_grad);

        if let Some(node) = &ops.parents[0] {
            grads.register::<B>(node.id, output.colors_sh_grad);
        }
        if let Some(node) = &ops.parents[1] {
            grads.register::<B>(node.id, output.opacities_grad);
        }
        if let Some(node) = &ops.parents[2] {
            grads.register::<B>(node.id, output.positions_grad);
        }
        if let Some(node) = &ops.parents[3] {
            grads.register::<B>(node.id, output.rotations_grad);
        }
        if let Some(node) = &ops.parents[4] {
            grads.register::<B>(node.id, output.scalings_grad);
        }

        grads.register::<B>(
            ops.state.positions_2d_grad_norm_ref_id,
            output.positions_2d_grad_norm,
        );
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
        Self::init_from_points(vec![Default::default(); 16], &Default::default())
    }
}

/// A polygon file header for 3D Gaussian splats.
///
/// <details>
/// <summary>
///     <strong>Click to expand</strong>
/// </summary>
/// <pre class=language-plaintext>
#[doc = include_str!("header.3dgs.ply")]
/// </pre>
/// </details>
pub static POLYGON_HEADER_3DGS: LazyLock<polygon::Header> = LazyLock::new(|| {
    include_str!("header.3dgs.ply")
        .parse::<polygon::Header>()
        .unwrap()
});

#[cfg(test)]
mod tests {
    use super::*;

    const VIEW: render::View = render::View {
        field_of_view_x: 1.39,
        field_of_view_y: 0.88,
        image_height: 600,
        image_width: 900,
        view_id: 0,
        view_position: [1.86, 0.45, 2.92],
        view_transform: [
            [-0.99, 0.08, -0.10, 0.0],
            [0.06, 0.99, 0.05, 0.000],
            [0.10, 0.05, -0.99, 0.00],
            [1.47, -0.69, 3.08, 1.00],
        ],
    };

    #[test]
    fn init_from_points() {
        use burn::backend::NdArray;

        let device = Default::default();
        let points = vec![
            Point {
                color_rgb: [1.0, 0.5, 0.0],
                position: [0.0, -0.5, 0.2],
            },
            Point {
                color_rgb: [0.5, 1.0, 0.2],
                position: [1.0, 0.0, -0.3],
            },
        ];

        let scene = Gaussian3dScene::<NdArray<f32>>::init_from_points(points, &device);

        let colors_sh = scene.get_colors_sh();
        assert_eq!(colors_sh.dims(), [2, 48]);

        let opacities = scene.get_opacities();
        assert_eq!(opacities.dims(), [2, 1]);

        let positions = scene.get_positions();
        assert_eq!(positions.dims(), [2, 3]);

        let rotations = scene.get_rotations();
        assert_eq!(rotations.dims(), [2, 4]);

        let scalings = scene.get_scalings();
        assert_eq!(scalings.dims(), [2, 3]);

        assert_eq!(scene.point_count(), 2);
        assert_eq!(scene.size(), (2 * 48 + 2 + 2 * 3 + 2 * 4 + 2 * 3) * 4);
    }

    #[test]
    fn default_render_wgpu() {
        Gaussian3dScene::<Wgpu>::default()
            .render(&VIEW, &Default::default())
            .unwrap();
    }

    #[test]
    fn default_render_wgpu_autodiff() {
        Gaussian3dScene::<Autodiff<Wgpu>>::default()
            .render(&VIEW, &Default::default())
            .unwrap()
            .colors_rgb_2d
            .backward();
    }
}
