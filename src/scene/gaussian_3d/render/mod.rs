pub mod backward;
pub mod forward;

pub use super::*;
pub use burn::{config::Config, tensor::Int};

use burn::{
    backend::autodiff::{
        checkpoint::{base::Checkpointer, strategy::NoCheckpointing},
        grads::Gradients,
        ops::{Backward, Ops, OpsKind},
        NodeID,
    },
    tensor::TensorPrimitive,
};
use std::{fmt, marker};

pub trait Gaussian3dRenderer<B: Backend>:
    'static + Send + Sized + fmt::Debug
{
    fn render_forward(
        input: forward::RenderInput<B>,
        view: &View,
        options: &Gaussian3dRendererOptions,
    ) -> forward::RenderOutput<B>;

    fn render_backward(
        state: backward::RenderInput<B>,
        colors_rgb_2d_grad: B::FloatTensorPrimitive<3>,
    ) -> backward::RenderOutput<B>;
}

#[derive(Config, Debug)]
pub struct Gaussian3dRendererOptions {
    #[config(default = "SH_DEGREE_MAX")]
    /// It should be no more than [`SH_DEGREE_MAX`].
    pub colors_sh_degree_max: u32,
}

#[derive(Clone)]
pub struct RenderOutput<B: Backend> {
    /// `[I_y, I_x, 3]`
    pub colors_rgb_2d: Tensor<B, 3>,
}

#[derive(Clone)]
pub struct RenderOutputAutodiff<B: Backend> {
    /// `[I_y, I_x, 3]`
    pub colors_rgb_2d: Tensor<Autodiff<B>, 3>,

    /// The shape of gradient is `[P]`
    ///
    /// ## Usage
    ///
    /// ```ignore
    /// use burn::backend::autodiff::grads::Gradients;
    ///
    /// let mut grads: Gradients = todo!();
    ///
    /// let positions_2d_grad_norm =
    ///     positions_2d_grad_norm_ref.grad_remove(&mut grads);
    /// ```
    pub positions_2d_grad_norm_ref: Tensor<Autodiff<B>, 1>,

    /// `[P]`
    pub radii: Tensor<B, 1, Int>,
}

#[derive(Clone, Copy, Debug, Default)]
struct Gaussian3dRendererBackward<B: Backend, R: Gaussian3dRenderer<B>> {
    __: marker::PhantomData<(B, R)>,
}

#[derive(Clone, Debug)]
struct Gaussian3dRendererBackwardState<B: Backend> {
    pub inner: backward::RenderInput<B>,
    pub positions_2d_grad_norm_ref_id: NodeID,
}

impl<B: Backend> Gaussian3dScene<B>
where
    Self: Gaussian3dRenderer<B>,
{
    pub fn render(
        &self,
        view: &View,
        options: &Gaussian3dRendererOptions,
    ) -> RenderOutput<B> {
        let input = forward::RenderInput {
            colors_sh: self.colors_sh().into_primitive().tensor(),
            opacities: self.opacities().into_primitive().tensor(),
            positions: self.positions().into_primitive().tensor(),
            rotations: self.rotations().into_primitive().tensor(),
            scalings: self.scalings().into_primitive().tensor(),
        };

        let output = Self::render_forward(input, view, options);

        let colors_rgb_2d =
            Tensor::new(TensorPrimitive::Float(output.colors_rgb_2d));

        RenderOutput { colors_rgb_2d }
    }
}

impl<B: Backend> Gaussian3dScene<Autodiff<B>>
where
    Self: Gaussian3dRenderer<B>,
{
    pub fn render(
        &self,
        view: &View,
        options: &Gaussian3dRendererOptions,
    ) -> RenderOutputAutodiff<B> {
        let (colors_sh, device) = {
            let values = self.colors_sh();
            let device = values.device();
            (values.into_primitive().tensor(), device)
        };
        let opacities = self.opacities().into_primitive().tensor();
        let positions = self.positions().into_primitive().tensor();
        let rotations = self.rotations().into_primitive().tensor();
        let scalings = self.scalings().into_primitive().tensor();

        let positions_2d_grad_norm_ref =
            Tensor::<Autodiff<B>, 1>::empty([1], &device);
        let positions_2d_grad_norm_ref_id = positions_2d_grad_norm_ref
            .to_owned()
            .into_primitive()
            .tensor()
            .node
            .id;

        let input = forward::RenderInput {
            colors_sh: colors_sh.primitive,
            opacities: opacities.primitive,
            positions: positions.primitive,
            rotations: rotations.primitive,
            scalings: scalings.primitive,
        };

        let output = Self::render_forward(input, view, options);

        let radii = Tensor::new(output.state.radii.to_owned());
        let colors_rgb_2d = Tensor::new(TensorPrimitive::Float(
            match Gaussian3dRendererBackward::<B, Self>::default()
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
                OpsKind::Tracked(prep) => prep.finish(
                    Gaussian3dRendererBackwardState {
                        inner: output.state,
                        positions_2d_grad_norm_ref_id,
                    },
                    output.colors_rgb_2d,
                ),
                OpsKind::UnTracked(prep) => prep.finish(output.colors_rgb_2d),
            },
        ));

        RenderOutputAutodiff {
            colors_rgb_2d,
            positions_2d_grad_norm_ref,
            radii,
        }
    }
}

impl<B: Backend, R: Gaussian3dRenderer<B>> Backward<B, 3, 5>
    for Gaussian3dRendererBackward<B, R>
{
    type State = Gaussian3dRendererBackwardState<B>;

    fn backward(
        self,
        ops: Ops<Self::State, 5>,
        grads: &mut Gradients,
        _checkpointer: &mut Checkpointer,
    ) {
        // #[cfg(debug_assertions)]
        use std::collections::BTreeMap;

        #[cfg(debug_assertions)]
        {
            log::debug!(
                target: "gausplat_renderer::scene",
                "Gaussian3dRendererBackward::backward",
            );
        }

        let colors_rgb_2d_grad = grads.consume::<B, 3>(&ops.node);

        // #[cfg(debug_assertions)]
        {
            let colors_rgb_2d_grad = Tensor::<B, 3>::new(
                TensorPrimitive::Float(colors_rgb_2d_grad.to_owned()),
            );

            #[cfg(debug_assertions)]
            {
                let gradient_means = BTreeMap::from([(
                    "colors_rgb_2d_grad.mean",
                    colors_rgb_2d_grad
                        .to_owned()
                        .mean_dim(0)
                        .mean_dim(1)
                        .into_data()
                        .to_vec::<f32>()
                        .unwrap(),
                )]);
                log::debug!(
                    target: "gausplat_renderer::scene",
                    "Gaussian3dRendererBackward::backward > Gradient means {gradient_means:#?}",
                );
            }

            if colors_rgb_2d_grad.contains_nan().into_scalar() {
                log::warn!(
                    target: "gausplat_renderer::scene",
                    "Gaussian3dRendererBackward::backward > colors_rgb_2d_grad.contains_nan",
                );
            }
        }

        let output = R::render_backward(ops.state.inner, colors_rgb_2d_grad);

        #[cfg(debug_assertions)]
        {
            let colors_sh_grad = Tensor::<B, 3>::new(TensorPrimitive::Float(
                output.colors_sh_grad.to_owned(),
            ));
            let opacities_grad = Tensor::<B, 2>::new(TensorPrimitive::Float(
                output.opacities_grad.to_owned(),
            ));
            let positions_2d_grad_norm =
                Tensor::<B, 1>::new(TensorPrimitive::Float(
                    output.positions_2d_grad_norm.to_owned(),
                ));
            let positions_grad = Tensor::<B, 2>::new(TensorPrimitive::Float(
                output.positions_grad.to_owned(),
            ));
            let rotations_grad = Tensor::<B, 2>::new(TensorPrimitive::Float(
                output.rotations_grad.to_owned(),
            ));
            let scalings_grad = Tensor::<B, 2>::new(TensorPrimitive::Float(
                output.scalings_grad.to_owned(),
            ));

            let gradient_means = BTreeMap::from([
                (
                    "colors_sh_grad.mean",
                    colors_sh_grad
                        .mean_dim(0)
                        .mean_dim(1)
                        .into_data()
                        .to_vec::<f32>()
                        .unwrap(),
                ),
                (
                    "opacities_grad.mean",
                    opacities_grad
                        .mean_dim(0)
                        .into_data()
                        .to_vec::<f32>()
                        .unwrap(),
                ),
                (
                    "positions_2d_grad_norm.mean",
                    positions_2d_grad_norm
                        .mean_dim(0)
                        .into_data()
                        .to_vec::<f32>()
                        .unwrap(),
                ),
                (
                    "positions_grad.mean",
                    positions_grad
                        .mean_dim(0)
                        .into_data()
                        .to_vec::<f32>()
                        .unwrap(),
                ),
                (
                    "rotations_grad.mean",
                    rotations_grad
                        .mean_dim(0)
                        .into_data()
                        .to_vec::<f32>()
                        .unwrap(),
                ),
                (
                    "scalings_grad.mean",
                    scalings_grad
                        .mean_dim(0)
                        .into_data()
                        .to_vec::<f32>()
                        .unwrap(),
                ),
            ]);
            log::debug!(
                target: "gausplat_renderer::scene",
                "Gaussian3dRendererBackward::backward > Gradient means {gradient_means:#?}",
            );
        }

        if let Some(node) = &ops.parents[0] {
            grads.register::<B, 3>(node.id, output.colors_sh_grad);
        }
        if let Some(node) = &ops.parents[1] {
            grads.register::<B, 2>(node.id, output.opacities_grad);
        }
        if let Some(node) = &ops.parents[2] {
            grads.register::<B, 2>(node.id, output.positions_grad);
        }
        if let Some(node) = &ops.parents[3] {
            grads.register::<B, 2>(node.id, output.rotations_grad);
        }
        if let Some(node) = &ops.parents[4] {
            grads.register::<B, 2>(node.id, output.scalings_grad);
        }

        grads.register::<B, 1>(
            ops.state.positions_2d_grad_norm_ref_id,
            output.positions_2d_grad_norm,
        );
    }
}

impl Default for Gaussian3dRendererOptions {
    #[inline]
    fn default() -> Self {
        Self::new()
    }
}

impl<B: Backend> fmt::Debug for RenderOutput<B> {
    fn fmt(
        &self,
        f: &mut fmt::Formatter<'_>,
    ) -> fmt::Result {
        f.debug_struct("RenderOutput")
            .field("colors_rgb_2d.dims()", &self.colors_rgb_2d.dims())
            .finish()
    }
}

impl<B: Backend> fmt::Debug for RenderOutputAutodiff<B> {
    fn fmt(
        &self,
        f: &mut fmt::Formatter<'_>,
    ) -> fmt::Result {
        let radii_dims = self.radii.dims();
        let point_count = radii_dims[0];
        let positions_2d_grad_norm_dims = [point_count];

        f.debug_struct("RenderOutputAutodiff")
            .field("colors_rgb_2d.dims()", &self.colors_rgb_2d.dims())
            .field(
                "positions_2d_grad_norm.dims()",
                &positions_2d_grad_norm_dims,
            )
            .field("radii.dims()", &radii_dims)
            .finish()
    }
}
