pub mod backward;
pub mod forward;
pub mod jit;

pub use crate::{
    preset::backend::{self, Autodiff},
    scene::gaussian_3d::{Gaussian3dScene, View},
};
pub use burn::{
    config::Config,
    record::Record,
    tensor::{
        backend::{AutodiffBackend, Backend},
        Int, Tensor,
    },
};

use crate::preset::spherical_harmonics::SH_DEGREE_MAX;
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
        colors_rgb_2d_grad: B::FloatTensorPrimitive,
    ) -> backward::RenderOutput<B>;
}

#[derive(Config, Debug, Record)]
pub struct Gaussian3dRendererOptions {
    #[config(default = "SH_DEGREE_MAX")]
    /// It should be no more than [`SH_DEGREE_MAX`].
    pub colors_sh_degree_max: u32,
}

#[derive(Clone)]
pub struct Gaussian3dRenderOutput<B: Backend> {
    /// `[I_y, I_x, 3]`
    pub colors_rgb_2d: Tensor<B, 3>,
}

#[derive(Clone)]
pub struct Gaussian3dRenderOutputAutodiff<AB: AutodiffBackend> {
    /// `[I_y, I_x, 3]`
    pub colors_rgb_2d: Tensor<AB, 3>,

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
    pub positions_2d_grad_norm_ref: Tensor<AB, 1>,

    /// `[P]`
    pub radii: Tensor<AB::InnerBackend, 1, Int>,
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
    ) -> Gaussian3dRenderOutput<B> {
        let input = forward::RenderInput {
            device: self.device(),
            point_count: self.point_count() as u64,
            colors_sh: self.colors_sh().into_primitive().tensor(),
            opacities: self.opacities().into_primitive().tensor(),
            positions: self.positions().into_primitive().tensor(),
            rotations: self.rotations().into_primitive().tensor(),
            scalings: self.scalings().into_primitive().tensor(),
        };

        let output = Self::render_forward(input, view, options);

        let colors_rgb_2d =
            Tensor::new(TensorPrimitive::Float(output.colors_rgb_2d));

        Gaussian3dRenderOutput { colors_rgb_2d }
    }
}

impl<B: Backend> Gaussian3dScene<Autodiff<B>>
where
    Self: Gaussian3dRenderer<B>,
{
    #[must_use = "The gradients should be used"]
    pub fn render(
        &self,
        view: &View,
        options: &Gaussian3dRendererOptions,
    ) -> Gaussian3dRenderOutputAutodiff<Autodiff<B>> {
        let device = self.device();
        let colors_sh = self.colors_sh().into_primitive().tensor();
        let opacities = self.opacities().into_primitive().tensor();
        let positions = self.positions().into_primitive().tensor();
        let rotations = self.rotations().into_primitive().tensor();
        let scalings = self.scalings().into_primitive().tensor();

        let positions_2d_grad_norm_ref =
            Tensor::<Autodiff<B>, 1>::empty([1], &device).require_grad();
        let positions_2d_grad_norm_ref_id = positions_2d_grad_norm_ref
            .to_owned()
            .into_primitive()
            .tensor()
            .node
            .id;

        let input = forward::RenderInput {
            device,
            point_count: self.point_count() as u64,
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

        Gaussian3dRenderOutputAutodiff {
            colors_rgb_2d,
            positions_2d_grad_norm_ref,
            radii,
        }
    }
}

impl<B: Backend, R: Gaussian3dRenderer<B>> Backward<B, 5>
    for Gaussian3dRendererBackward<B, R>
{
    type State = Gaussian3dRendererBackwardState<B>;

    fn backward(
        self,
        ops: Ops<Self::State, 5>,
        grads: &mut Gradients,
        _checkpointer: &mut Checkpointer,
    ) {
        #[cfg(debug_assertions)]
        log::debug!(
            target: "gausplat_renderer::scene",
            "Gaussian3dRendererBackward::backward",
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

impl Default for Gaussian3dRendererOptions {
    #[inline]
    fn default() -> Self {
        Self::new()
    }
}

impl<B: Backend> fmt::Debug for Gaussian3dRenderOutput<B> {
    fn fmt(
        &self,
        f: &mut fmt::Formatter<'_>,
    ) -> fmt::Result {
        f.debug_struct(&format!("RenderOutput<{}>", B::name()))
            .field("colors_rgb_2d.dims()", &self.colors_rgb_2d.dims())
            .finish()
    }
}

impl<AB: AutodiffBackend> fmt::Debug for Gaussian3dRenderOutputAutodiff<AB> {
    fn fmt(
        &self,
        f: &mut fmt::Formatter<'_>,
    ) -> fmt::Result {
        let radii_dims = self.radii.dims();
        let positions_2d_grad_norm_dims = &radii_dims;

        f.debug_struct(&format!("RenderOutputAutodiff<{}>", AB::name()))
            .field("colors_rgb_2d.dims()", &self.colors_rgb_2d.dims())
            .field(
                "positions_2d_grad_norm.dims()",
                &positions_2d_grad_norm_dims,
            )
            .field("radii.dims()", &radii_dims)
            .finish()
    }
}
