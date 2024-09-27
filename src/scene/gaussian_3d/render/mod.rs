pub mod backward;
pub mod forward;

pub use super::*;
pub use burn::{config::Config, record::Record, tensor::Int};

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

#[derive(Config, Debug, Record)]
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
pub struct RenderOutputAutodiff<AB: AutodiffBackend> {
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
    #[must_use = "The gradients should be consumed"]
    pub fn render(
        &self,
        view: &View,
        options: &Gaussian3dRendererOptions,
    ) -> RenderOutputAutodiff<Autodiff<B>> {
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
            Tensor::<Autodiff<B>, 1>::empty([1], &device).require_grad();
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
        #[cfg(debug_assertions)]
        log::debug!(
            target: "gausplat_renderer::scene",
            "Gaussian3dRendererBackward::backward",
        );

        let colors_rgb_2d_grad = grads.consume::<B, 3>(&ops.node);

        if ops.parents.iter().all(Option::is_none) {
            return;
        }

        let output = R::render_backward(ops.state.inner, colors_rgb_2d_grad);

        if let Some(node) = &ops.parents[0] {
            grads.register::<B, 2>(node.id, output.colors_sh_grad);
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
        f.debug_struct(&format!("RenderOutput<{}>", B::name()))
            .field("colors_rgb_2d.dims()", &self.colors_rgb_2d.dims())
            .finish()
    }
}

impl<AB: AutodiffBackend> fmt::Debug for RenderOutputAutodiff<AB> {
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
    fn default_render_wgpu() {
        use super::*;

        Gaussian3dScene::<Wgpu>::default().render(&VIEW, &Default::default());
    }

    #[test]
    fn default_render_wgpu_autodiff() {
        use super::*;

        Gaussian3dScene::<Autodiff<Wgpu>>::default()
            .render(&VIEW, &Default::default())
            .colors_rgb_2d
            .backward();
    }
}
