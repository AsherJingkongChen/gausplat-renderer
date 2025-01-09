//! 3DGS scene representation.

pub mod export;
pub mod import;
pub mod property;

pub use super::point::*;
pub use crate::spherical_harmonics::{SH_COUNT_MAX, SH_DEGREE_MAX};
pub use crate::{
    backend::{self, *},
    error::Error,
    render::gaussian_3d as render,
};
pub use burn::{
    module::{AutodiffModule, Module, Param},
    tensor::{Tensor, TensorData},
};
pub use render::{
    Gaussian3dRenderOptions, Gaussian3dRenderOutput, Gaussian3dRenderOutputAutodiff,
    Gaussian3dRenderer,
};

use crate::spherical_harmonics::SH_COEF;
use autodiff::{
    checkpoint::{base::Checkpointer, strategy::NoCheckpointing},
    grads::Gradients,
    ops::{Backward, Ops, OpsKind},
    NodeID,
};
use burn::tensor::{DType, TensorPrimitive};
use gausplat_loader::source::polygon;
use std::{fmt, marker, sync::LazyLock};

/// 3DGS default seed.
pub const SEED: u64 = 0x3D65;

/// A polygon file header for 3DGS.
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

/// 3DGS representation.
#[derive(Module)]
pub struct Gaussian3dScene<B: Backend> {
    /// Colors in SH space. (Inner value)
    ///
    /// The shape is `[P, M * 3]`, which derives from `[P, M, 3]`.
    /// - `P` is [`Self::point_count`].
    /// - `M` is [`SH_COUNT_MAX`].
    ///
    /// It is represented as orthonormalized spherical harmonic with RGB channels.
    pub colors_sh: Param<Tensor<B, 2>>,
    /// Opacities. (Inner value)
    ///
    /// The shape is `[P, 1]`.
    pub opacities: Param<Tensor<B, 2>>,
    /// 3D Positions. (Inner value)
    ///
    /// The shape is `[P, 3]`.
    pub positions: Param<Tensor<B, 2>>,
    /// Rotations. (Inner value)
    ///
    /// The shape is `[P, 4]`.
    ///
    /// They are represented as Hamilton quaternions in scalar-last order,
    /// i.e., `[x, y, z, w]`.
    pub rotations: Param<Tensor<B, 2>>,
    /// 3D scalings. (Inner value)
    ///
    /// The shape is `[P, 3]`.
    pub scalings: Param<Tensor<B, 2>>,
}

/// 3DGS render backward operator.
#[derive(Clone, Copy, Debug, Default)]
struct Gaussian3dRenderBackwardOp<B: Backend, R: Gaussian3dRenderer<B>> {
    __: marker::PhantomData<(B, R)>,
}

/// 3DGS render backward state.
#[derive(Clone, Debug)]
pub struct Gaussian3dRenderBackwardState<B: Backend> {
    /// Inner state.
    pub inner: render::backward::RenderInput<B>,
    /// The gradient norm of the 2D positions.
    pub positions_2d_grad_norm_ref_id: NodeID,
}

impl<
        R: jit::JitRuntime,
        F: jit::FloatElement,
        I: jit::IntElement,
        B: jit::BoolElement,
    > Gaussian3dRenderer<JitBackend<R, F, I, B>>
    for Gaussian3dScene<JitBackend<R, F, I, B>>
{
    #[inline]
    fn render_forward(
        input: render::forward::RenderInput<JitBackend<R, F, I, B>>,
        view: &render::View,
        options: &render::Gaussian3dRenderOptions,
    ) -> Result<render::forward::RenderOutput<JitBackend<R, F, I, B>>, Error> {
        render::jit::forward(input, view, options)
    }

    #[inline]
    fn render_backward(
        state: render::backward::RenderInput<JitBackend<R, F, I, B>>,
        colors_rgb_2d_grad: <JitBackend<R, F, I, B> as Backend>::FloatTensorPrimitive,
    ) -> render::backward::RenderOutput<JitBackend<R, F, I, B>> {
        render::jit::backward(state, colors_rgb_2d_grad)
    }
}

// TODO: Move render implementation to here.
impl<
        R: jit::JitRuntime,
        F: jit::FloatElement,
        I: jit::IntElement,
        B: jit::BoolElement,
    > Gaussian3dRenderer<JitBackend<R, F, I, B>>
    for Gaussian3dScene<Autodiff<JitBackend<R, F, I, B>>>
{
    #[inline]
    fn render_forward(
        input: render::forward::RenderInput<JitBackend<R, F, I, B>>,
        view: &render::View,
        options: &render::Gaussian3dRenderOptions,
    ) -> Result<render::forward::RenderOutput<JitBackend<R, F, I, B>>, Error> {
        render::jit::forward(input, view, options)
    }

    #[inline]
    fn render_backward(
        state: render::backward::RenderInput<JitBackend<R, F, I, B>>,
        colors_rgb_2d_grad: <JitBackend<R, F, I, B> as Backend>::FloatTensorPrimitive,
    ) -> render::backward::RenderOutput<JitBackend<R, F, I, B>> {
        render::jit::backward(state, colors_rgb_2d_grad)
    }
}

impl<B: Backend> Gaussian3dScene<B>
where
    Self: Gaussian3dRenderer<B>,
{
    /// Render the 3DGS scene.
    ///
    /// It renders an image with the given [`view`](render::View).
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
    /// Render the 3DGS scene with autodiff enabled.
    ///
    /// It renders a learnable image with the given [`view`](render::View).
    #[must_use = "The gradients should be used"]
    pub fn render(
        &self,
        view: &render::View,
        options: &Gaussian3dRenderOptions,
    ) -> Result<Gaussian3dRenderOutputAutodiff<Autodiff<B>>, Error> {
        let device = &self.device();
        let colors_sh = self.colors_sh.val().into_primitive().tensor();
        let opacities = self.opacities.val().into_primitive().tensor();
        let positions = self.positions.val().into_primitive().tensor();
        let rotations = self.rotations.val().into_primitive().tensor();
        let scalings = self.scalings.val().into_primitive().tensor();

        let input = render::forward::RenderInput {
            device: device.to_owned(),
            point_count: self.point_count() as u64,
            colors_sh: colors_sh.primitive,
            opacities: opacities.primitive,
            positions: positions.primitive,
            rotations: rotations.primitive,
            scalings: scalings.primitive,
        };

        let output = Self::render_forward(input, view, options)?;

        // It refers to the gradient norm of the 2D positions.
        let positions_2d_grad_norm_ref =
            Tensor::<Autodiff<B>, 1>::empty([1], device).set_require_grad(true);
        let positions_2d_grad_norm_ref_id = positions_2d_grad_norm_ref
            .to_owned()
            .into_primitive()
            .tensor()
            .node
            .id;
        let radii = Tensor::new(output.state.radii.to_owned());
        let colors_rgb_2d = Tensor::new(TensorPrimitive::Float(
            match Gaussian3dRenderBackwardOp::<B, Self>::default()
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
    for Gaussian3dRenderBackwardOp<B, R>
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

        // The gradient norm of the 2D positions will be obtained later.
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
            .field("device", &self.device())
            .field("point_count", &self.point_count())
            .field("size", &self.size_readable())
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
        Self::from_points(vec![Default::default(); 16], &Default::default())
    }
}

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
