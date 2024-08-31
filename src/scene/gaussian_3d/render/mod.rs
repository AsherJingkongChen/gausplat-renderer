pub mod backward;
pub mod forward;

pub use super::*;
pub use burn::{config::Config, tensor::Int};
pub use gausplat_importer::scene::sparse_view;

use burn::{
    backend::autodiff::{
        checkpoint::{base::Checkpointer, strategy::NoCheckpointing},
        grads::Gradients,
        ops::{Backward, Ops, OpsKind},
    },
    tensor::TensorPrimitive,
};
use std::{fmt, marker};

pub trait Gaussian3dRenderer<B: Backend>:
    'static + Sized + fmt::Debug + marker::Send
{
    fn render_forward(
        input: forward::RenderInput<B>,
        view: &sparse_view::View,
        options: RenderOptions,
    ) -> forward::RenderOutput<B>;

    fn render_backward(
        state: backward::RenderInput<B>,
        colors_rgb_2d_grad: B::FloatTensorPrimitive<3>,
    ) -> backward::RenderOutput<B>;
}

#[derive(Config, Copy, Debug)]
pub struct RenderOptions {
    #[config(default = "3")]
    /// It should be no more than `3`
    pub colors_sh_degree_max: u32,
}

#[derive(Clone, Debug)]
pub struct RenderOutput<B: Backend> {
    /// `[I_Y, I_X, 3]`
    pub colors_rgb_2d: Tensor<B, 3>,
}

#[derive(Clone, Debug)]
pub struct RenderOutputAutodiff<B: Backend> {
    /// `[I_Y, I_X, 3]`
    pub colors_rgb_2d: Tensor<Autodiff<B>, 3>,
    // /// `[P]`
    // pub positions_2d_grad_norm: Tensor<B, 1>,
    /// `[P]`
    pub radii: Tensor<B, 1, Int>,
}

#[derive(Clone, Copy, Debug, Default)]
struct Gaussian3dRendererBackward<B: Backend, R: Gaussian3dRenderer<B>> {
    _b: marker::PhantomData<B>,
    _r: marker::PhantomData<R>,
}

impl<B: Backend> Gaussian3dScene<B>
where
    Self: Gaussian3dRenderer<B>,
{
    pub fn render(
        &self,
        view: &sparse_view::View,
        options: RenderOptions,
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
        view: &sparse_view::View,
        options: RenderOptions,
    ) -> RenderOutputAutodiff<B> {
        let colors_sh = self.colors_sh().into_primitive().tensor();
        let opacities = self.opacities().into_primitive().tensor();
        let positions = self.positions().into_primitive().tensor();
        let rotations = self.rotations().into_primitive().tensor();
        let scalings = self.scalings().into_primitive().tensor();

        let input = forward::RenderInput {
            colors_sh: colors_sh.primitive,
            opacities: opacities.primitive,
            positions: positions.primitive,
            rotations: rotations.primitive,
            scalings: scalings.primitive,
        };

        let output = Self::render_forward(input, view, options);

        let nodes = [
            colors_sh.node,
            opacities.node,
            positions.node,
            rotations.node,
            scalings.node,
        ];

        let radii = Tensor::new(output.state.radii.to_owned());
        let colors_rgb_2d = Tensor::new(TensorPrimitive::Float(
            match Gaussian3dRendererBackward::<B, Self>::default()
                .prepare::<NoCheckpointing>(nodes)
                .compute_bound()
                .stateful()
            {
                OpsKind::Tracked(prep) => {
                    prep.finish(output.state, output.colors_rgb_2d)
                },
                OpsKind::UnTracked(prep) => prep.finish(output.colors_rgb_2d),
            },
        ));

        RenderOutputAutodiff {
            colors_rgb_2d,
            radii,
        }
    }
}

impl<B: Backend, R: Gaussian3dRenderer<B>> Backward<B, 3, 5>
    for Gaussian3dRendererBackward<B, R>
{
    type State = backward::RenderInput<B>;

    fn backward(
        self,
        ops: Ops<Self::State, 5>,
        grads: &mut Gradients,
        _checkpointer: &mut Checkpointer,
    ) {
        let colors_rgb_2d_grad = grads.consume::<B, 3>(&ops.node);

        #[cfg(debug_assertions)]
        {
            let colors_rgb_2d_grad = Tensor::<B, 3>::new(
                TensorPrimitive::Float(colors_rgb_2d_grad.to_owned()),
            );
            println!(
                "colors_rgb_2d_grad.dims: {:?}",
                colors_rgb_2d_grad.dims()
            );
            println!(
                "colors_rgb_2d_grad.max: {:?}",
                colors_rgb_2d_grad.to_owned().max().into_scalar()
            );
            println!(
                "colors_rgb_2d_grad.mean: {:?}",
                colors_rgb_2d_grad.to_owned().mean().into_scalar()
            );
            println!(
                "colors_rgb_2d_grad.min: {:?}",
                colors_rgb_2d_grad.to_owned().min().into_scalar()
            );
        }

        let output = R::render_backward(ops.state, colors_rgb_2d_grad);

        #[cfg(debug_assertions)]
        {
            let colors_sh_grad = Tensor::<B, 3>::new(
                TensorPrimitive::Float(output.colors_sh_grad.to_owned()),
            );
            let opacities_grad = Tensor::<B, 2>::new(
                TensorPrimitive::Float(output.opacities_grad.to_owned()),
            );
            let positions_2d_grad_norm =
                Tensor::<B, 1>::new(TensorPrimitive::Float(
                    output.positions_2d_grad_norm.to_owned(),
                ));
            let positions_grad = Tensor::<B, 2>::new(
                TensorPrimitive::Float(output.positions_grad.to_owned()),
            );
            let rotations_grad = Tensor::<B, 2>::new(
                TensorPrimitive::Float(output.rotations_grad.to_owned()),
            );
            let scalings_grad = Tensor::<B, 2>::new(
                TensorPrimitive::Float(output.scalings_grad.to_owned()),
            );

            B::sync(
                &scalings_grad.device(),
                cubecl::client::SyncType::Wait,
            );

            println!(
                "colors_sh_grad: {} {}",
                colors_sh_grad.to_owned().mean_dim(0).to_data(),
                colors_sh_grad.to_owned().var(0).to_data()
            );
            println!(
                "opacities_grad: {} {}",
                opacities_grad.to_owned().mean_dim(0).to_data(),
                opacities_grad.to_owned().var(0).to_data()
            );
            println!(
                "positions_2d_grad_norm: {} {}",
                positions_2d_grad_norm.to_owned().mean_dim(0).to_data(),
                positions_2d_grad_norm.to_owned().var(0).to_data()
            );
            println!(
                "positions_grad: {} {}",
                positions_grad.to_owned().mean_dim(0).to_data(),
                positions_grad.to_owned().var(0).to_data()
            );
            println!(
                "rotations_grad: {} {}",
                rotations_grad.to_owned().mean_dim(0).to_data(),
                rotations_grad.to_owned().var(0).to_data()
            );
            println!(
                "scalings_grad: {} {}",
                scalings_grad.to_owned().mean_dim(0).to_data(),
                scalings_grad.to_owned().var(0).to_data()
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
    }
}

impl Default for RenderOptions {
    fn default() -> Self {
        Self::new()
    }
}
