pub mod backward;
pub mod forward;

pub use super::*;
pub use burn::{
    backend::autodiff::{checkpoint::strategy::CheckpointStrategy, Autodiff},
    config::Config,
    tensor::ops::FloatTensor,
};
pub use gausplat_importer::scene::sparse_view;

use burn::{
    backend::{autodiff::checkpoint::strategy::NoCheckpointing, wgpu},
    tensor::Int,
};

pub type Wgpu =
    wgpu::JitBackend<wgpu::WgpuRuntime<wgpu::AutoGraphicsApi, f32, i32>>;

pub trait Gaussian3dRenderer<B: Backend> {
    fn backward(
        state: backward::RendererInput<B>,
        grad: FloatTensor<B, 3>,
    ) -> backward::RendererOutput<B>;

    fn forward(
        input: forward::RendererInput<B>,
        view: &sparse_view::View,
        options: RendererOptions,
    ) -> forward::RendererOutput<B>;
}

#[derive(Config, Copy, Debug)]
pub struct RendererOptions {
    /// It should be no more than `3`
    pub colors_sh_degree_max: u32,
}

#[derive(Clone, Debug)]
pub struct RenderOutput<B: Backend> {
    /// `[I_Y, I_X, 3]`
    pub colors_rgb_2d: Tensor<B, 3>,
}

#[derive(Clone, Debug)]
pub struct RenderOutputAutodiff<
    B: Backend,
    C: CheckpointStrategy = NoCheckpointing,
> {
    /// `[I_Y, I_X, 3]`
    pub colors_rgb_2d: Tensor<Autodiff<B, C>, 3>,
    // /// `[P]`
    // pub positions_2d_grad_norm: Tensor<B, 1>,
    /// `[P]`
    pub radii: Tensor<B, 1, Int>,
}

impl Gaussian3dScene<Wgpu> {
    pub fn render(
        &self,
        view: &sparse_view::View,
        options: RendererOptions,
    ) -> RenderOutput<Wgpu> {
        let input = forward::RendererInput {
            colors_sh: self.colors_sh().into_primitive(),
            opacities: self.opacities().into_primitive(),
            positions: self.positions().into_primitive(),
            rotations: self.rotations().into_primitive(),
            scalings: self.scalings().into_primitive(),
        };

        let output = Self::forward(input, view, options);

        let colors_rgb_2d = Tensor::new(output.colors_rgb_2d);

        RenderOutput { colors_rgb_2d }
    }
}

impl<C: CheckpointStrategy> Gaussian3dScene<Autodiff<Wgpu, C>> {
    pub fn render(
        &self,
        view: &sparse_view::View,
        options: RendererOptions,
    ) -> RenderOutputAutodiff<Wgpu, C> {
        use burn::backend::autodiff::{
            checkpoint::base::Checkpointer,
            grads::Gradients,
            ops::{Backward, Ops, OpsKind},
        };

        let colors_sh = self.colors_sh().into_primitive();
        let opacities = self.opacities().into_primitive();
        let positions = self.positions().into_primitive();
        let rotations = self.rotations().into_primitive();
        let scalings = self.scalings().into_primitive();

        let input = forward::RendererInput {
            colors_sh: colors_sh.primitive,
            opacities: opacities.primitive,
            positions: positions.primitive,
            rotations: rotations.primitive,
            scalings: scalings.primitive,
        };

        let output = Gaussian3dScene::forward(input, view, options);

        let nodes = [
            colors_sh.node,
            opacities.node,
            positions.node,
            rotations.node,
            scalings.node,
        ];

        let radii = Tensor::new(output.state.radii.to_owned());
        let colors_rgb_2d = Tensor::new(
            match BackwardOps.prepare::<C>(nodes).compute_bound().stateful() {
                OpsKind::Tracked(prep) => {
                    prep.finish(output.state, output.colors_rgb_2d)
                },
                OpsKind::UnTracked(prep) => prep.finish(output.colors_rgb_2d),
            },
        );

        #[derive(Debug)]
        struct BackwardOps;

        impl Backward<Wgpu, 3, 5> for BackwardOps {
            type State = backward::RendererInput<Wgpu>;

            fn backward(
                self,
                ops: Ops<Self::State, 5>,
                grads: &mut Gradients,
                _checkpointer: &mut Checkpointer,
            ) {
                let grad = grads.consume::<Wgpu, 3>(&ops.node);
                {
                    let grad = Tensor::<Wgpu, 3>::new(grad.to_owned());
                    println!("grad.dims: {:?}", grad.dims());
                    println!(
                        "grad.max: {:?}",
                        grad.to_owned().max().into_scalar()
                    );
                    println!(
                        "grad.mean: {:?}",
                        grad.to_owned().mean().into_scalar()
                    );
                    println!(
                        "grad.min: {:?}",
                        grad.to_owned().min().into_scalar()
                    );
                }

                let state = ops.state;
                let output = Gaussian3dScene::backward(state, grad);

                let colors_sh_grad =
                    Tensor::<Wgpu, 3>::new(output.colors_sh_grad.to_owned());
                let opacities_grad =
                    Tensor::<Wgpu, 2>::new(output.opacities_grad.to_owned());
                let positions_2d_grad_norm = Tensor::<Wgpu, 1>::new(
                    output.positions_2d_grad_norm.to_owned(),
                );
                let positions_grad =
                    Tensor::<Wgpu, 2>::new(output.positions_grad.to_owned());
                let rotations_grad =
                    Tensor::<Wgpu, 2>::new(output.rotations_grad.to_owned());
                let scalings_grad =
                    Tensor::<Wgpu, 2>::new(output.scalings_grad.to_owned());

                println!(
                    "colors_sh_grad: {:?} {:?}",
                    colors_sh_grad.to_owned().mean_dim(0).to_data().value,
                    colors_sh_grad.to_owned().var(0).to_data().value
                );
                println!(
                    "opacities_grad: {:?} {:?}",
                    opacities_grad.to_owned().mean_dim(0).to_data().value,
                    opacities_grad.to_owned().var(0).to_data().value
                );
                println!(
                    "positions_2d_grad_norm: {:?} {:?}",
                    positions_2d_grad_norm.to_owned().mean_dim(0).to_data().value,
                    positions_2d_grad_norm.to_owned().var(0).to_data().value
                );
                println!(
                    "positions_grad: {:?} {:?}",
                    positions_grad.to_owned().mean_dim(0).to_data().value,
                    positions_grad.to_owned().var(0).to_data().value
                );
                println!(
                    "rotations_grad: {:?} {:?}",
                    rotations_grad.to_owned().mean_dim(0).to_data().value,
                    rotations_grad.to_owned().var(0).to_data().value
                );
                println!(
                    "scalings_grad: {:?} {:?}",
                    scalings_grad.to_owned().mean_dim(0).to_data().value,
                    scalings_grad.to_owned().var(0).to_data().value
                );

                if let Some(node) = &ops.parents[0] {
                    grads.register::<Wgpu, 3>(node.id, output.colors_sh_grad);
                }
                if let Some(node) = &ops.parents[1] {
                    grads.register::<Wgpu, 2>(node.id, output.opacities_grad);
                }
                if let Some(node) = &ops.parents[2] {
                    grads.register::<Wgpu, 2>(node.id, output.positions_grad);
                }
                if let Some(node) = &ops.parents[3] {
                    grads.register::<Wgpu, 2>(node.id, output.rotations_grad);
                }
                if let Some(node) = &ops.parents[4] {
                    grads.register::<Wgpu, 2>(node.id, output.scalings_grad);
                }
            }
        }

        RenderOutputAutodiff {
            colors_rgb_2d,
            radii,
        }
    }
}

impl Gaussian3dRenderer<Wgpu> for Gaussian3dScene<Wgpu> {
    fn backward(
        state: backward::RendererInput<Wgpu>,
        grad: FloatTensor<Wgpu, 3>,
    ) -> backward::RendererOutput<Wgpu> {
        backward::wgpu::render_gaussian_3d_scene(state, grad)
    }

    fn forward(
        input: forward::RendererInput<Wgpu>,
        view: &sparse_view::View,
        options: RendererOptions,
    ) -> forward::RendererOutput<Wgpu> {
        forward::wgpu::render_gaussian_3d_scene(input, view, options)
    }
}

impl Default for RendererOptions {
    fn default() -> Self {
        Self {
            colors_sh_degree_max: 3,
        }
    }
}

// grad.max: 1.43064235e-5
// grad.mean: 4.1382722e-7
// grad.min: -1.5088152e-5
// Duration (loss): 1.219985167s
// score_mae: 0.053359266
// score_mssim: 0.8354029
// score_loss: 0.07560683

// {'grad_out_color': {'max': 1.5975578207871877e-05,
// 'mean': -1.9003134354989015e-07,
// 'min': -1.406932369718561e-05}} [24/08 16:00:10]
// {'grad_colors_precomp': tensor([-3.3955e-08, -2.9069e-08,  5.3411e-08], device='cuda:0'),
// 'grad_cov3Ds_precomp': tensor([-2.3147e-04, -2.1490e-04,  2.5882e-05, -1.0465e-03,  3.2294e-05,
// -1.0377e-03], device='cuda:0'),
// 'grad_means2D_norm': tensor([2.0924e-05], device='cuda:0'),
// 'grad_means3D': tensor([-2.8489e-07,  3.8036e-08,  6.0191e-08], device='cuda:0'),
// 'grad_opacities': tensor([-1.6490e-06], device='cuda:0'),
// 'grad_rotations': tensor([ 0.0000e+00, -3.1407e-17, -1.1906e-15,  1.9865e-16], device='cuda:0'),
// 'grad_scales': tensor([-1.3112e-06, -4.9944e-06, -4.8579e-06], device='cuda:0'),
// 'grad_sh': tensor([[-9.5786e-09, -8.2002e-09,  1.5067e-08],
// [ 0.0000e+00,  0.0000e+00,  0.0000e+00],
// [ 0.0000e+00,  0.0000e+00,  0.0000e+00],
// [ 0.0000e+00,  0.0000e+00,  0.0000e+00],
// [ 0.0000e+00,  0.0000e+00,  0.0000e+00],
// [ 0.0000e+00,  0.0000e+00,  0.0000e+00],
// [ 0.0000e+00,  0.0000e+00,  0.0000e+00],
// [ 0.0000e+00,  0.0000e+00,  0.0000e+00],
// [ 0.0000e+00,  0.0000e+00,  0.0000e+00],
// [ 0.0000e+00,  0.0000e+00,  0.0000e+00],
// [ 0.0000e+00,  0.0000e+00,  0.0000e+00],
// [ 0.0000e+00,  0.0000e+00,  0.0000e+00],
// [ 0.0000e+00,  0.0000e+00,  0.0000e+00],
// [ 0.0000e+00,  0.0000e+00,  0.0000e+00],
// [ 0.0000e+00,  0.0000e+00,  0.0000e+00],
// [ 0.0000e+00,  0.0000e+00,  0.0000e+00]], device='cuda:0')} [24/08 16:00:10]
