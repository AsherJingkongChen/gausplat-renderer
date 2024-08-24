pub mod backward;
pub mod forward;

pub use super::*;
pub use burn::{backend::autodiff::Autodiff, config::Config};
pub use gausplat_importer::scene::sparse_view;

use burn::backend::wgpu;
use gausplat_importer::scene::sparse_view::view;

pub type Wgpu =
    wgpu::JitBackend<wgpu::WgpuRuntime<wgpu::AutoGraphicsApi, f32, i32>>;

pub trait Gaussian3dRenderer<B: Backend> {
    fn backward(
        output: forward::RendererOutput<B>,
        view: &sparse_view::View,
        options: &RendererOptions,
    ) -> backward::RendererOutput<B>;

    fn forward(
        scene: &Gaussian3dScene<B>,
        view: &sparse_view::View,
        options: &RendererOptions,
    ) -> forward::RendererOutput<B>;
}

#[derive(Config, Copy, Debug)]
pub struct RendererOptions {
    /// `<= 3`
    pub colors_sh_degree_max: u32,
}

#[derive(Clone, Debug)]
pub struct RenderOutput<B: Backend> {
    pub colors_rgb_2d: Tensor<B, 3>,
}

#[derive(Clone, Debug)]
pub struct RenderOutputAutodiff<AB: AutodiffBackend> {
    pub colors_rgb_2d: Tensor<AB, 3>,
    pub radii: Tensor<AB, 1>,
}

impl Gaussian3dScene<Wgpu> {
    pub fn render(
        &self,
        view: &sparse_view::View,
        options: &RendererOptions,
    ) -> RenderOutput<Wgpu> {
        let output = Self::forward(self, view, options);
        let colors_rgb_2d = Tensor::new(output.colors_rgb_2d);

        RenderOutput { colors_rgb_2d }
    }
}

impl Gaussian3dScene<Autodiff<Wgpu>> {
    pub fn render(
        &self,
        view: &sparse_view::View,
        options: &RendererOptions,
    ) -> Tensor<Autodiff<Wgpu>, 3> {
        use burn::backend::autodiff::{
            checkpoint::{base::Checkpointer, strategy::NoCheckpointing},
            grads::Gradients,
            ops::{Backward, Ops, OpsKind},
        };
        use burn::module::AutodiffModule;

        #[derive(Debug)]
        struct BackwardOps;

        // Implement the backward trait for the given backend B, the node gradient being of rank D
        // with three other gradients to calculate (lhs, rhs, and bias).
        impl Backward<Wgpu, 3, 1> for BackwardOps {
            // Our state that we must build during the forward pass to compute the backward pass.
            //
            // Note that we could improve the performance further by only keeping the state of
            // tensors that are tracked, improving memory management, but for simplicity, we avoid
            // that part.
            type State = ();

            fn backward(
                self,
                ops: Ops<Self::State, 1>,
                grads: &mut Gradients,
                checkpointer: &mut Checkpointer,
            ) {
                let grad = Tensor::<Wgpu, 3>::new(grads.consume::<Wgpu, 3>(&ops.node));
                println!("grad.max: {:?}", grad.to_owned().max().into_scalar());
                println!("grad.mean: {:?}", grad.to_owned().mean().into_scalar());
                println!("grad.min: {:?}", grad.to_owned().min().into_scalar());

                // grad.max: 1.1158236e-5
                // grad.mean: -2.8301318e-7
                // grad.min: -1.5639705e-5
                // Duration (loss): 1.254431708s
                // score_mae: 0.2822224
                // score_mssim: 0.27705392
                // score_loss: 0.3703671

                // loss l1: 0.23170578479766846 [23/08 23:51:54]
                // loss lmssim: 0.27757564187049866 [23/08 23:51:54]
                // loss: 0.329849511384964 [23/08 23:51:54]
                // {'grad_out_color': {'max': 1.5975578207871877e-05,
                //                     'mean': -1.9003134354989015e-07,
                //                     'min': -1.406932369718561e-05}} [23/08 23:51:54]
                // {'grad_colors_precomp': tensor([-3.3955e-08, -2.9069e-08,  5.3412e-08], device='cuda:0'),
                //  'grad_cov3Ds_precomp': tensor([-2.3147e-04, -2.1490e-04,  2.5882e-05, -1.0465e-03,  3.2294e-05,
                //         -1.0377e-03], device='cuda:0'),
                //  'grad_means2D_norm': tensor([2.0924e-05], device='cuda:0'),
                //  'grad_means3D': tensor([-2.8489e-07,  3.8037e-08,  6.0191e-08], device='cuda:0'),
                //  'grad_opacities': tensor([-1.6490e-06], device='cuda:0'),
                //  'grad_rotations': tensor([0.0000e+00, 3.7310e-16, 7.7768e-16, 5.6407e-17], device='cuda:0'),
                //  'grad_scales': tensor([-1.3112e-06, -4.9944e-06, -4.8579e-06], device='cuda:0'),
                //  'grad_sh': tensor([[-9.5786e-09, -8.2002e-09,  1.5067e-08],
                //         [ 0.0000e+00,  0.0000e+00,  0.0000e+00],
                //         [ 0.0000e+00,  0.0000e+00,  0.0000e+00],
                //         [ 0.0000e+00,  0.0000e+00,  0.0000e+00],
                //         [ 0.0000e+00,  0.0000e+00,  0.0000e+00],
                //         [ 0.0000e+00,  0.0000e+00,  0.0000e+00],
                //         [ 0.0000e+00,  0.0000e+00,  0.0000e+00],
                //         [ 0.0000e+00,  0.0000e+00,  0.0000e+00],
                //         [ 0.0000e+00,  0.0000e+00,  0.0000e+00],
                //         [ 0.0000e+00,  0.0000e+00,  0.0000e+00],
                //         [ 0.0000e+00,  0.0000e+00,  0.0000e+00],
                //         [ 0.0000e+00,  0.0000e+00,  0.0000e+00],
                //         [ 0.0000e+00,  0.0000e+00,  0.0000e+00],
                //         [ 0.0000e+00,  0.0000e+00,  0.0000e+00],
                //         [ 0.0000e+00,  0.0000e+00,  0.0000e+00],
                //         [ 0.0000e+00,  0.0000e+00,  0.0000e+00]], device='cuda:0')} [23/08 23:51:54]
            }
        }

        // Each node can be fetched with `ops.parents` in the same order as defined here.
        let ad_tensor = match BackwardOps
            .prepare::<NoCheckpointing>([self
                .colors_sh()
                .into_primitive()
                .node])
            .compute_bound()
            .stateful()
        {
            OpsKind::Tracked(prep) => prep.finish(
                (),
                Self::forward(&self.valid(), view, options).colors_rgb_2d,
            ),
            OpsKind::UnTracked(_) => {
                unimplemented!("Untracked operations are not supported");
            },
        };

        Tensor::<Autodiff<Wgpu>, 3>::new(ad_tensor)
    }
}

impl Gaussian3dRenderer<Wgpu> for Gaussian3dScene<Wgpu> {
    fn backward(
        output: forward::RendererOutput<Wgpu>,
        view: &sparse_view::View,
        options: &RendererOptions,
    ) -> backward::RendererOutput<Wgpu> {
        backward::wgpu::render_gaussian_3d_scene(output, view, options)
    }

    fn forward(
        scene: &Gaussian3dScene<Wgpu>,
        view: &sparse_view::View,
        options: &RendererOptions,
    ) -> forward::RendererOutput<Wgpu> {
        forward::wgpu::render_gaussian_3d_scene(scene, view, options)
    }
}

impl Gaussian3dRenderer<Wgpu> for Gaussian3dScene<Autodiff<Wgpu>> {
    fn backward(
        output: forward::RendererOutput<Wgpu>,
        view: &sparse_view::View,
        options: &RendererOptions,
    ) -> backward::RendererOutput<Wgpu> {
        backward::wgpu::render_gaussian_3d_scene(output, view, options)
    }

    fn forward(
        scene: &Gaussian3dScene<Wgpu>,
        view: &sparse_view::View,
        options: &RendererOptions,
    ) -> forward::RendererOutput<Wgpu> {
        forward::wgpu::render_gaussian_3d_scene(scene, view, options)
    }
}

impl Default for RendererOptions {
    fn default() -> Self {
        Self {
            colors_sh_degree_max: 3,
        }
    }
}
