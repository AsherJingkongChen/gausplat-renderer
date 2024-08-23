pub mod backward;
pub mod forward;

pub use super::*;
pub use burn::{backend::autodiff::Autodiff, config::Config};
pub use gausplat_importer::scene::sparse_view;

use burn::backend::wgpu;

pub type Wgpu =
    wgpu::JitBackend<wgpu::WgpuRuntime<wgpu::AutoGraphicsApi, f32, i32>>;

pub trait Gaussian3dRenderer<B: Backend> {
    fn backward(
        output: forward::RendererOutput<B>,
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
                let grad = grads.consume::<Wgpu, 3>(&ops.node);
                println!("grad: {:?}", grad);
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
        options: &RendererOptions,
    ) -> backward::RendererOutput<Wgpu> {
        backward::wgpu::render_gaussian_3d_scene(output, options)
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
        options: &RendererOptions,
    ) -> backward::RendererOutput<Wgpu> {
        backward::wgpu::render_gaussian_3d_scene(output, options)
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
