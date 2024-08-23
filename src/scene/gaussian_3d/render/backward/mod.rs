pub(super) mod wgpu;

pub use super::*;

#[derive(Clone, Debug)]
pub struct RendererOutput<B: Backend> {
    pub _b: std::marker::PhantomData<B>,
}
