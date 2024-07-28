use burn::{
    backend::wgpu::{AutoGraphicsApi, JitBackend, WgpuRuntime},
    tensor::{backend, BasicOps, Element, Tensor},
};

pub type Wgpu = JitBackend<WgpuRuntime<AutoGraphicsApi, f32, i32>>;

pub trait TensorBackendExtension<B: backend::Backend, const D: usize> {
    fn from_wgpu_tensor<K: BasicOps<Wgpu>>(
        source: Tensor<Wgpu, D, K>,
        device: &B::Device,
    ) -> Self
    where
        <K as BasicOps<Wgpu>>::Elem: Element;

    fn into_wgpu_tensor<K: BasicOps<Wgpu>>(self) -> Tensor<Wgpu, D, K>
    where
        <K as BasicOps<Wgpu>>::Elem: Element;
}

impl<B: backend::Backend, const D: usize, K: BasicOps<B>>
    TensorBackendExtension<B, D> for Tensor<B, D, K>
where
    <K as BasicOps<B>>::Elem: Element,
{
    fn from_wgpu_tensor<K2: BasicOps<Wgpu>>(
        source: Tensor<Wgpu, D, K2>,
        device: &B::Device,
    ) -> Self
    where
        <K2 as BasicOps<Wgpu>>::Elem: Element,
    {
        Tensor::from_data(source.into_data().convert(), device)
    }

    fn into_wgpu_tensor<K2: BasicOps<Wgpu>>(self) -> Tensor<Wgpu, D, K2>
    where
        <K2 as BasicOps<Wgpu>>::Elem: Element,
    {
        Tensor::from_data(self.into_data().convert(), &Default::default())
    }
}
