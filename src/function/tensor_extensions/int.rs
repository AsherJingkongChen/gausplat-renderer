use burn::tensor::{backend, Int, Tensor};

pub trait TensorIntExtension {
    fn zeros_like(
        &self,
    ) -> Self;
}

impl<B: backend::Backend, const D: usize> TensorIntExtension
    for Tensor<B, D, Int>
{
    fn zeros_like(
        &self,
    ) -> Self {
        Tensor::zeros(self.shape(), &self.device())
    }
}
