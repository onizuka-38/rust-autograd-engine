use crate::nn::Module;
use crate::Tensor;

pub struct ReLU;
pub struct Sigmoid;

impl Module for ReLU {
    fn forward(&self, input: &Tensor) -> Tensor {
        input.relu()
    }

    fn parameters(&self) -> Vec<Tensor> {
        Vec::new()
    }
}

impl Module for Sigmoid {
    fn forward(&self, input: &Tensor) -> Tensor {
        input.sigmoid()
    }

    fn parameters(&self) -> Vec<Tensor> {
        Vec::new()
    }
}
