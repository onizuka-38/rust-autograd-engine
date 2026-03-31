pub mod activations;
pub mod linear;
pub mod sequential;

use crate::Tensor;

pub trait Module {
    fn forward(&self, input: &Tensor) -> Tensor;
    fn parameters(&self) -> Vec<Tensor>;
}

pub fn mse_loss(pred: &Tensor, target: &Tensor) -> Tensor {
    let diff = pred.sub(target);
    let sq = diff.mul(&diff);
    sq.mean()
}
