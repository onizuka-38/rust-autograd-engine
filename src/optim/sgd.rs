use crate::Tensor;

pub struct Sgd {
    params: Vec<Tensor>,
    lr: f32,
}

impl Sgd {
    pub fn new(params: Vec<Tensor>, lr: f32) -> Self {
        Self { params, lr }
    }

    pub fn step(&self) {
        for p in &self.params {
            p.apply_grad(self.lr);
        }
    }

    pub fn zero_grad(&self) {
        for p in &self.params {
            p.zero_grad();
        }
    }
}
