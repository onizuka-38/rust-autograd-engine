use crate::Tensor;

pub struct MomentumSgd {
    params: Vec<Tensor>,
    velocity: Vec<Vec<f32>>,
    lr: f32,
    momentum: f32,
}

impl MomentumSgd {
    pub fn new(params: Vec<Tensor>, lr: f32, momentum: f32) -> Self {
        let velocity = params
            .iter()
            .map(|p| vec![0.0; p.data().len()])
            .collect::<Vec<_>>();

        Self {
            params,
            velocity,
            lr,
            momentum,
        }
    }

    pub fn step(&mut self) {
        for (idx, p) in self.params.iter().enumerate() {
            let grad = p.grad();
            for (j, g) in grad.iter().enumerate() {
                self.velocity[idx][j] = self.momentum * self.velocity[idx][j] + *g;
            }

            let mut n = p.inner.borrow_mut();
            if !n.requires_grad {
                continue;
            }
            for j in 0..n.data.len() {
                n.data[j] -= self.lr * self.velocity[idx][j];
            }
        }
    }

    pub fn zero_grad(&self) {
        for p in &self.params {
            p.zero_grad();
        }
    }
}
