use crate::nn::Module;
use crate::Tensor;

pub struct Linear {
    weight: Tensor,
    bias: Tensor,
}

impl Linear {
    pub fn new(in_features: usize, out_features: usize) -> Self {
        let mut weight = Vec::with_capacity(in_features * out_features);
        for i in 0..(in_features * out_features) {
            let v = ((i as f32 + 1.0) * 0.37).sin() * 0.5;
            weight.push(v);
        }
        let bias = vec![0.0f32; out_features];

        Self {
            weight: Tensor::new(weight, vec![in_features, out_features], true),
            bias: Tensor::new(bias, vec![1, out_features], true),
        }
    }
}

impl Module for Linear {
    fn forward(&self, input: &Tensor) -> Tensor {
        input.matmul(&self.weight).add(&self.bias)
    }

    fn parameters(&self) -> Vec<Tensor> {
        vec![self.weight.clone(), self.bias.clone()]
    }
}
