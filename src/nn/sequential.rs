use crate::nn::Module;
use crate::Tensor;

pub struct Sequential {
    layers: Vec<Box<dyn Module>>,
}

impl Sequential {
    pub fn new() -> Self {
        Self { layers: Vec::new() }
    }

    pub fn add<M: Module + 'static>(&mut self, layer: M) {
        self.layers.push(Box::new(layer));
    }
}

impl Module for Sequential {
    fn forward(&self, input: &Tensor) -> Tensor {
        let mut out = input.clone();
        for layer in &self.layers {
            out = layer.forward(&out);
        }
        out
    }

    fn parameters(&self) -> Vec<Tensor> {
        let mut params = Vec::new();
        for layer in &self.layers {
            params.extend(layer.parameters());
        }
        params
    }
}
