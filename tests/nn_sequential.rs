use coregrad::nn::activations::ReLU;
use coregrad::nn::linear::Linear;
use coregrad::nn::sequential::Sequential;
use coregrad::{Module, Tensor};

#[test]
fn sequential_forward_preserves_expected_shape() {
    let x = Tensor::new(vec![0.0, 1.0, 1.0, 0.0], vec![2, 2], false);

    let mut model = Sequential::new();
    model.add(Linear::new(2, 3));
    model.add(ReLU);
    model.add(Linear::new(3, 1));

    let out = model.forward(&x);
    assert_eq!(out.shape(), vec![2, 1]);
}

#[test]
fn sequential_collects_all_trainable_parameters() {
    let mut model = Sequential::new();
    model.add(Linear::new(2, 3));
    model.add(ReLU);
    model.add(Linear::new(3, 1));

    let params = model.parameters();
    assert_eq!(params.len(), 4);

    let shapes = params.into_iter().map(|p| p.shape()).collect::<Vec<_>>();
    assert_eq!(shapes, vec![vec![2, 3], vec![1, 3], vec![3, 1], vec![1, 1]]);
}
