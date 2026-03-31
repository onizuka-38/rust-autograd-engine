use coregrad::nn::activations::{ReLU, Sigmoid};
use coregrad::nn::linear::Linear;
use coregrad::{mse_loss, Module, Sgd, Tensor};

#[test]
fn xor_training_loss_decreases() {
    let x = Tensor::new(vec![0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0], vec![4, 2], false);
    let y = Tensor::new(vec![0.0, 1.0, 1.0, 0.0], vec![4, 1], false);

    let l1 = Linear::new(2, 4);
    let l2 = Linear::new(4, 1);
    let relu = ReLU;
    let sigmoid = Sigmoid;

    let mut params = Vec::new();
    params.extend(l1.parameters());
    params.extend(l2.parameters());
    let opt = Sgd::new(params, 0.2);

    let initial = {
        let h1 = relu.forward(&l1.forward(&x));
        let out = l2.forward(&h1);
        let pred = sigmoid.forward(&out);
        mse_loss(&pred, &y).data()[0]
    };

    for _ in 0..2000 {
        let h1 = relu.forward(&l1.forward(&x));
        let out = l2.forward(&h1);
        let pred = sigmoid.forward(&out);

        let loss = mse_loss(&pred, &y);
        loss.backward();
        opt.step();
        opt.zero_grad();
    }

    let final_loss = {
        let h1 = relu.forward(&l1.forward(&x));
        let out = l2.forward(&h1);
        let pred = sigmoid.forward(&out);
        mse_loss(&pred, &y).data()[0]
    };

    assert!(
        final_loss < initial,
        "expected final loss {final_loss} < initial loss {initial}"
    );
}
