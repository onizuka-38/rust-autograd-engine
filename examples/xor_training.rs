use coregrad::nn::activations::{ReLU, Sigmoid};
use coregrad::nn::linear::Linear;
use coregrad::{mse_loss, Module, Sgd, Tensor};

fn main() {
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

    for epoch in 0..3000 {
        let h1 = l1.forward(&x);
        let h1 = relu.forward(&h1);
        let out = l2.forward(&h1);
        let pred = sigmoid.forward(&out);

        let loss = mse_loss(&pred, &y);
        loss.backward();

        opt.step();
        opt.zero_grad();

        if epoch % 500 == 0 {
            println!("epoch={epoch} loss={:.6}", loss.data()[0]);
        }
    }

    let h1 = l1.forward(&x);
    let h1 = relu.forward(&h1);
    let out = l2.forward(&h1);
    let pred = sigmoid.forward(&out);
    println!("pred={:?}", pred.data());
}
