use coregrad::nn::linear::Linear;
use coregrad::{mse_loss, Module, Sgd, Tensor};

fn approx_eq(a: f32, b: f32, eps: f32) -> bool {
    (a - b).abs() <= eps
}

#[test]
fn backward_scalar_expression_matches_manual_gradient() {
    let a = Tensor::from_scalar(2.0, true);
    let b = Tensor::from_scalar(3.0, true);

    let y = a.mul(&b).add(&a).mean();
    y.backward();

    let ga = a.grad()[0];
    let gb = b.grad()[0];
    assert!(approx_eq(ga, 4.0, 1e-6));
    assert!(approx_eq(gb, 2.0, 1e-6));
}

#[test]
fn sgd_step_reduces_loss_on_simple_linear_fit() {
    let x = Tensor::new(vec![0.0, 1.0, 2.0, 3.0], vec![4, 1], false);
    let y = Tensor::new(vec![0.0, 2.0, 4.0, 6.0], vec![4, 1], false);

    let linear = Linear::new(1, 1);
    let opt = Sgd::new(linear.parameters(), 0.05);

    let pred0 = linear.forward(&x);
    let loss0 = mse_loss(&pred0, &y);
    let start = loss0.data()[0];

    for _ in 0..300 {
        let pred = linear.forward(&x);
        let loss = mse_loss(&pred, &y);
        loss.backward();
        opt.step();
        opt.zero_grad();
    }

    let pred1 = linear.forward(&x);
    let loss1 = mse_loss(&pred1, &y);
    let end = loss1.data()[0];

    assert!(end < start, "expected end loss {end} < start loss {start}");
}
