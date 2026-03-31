use coregrad::Tensor;

#[test]
fn add_supports_row_broadcast() {
    let a = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2], false);
    let b = Tensor::new(vec![10.0, 20.0], vec![1, 2], false);

    let c = a.add(&b);
    assert_eq!(c.shape(), vec![2, 2]);
    assert_eq!(c.data(), vec![11.0, 22.0, 13.0, 24.0]);
}

#[test]
fn matmul_computes_expected_values() {
    let a = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2], false);
    let b = Tensor::new(vec![5.0, 6.0, 7.0, 8.0], vec![2, 2], false);

    let c = a.matmul(&b);
    assert_eq!(c.shape(), vec![2, 2]);
    assert_eq!(c.data(), vec![19.0, 22.0, 43.0, 50.0]);
}
