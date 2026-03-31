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
fn add_supports_scalar_to_matrix_broadcast() {
    let a = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2], false);
    let s = Tensor::from_scalar(5.0, false);

    let c = a.add(&s);
    assert_eq!(c.shape(), vec![2, 2]);
    assert_eq!(c.data(), vec![6.0, 7.0, 8.0, 9.0]);
}

#[test]
fn mul_supports_scalar_to_vector_broadcast() {
    let v = Tensor::new(vec![2.0, 4.0, 6.0], vec![3], false);
    let s = Tensor::from_scalar(0.5, false);

    let out = v.mul(&s);
    assert_eq!(out.shape(), vec![3]);
    assert_eq!(out.data(), vec![1.0, 2.0, 3.0]);
}

#[test]
fn add_supports_vector_to_3d_broadcast() {
    let a = Tensor::new((1..=8).map(|x| x as f32).collect::<Vec<_>>(), vec![2, 2, 2], false);
    let b = Tensor::new(vec![10.0, 20.0], vec![2], false);

    let out = a.add(&b);
    assert_eq!(out.shape(), vec![2, 2, 2]);
    assert_eq!(out.data(), vec![11.0, 22.0, 13.0, 24.0, 15.0, 26.0, 17.0, 28.0]);
}

#[test]
fn sub_supports_column_broadcast() {
    let a = Tensor::new(vec![10.0, 20.0, 30.0, 40.0], vec![2, 2], false);
    let b = Tensor::new(vec![1.0, 2.0], vec![2, 1], false);

    let out = a.sub(&b);
    assert_eq!(out.shape(), vec![2, 2]);
    assert_eq!(out.data(), vec![9.0, 19.0, 28.0, 38.0]);
}

#[test]
fn mul_supports_2d_to_3d_broadcast() {
    let a = Tensor::new((1..=8).map(|x| x as f32).collect::<Vec<_>>(), vec![2, 2, 2], false);
    let b = Tensor::new(vec![1.0, 10.0, 100.0, 1000.0], vec![2, 2], false);

    let out = a.mul(&b);
    assert_eq!(out.shape(), vec![2, 2, 2]);
    assert_eq!(out.data(), vec![1.0, 20.0, 300.0, 4000.0, 5.0, 60.0, 700.0, 8000.0]);
}

#[test]
fn zero_like_and_ones_like_follow_shape() {
    let x = Tensor::new(vec![3.0, 4.0, 5.0, 6.0], vec![2, 2], true);

    let z = x.zero_like();
    let o = x.ones_like();

    assert_eq!(z.shape(), vec![2, 2]);
    assert_eq!(o.shape(), vec![2, 2]);
    assert_eq!(z.data(), vec![0.0, 0.0, 0.0, 0.0]);
    assert_eq!(o.data(), vec![1.0, 1.0, 1.0, 1.0]);
    assert!(!z.requires_grad());
    assert!(!o.requires_grad());
}

#[test]
fn matmul_computes_expected_values() {
    let a = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2], false);
    let b = Tensor::new(vec![5.0, 6.0, 7.0, 8.0], vec![2, 2], false);

    let c = a.matmul(&b);
    assert_eq!(c.shape(), vec![2, 2]);
    assert_eq!(c.data(), vec![19.0, 22.0, 43.0, 50.0]);
}
