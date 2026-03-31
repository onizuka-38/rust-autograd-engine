use crate::tensor::{Op, Tensor};

pub fn numel(shape: &[usize]) -> usize {
    shape.iter().product::<usize>()
}

pub fn compute_strides(shape: &[usize]) -> Vec<usize> {
    if shape.is_empty() {
        return Vec::new();
    }
    let mut strides = vec![1; shape.len()];
    for i in (0..shape.len() - 1).rev() {
        strides[i] = strides[i + 1] * shape[i + 1];
    }
    strides
}

fn linear_to_multi(mut index: usize, shape: &[usize]) -> Vec<usize> {
    let strides = compute_strides(shape);
    let mut out = vec![0usize; shape.len()];
    for i in 0..shape.len() {
        let stride = strides[i];
        out[i] = index / stride;
        index %= stride;
    }
    out
}

fn multi_to_linear(index: &[usize], strides: &[usize]) -> usize {
    index
        .iter()
        .zip(strides.iter())
        .map(|(a, b)| a * b)
        .sum::<usize>()
}

pub fn broadcast_shape(a: &[usize], b: &[usize]) -> Vec<usize> {
    let ndim = a.len().max(b.len());
    let mut out = vec![1usize; ndim];

    for i in 0..ndim {
        let a_dim = if i >= ndim - a.len() {
            a[i - (ndim - a.len())]
        } else {
            1
        };
        let b_dim = if i >= ndim - b.len() {
            b[i - (ndim - b.len())]
        } else {
            1
        };
        assert!(
            a_dim == b_dim || a_dim == 1 || b_dim == 1,
            "incompatible shapes for broadcasting: {:?} vs {:?}",
            a,
            b
        );
        out[i] = a_dim.max(b_dim);
    }

    out
}

fn map_out_index_to_in_index(out_index: &[usize], in_shape: &[usize]) -> Vec<usize> {
    let ndim_out = out_index.len();
    let ndim_in = in_shape.len();
    let offset = ndim_out - ndim_in;

    let mut mapped = vec![0usize; ndim_in];
    for i in 0..ndim_in {
        let idx = out_index[offset + i];
        mapped[i] = if in_shape[i] == 1 { 0 } else { idx };
    }
    mapped
}

pub fn broadcast_to(data: &[f32], in_shape: &[usize], out_shape: &[usize]) -> Vec<f32> {
    let in_strides = compute_strides(in_shape);
    let out_len = numel(out_shape);
    let mut out = vec![0.0f32; out_len];

    for (linear, slot) in out.iter_mut().enumerate() {
        let out_index = linear_to_multi(linear, out_shape);
        let in_index = map_out_index_to_in_index(&out_index, in_shape);
        let in_linear = multi_to_linear(&in_index, &in_strides);
        *slot = data[in_linear];
    }

    out
}

pub fn reduce_sum_to_shape(grad_out: &[f32], out_shape: &[usize], in_shape: &[usize]) -> Vec<f32> {
    let in_len = numel(in_shape);
    let in_strides = compute_strides(in_shape);
    let mut out = vec![0.0f32; in_len];

    for (linear, g) in grad_out.iter().enumerate() {
        let out_index = linear_to_multi(linear, out_shape);
        let in_index = map_out_index_to_in_index(&out_index, in_shape);
        let in_linear = multi_to_linear(&in_index, &in_strides);
        out[in_linear] += *g;
    }

    out
}

impl Tensor {
    pub fn add(&self, other: &Tensor) -> Tensor {
        let (left_data, left_shape, left_grad) = {
            let n = self.inner.borrow();
            (n.data.clone(), n.shape.clone(), n.requires_grad)
        };
        let (right_data, right_shape, right_grad) = {
            let n = other.inner.borrow();
            (n.data.clone(), n.shape.clone(), n.requires_grad)
        };

        let out_shape = broadcast_shape(&left_shape, &right_shape);
        let left_b = broadcast_to(&left_data, &left_shape, &out_shape);
        let right_b = broadcast_to(&right_data, &right_shape, &out_shape);

        let data = left_b
            .iter()
            .zip(right_b.iter())
            .map(|(a, b)| a + b)
            .collect::<Vec<_>>();

        Tensor::from_op(
            data,
            out_shape,
            left_grad || right_grad,
            vec![self.clone(), other.clone()],
            Op::Add {
                left_shape,
                right_shape,
            },
        )
    }

    pub fn sub(&self, other: &Tensor) -> Tensor {
        let (left_data, left_shape, left_grad) = {
            let n = self.inner.borrow();
            (n.data.clone(), n.shape.clone(), n.requires_grad)
        };
        let (right_data, right_shape, right_grad) = {
            let n = other.inner.borrow();
            (n.data.clone(), n.shape.clone(), n.requires_grad)
        };

        let out_shape = broadcast_shape(&left_shape, &right_shape);
        let left_b = broadcast_to(&left_data, &left_shape, &out_shape);
        let right_b = broadcast_to(&right_data, &right_shape, &out_shape);

        let data = left_b
            .iter()
            .zip(right_b.iter())
            .map(|(a, b)| a - b)
            .collect::<Vec<_>>();

        Tensor::from_op(
            data,
            out_shape,
            left_grad || right_grad,
            vec![self.clone(), other.clone()],
            Op::Sub {
                left_shape,
                right_shape,
            },
        )
    }

    pub fn mul(&self, other: &Tensor) -> Tensor {
        let (left_data, left_shape, left_grad) = {
            let n = self.inner.borrow();
            (n.data.clone(), n.shape.clone(), n.requires_grad)
        };
        let (right_data, right_shape, right_grad) = {
            let n = other.inner.borrow();
            (n.data.clone(), n.shape.clone(), n.requires_grad)
        };

        let out_shape = broadcast_shape(&left_shape, &right_shape);
        let left_b = broadcast_to(&left_data, &left_shape, &out_shape);
        let right_b = broadcast_to(&right_data, &right_shape, &out_shape);

        let data = left_b
            .iter()
            .zip(right_b.iter())
            .map(|(a, b)| a * b)
            .collect::<Vec<_>>();

        Tensor::from_op(
            data,
            out_shape,
            left_grad || right_grad,
            vec![self.clone(), other.clone()],
            Op::Mul {
                left_data,
                right_data,
                left_shape,
                right_shape,
            },
        )
    }
}
