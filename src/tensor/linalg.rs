use crate::tensor::{Op, Tensor};

impl Tensor {
    pub fn matmul(&self, other: &Tensor) -> Tensor {
        let (left_data, left_shape, left_grad) = {
            let n = self.inner.borrow();
            (n.data.clone(), n.shape.clone(), n.requires_grad)
        };
        let (right_data, right_shape, right_grad) = {
            let n = other.inner.borrow();
            (n.data.clone(), n.shape.clone(), n.requires_grad)
        };

        assert_eq!(left_shape.len(), 2, "left tensor must be 2D");
        assert_eq!(right_shape.len(), 2, "right tensor must be 2D");

        let m = left_shape[0];
        let k_left = left_shape[1];
        let k_right = right_shape[0];
        let n = right_shape[1];

        assert_eq!(k_left, k_right, "matmul dimension mismatch");

        let mut out = vec![0.0f32; m * n];
        for i in 0..m {
            for j in 0..n {
                let mut acc = 0.0f32;
                for k in 0..k_left {
                    acc += left_data[i * k_left + k] * right_data[k * n + j];
                }
                out[i * n + j] = acc;
            }
        }

        Tensor::from_op(
            out,
            vec![m, n],
            left_grad || right_grad,
            vec![self.clone(), other.clone()],
            Op::MatMul {
                left_data,
                right_data,
                left_rows: m,
                left_cols: k_left,
                right_cols: n,
            },
        )
    }
}
