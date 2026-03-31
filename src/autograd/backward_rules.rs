use crate::tensor::ops::{broadcast_shape, broadcast_to, reduce_sum_to_shape};
use crate::tensor::{Op, Tensor};

pub(crate) fn apply(node: &Tensor, grad_out: &[f32], out_shape: &[usize], parents: &[Tensor]) {
    match node.op() {
        Op::None => {}
        Op::Add {
            left_shape,
            right_shape,
        } => {
            if parents.len() == 2 {
                let grad_left = reduce_sum_to_shape(grad_out, out_shape, &left_shape);
                let grad_right = reduce_sum_to_shape(grad_out, out_shape, &right_shape);
                parents[0].add_grad(&grad_left);
                parents[1].add_grad(&grad_right);
            }
        }
        Op::Sub {
            left_shape,
            right_shape,
        } => {
            if parents.len() == 2 {
                let grad_left = reduce_sum_to_shape(grad_out, out_shape, &left_shape);
                let mut grad_right = reduce_sum_to_shape(grad_out, out_shape, &right_shape);
                for g in &mut grad_right {
                    *g = -*g;
                }
                parents[0].add_grad(&grad_left);
                parents[1].add_grad(&grad_right);
            }
        }
        Op::Mul {
            left_data,
            right_data,
            left_shape,
            right_shape,
        } => {
            if parents.len() == 2 {
                let local_shape = broadcast_shape(&left_shape, &right_shape);
                let left_b = broadcast_to(&left_data, &left_shape, &local_shape);
                let right_b = broadcast_to(&right_data, &right_shape, &local_shape);

                let grad_left_full = grad_out
                    .iter()
                    .zip(right_b.iter())
                    .map(|(g, r)| g * r)
                    .collect::<Vec<_>>();
                let grad_right_full = grad_out
                    .iter()
                    .zip(left_b.iter())
                    .map(|(g, l)| g * l)
                    .collect::<Vec<_>>();

                let grad_left = reduce_sum_to_shape(&grad_left_full, &local_shape, &left_shape);
                let grad_right = reduce_sum_to_shape(&grad_right_full, &local_shape, &right_shape);

                parents[0].add_grad(&grad_left);
                parents[1].add_grad(&grad_right);
            }
        }
        Op::MatMul {
            left_data,
            right_data,
            left_rows,
            left_cols,
            right_cols,
        } => {
            if parents.len() == 2 {
                let mut grad_left = vec![0.0f32; left_rows * left_cols];
                for i in 0..left_rows {
                    for k in 0..left_cols {
                        let mut acc = 0.0f32;
                        for j in 0..right_cols {
                            acc += grad_out[i * right_cols + j] * right_data[k * right_cols + j];
                        }
                        grad_left[i * left_cols + k] += acc;
                    }
                }

                let mut grad_right = vec![0.0f32; left_cols * right_cols];
                for k in 0..left_cols {
                    for j in 0..right_cols {
                        let mut acc = 0.0f32;
                        for i in 0..left_rows {
                            acc += left_data[i * left_cols + k] * grad_out[i * right_cols + j];
                        }
                        grad_right[k * right_cols + j] += acc;
                    }
                }

                parents[0].add_grad(&grad_left);
                parents[1].add_grad(&grad_right);
            }
        }
        Op::Relu { input_data } => {
            if parents.len() == 1 {
                let grad = grad_out
                    .iter()
                    .zip(input_data.iter())
                    .map(|(g, x)| if *x > 0.0 { *g } else { 0.0 })
                    .collect::<Vec<_>>();
                parents[0].add_grad(&grad);
            }
        }
        Op::Sigmoid { output_data } => {
            if parents.len() == 1 {
                let grad = grad_out
                    .iter()
                    .zip(output_data.iter())
                    .map(|(g, y)| g * y * (1.0 - y))
                    .collect::<Vec<_>>();
                parents[0].add_grad(&grad);
            }
        }
        Op::Mean { input_shape } => {
            if parents.len() == 1 {
                let denom = input_shape.iter().product::<usize>() as f32;
                let each = grad_out[0] / denom;
                let grad = vec![each; input_shape.iter().product::<usize>()];
                parents[0].add_grad(&grad);
            }
        }
        Op::Sum { input_shape } => {
            if parents.len() == 1 {
                let grad = vec![grad_out[0]; input_shape.iter().product::<usize>()];
                parents[0].add_grad(&grad);
            }
        }
    }
}
