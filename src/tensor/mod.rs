use std::cell::RefCell;
use std::rc::{Rc, Weak};
use std::sync::atomic::{AtomicUsize, Ordering};

pub mod linalg;
pub mod ops;

static NEXT_NODE_ID: AtomicUsize = AtomicUsize::new(1);

#[derive(Clone)]
pub enum Op {
    None,
    Add {
        left_shape: Vec<usize>,
        right_shape: Vec<usize>,
    },
    Sub {
        left_shape: Vec<usize>,
        right_shape: Vec<usize>,
    },
    Mul {
        left_data: Vec<f32>,
        right_data: Vec<f32>,
        left_shape: Vec<usize>,
        right_shape: Vec<usize>,
    },
    MatMul {
        left_data: Vec<f32>,
        right_data: Vec<f32>,
        left_rows: usize,
        left_cols: usize,
        right_cols: usize,
    },
    Relu {
        input_data: Vec<f32>,
    },
    Sigmoid {
        output_data: Vec<f32>,
    },
    Mean {
        input_shape: Vec<usize>,
    },
    Sum {
        input_shape: Vec<usize>,
    },
}

pub(crate) struct Node {
    pub id: usize,
    pub data: Vec<f32>,
    pub shape: Vec<usize>,
    pub grad: Vec<f32>,
    pub requires_grad: bool,
    pub parents: Vec<Weak<RefCell<Node>>>,
    pub op: Op,
}

#[derive(Clone)]
pub struct Tensor {
    pub(crate) inner: Rc<RefCell<Node>>,
}

impl Tensor {
    pub fn new(data: Vec<f32>, shape: Vec<usize>, requires_grad: bool) -> Self {
        let expected_len = ops::numel(&shape);
        assert_eq!(
            data.len(),
            expected_len,
            "data len {} != shape numel {}",
            data.len(),
            expected_len
        );

        let id = NEXT_NODE_ID.fetch_add(1, Ordering::Relaxed);
        let grad = vec![0.0; data.len()];
        let node = Node {
            id,
            data,
            shape,
            grad,
            requires_grad,
            parents: Vec::new(),
            op: Op::None,
        };

        Self {
            inner: Rc::new(RefCell::new(node)),
        }
    }

    pub(crate) fn from_op(
        data: Vec<f32>,
        shape: Vec<usize>,
        requires_grad: bool,
        parents: Vec<Tensor>,
        op: Op,
    ) -> Self {
        let id = NEXT_NODE_ID.fetch_add(1, Ordering::Relaxed);
        let grad = vec![0.0; data.len()];
        let parent_refs = parents
            .into_iter()
            .map(|t| Rc::downgrade(&t.inner))
            .collect::<Vec<_>>();

        let node = Node {
            id,
            data,
            shape,
            grad,
            requires_grad,
            parents: parent_refs,
            op,
        };

        Self {
            inner: Rc::new(RefCell::new(node)),
        }
    }

    pub fn zeros(shape: Vec<usize>, requires_grad: bool) -> Self {
        let data = vec![0.0; ops::numel(&shape)];
        Self::new(data, shape, requires_grad)
    }

    pub fn from_scalar(v: f32, requires_grad: bool) -> Self {
        Self::new(vec![v], vec![1], requires_grad)
    }

    pub fn zero_like(&self) -> Self {
        let shape = self.shape();
        Self::zeros(shape, false)
    }

    pub fn ones_like(&self) -> Self {
        let shape = self.shape();
        let len = ops::numel(&shape);
        Self::new(vec![1.0; len], shape, false)
    }

    pub fn shape(&self) -> Vec<usize> {
        self.inner.borrow().shape.clone()
    }

    pub fn data(&self) -> Vec<f32> {
        self.inner.borrow().data.clone()
    }

    pub fn grad(&self) -> Vec<f32> {
        self.inner.borrow().grad.clone()
    }

    pub fn requires_grad(&self) -> bool {
        self.inner.borrow().requires_grad
    }

    pub fn zero_grad(&self) {
        let mut n = self.inner.borrow_mut();
        n.grad.fill(0.0);
    }

    pub fn set_grad_scalar(&self, g: f32) {
        let mut n = self.inner.borrow_mut();
        assert_eq!(n.grad.len(), 1, "set_grad_scalar expects scalar tensor");
        n.grad[0] = g;
    }

    pub fn mean(&self) -> Tensor {
        let (sum, len, requires_grad, shape) = {
            let n = self.inner.borrow();
            (
                n.data.iter().copied().sum::<f32>(),
                n.data.len(),
                n.requires_grad,
                n.shape.clone(),
            )
        };
        let value = sum / len as f32;

        Tensor::from_op(
            vec![value],
            vec![1],
            requires_grad,
            vec![self.clone()],
            Op::Mean { input_shape: shape },
        )
    }

    pub fn sum(&self) -> Tensor {
        let (sum, requires_grad, shape) = {
            let n = self.inner.borrow();
            (
                n.data.iter().copied().sum::<f32>(),
                n.requires_grad,
                n.shape.clone(),
            )
        };

        Tensor::from_op(
            vec![sum],
            vec![1],
            requires_grad,
            vec![self.clone()],
            Op::Sum { input_shape: shape },
        )
    }

    pub fn relu(&self) -> Tensor {
        let (input, shape, requires_grad) = {
            let n = self.inner.borrow();
            (n.data.clone(), n.shape.clone(), n.requires_grad)
        };
        let out = input
            .iter()
            .map(|v| if *v > 0.0 { *v } else { 0.0 })
            .collect::<Vec<_>>();

        Tensor::from_op(
            out,
            shape,
            requires_grad,
            vec![self.clone()],
            Op::Relu { input_data: input },
        )
    }

    pub fn sigmoid(&self) -> Tensor {
        let (input, shape, requires_grad) = {
            let n = self.inner.borrow();
            (n.data.clone(), n.shape.clone(), n.requires_grad)
        };
        let out = input
            .iter()
            .map(|v| 1.0f32 / (1.0 + (-*v).exp()))
            .collect::<Vec<_>>();

        Tensor::from_op(
            out.clone(),
            shape,
            requires_grad,
            vec![self.clone()],
            Op::Sigmoid { output_data: out },
        )
    }

    pub fn backward(&self) {
        crate::autograd::backward(self);
    }

    pub fn apply_grad(&self, lr: f32) {
        let mut n = self.inner.borrow_mut();
        if !n.requires_grad {
            return;
        }
        for i in 0..n.data.len() {
            n.data[i] -= lr * n.grad[i];
        }
    }

    pub(crate) fn id(&self) -> usize {
        self.inner.borrow().id
    }

    pub(crate) fn parents(&self) -> Vec<Tensor> {
        self.inner
            .borrow()
            .parents
            .iter()
            .filter_map(|w| w.upgrade())
            .map(|rc| Tensor { inner: rc })
            .collect::<Vec<_>>()
    }

    pub(crate) fn op(&self) -> Op {
        self.inner.borrow().op.clone()
    }

    pub(crate) fn add_grad(&self, grad: &[f32]) {
        let mut n = self.inner.borrow_mut();
        if !n.requires_grad {
            return;
        }
        assert_eq!(n.grad.len(), grad.len(), "gradient size mismatch");
        for (idx, g) in grad.iter().enumerate() {
            n.grad[idx] += *g;
        }
    }

    pub(crate) fn grad_data_shape(&self) -> (Vec<f32>, Vec<f32>, Vec<usize>, bool) {
        let n = self.inner.borrow();
        (n.grad.clone(), n.data.clone(), n.shape.clone(), n.requires_grad)
    }
}
