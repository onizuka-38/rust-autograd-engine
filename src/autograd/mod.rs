mod backward_rules;
mod graph;

use crate::tensor::Tensor;

pub use graph::topo_sort;

pub fn backward(loss: &Tensor) {
    let shape = loss.shape();
    assert_eq!(shape, vec![1], "backward expects scalar loss tensor");

    loss.set_grad_scalar(1.0);
    let topo = topo_sort(loss);

    for node in topo.into_iter().rev() {
        let (grad_out, _data, out_shape, requires_grad) = node.grad_data_shape();
        if !requires_grad {
            continue;
        }

        let parents = node.parents();
        backward_rules::apply(&node, &grad_out, &out_shape, &parents);
    }
}
