use std::collections::HashSet;

use crate::tensor::Tensor;

pub fn topo_sort(root: &Tensor) -> Vec<Tensor> {
    fn dfs(node: &Tensor, visited: &mut HashSet<usize>, order: &mut Vec<Tensor>) {
        let id = node.id();
        if !visited.insert(id) {
            return;
        }

        for parent in node.parents() {
            dfs(&parent, visited, order);
        }

        order.push(node.clone());
    }

    let mut visited = HashSet::new();
    let mut order = Vec::new();
    dfs(root, &mut visited, &mut order);
    order
}
