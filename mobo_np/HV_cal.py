import numpy as np
from typing import Callable, List, Optional, Union
import matplotlib.pyplot as plt

class Node:
    r"""Node in the MultiList data structure."""

    def __init__(
        self,
        m: int,
        dtype: np.dtype,
        data: Optional[np.ndarray] = None,
    ) -> None:
        r"""Initialize MultiList.

        Args:
            m: The number of objectives
            dtype: The dtype
            data: The array data to be stored in this Node.
        """
        self.data = data
        self.next = [None] * m
        self.prev = [None] * m
        self.ignore = 0
        self.area = np.zeros(m, dtype=dtype)
        self.volume = np.zeros_like(self.area)

class MultiList:
    r"""A special data structure used in hypervolume computation.

    It consists of several doubly linked lists that share common nodes.
    Every node has multiple predecessors and successors, one in every list.
    """

    def __init__(self, m: int, dtype: np.dtype) -> None:
        r"""Initialize `m` doubly linked lists.

        Args:
            m: number of doubly linked lists
            dtype: the dtype

        """
        self.m = m
        self.sentinel = Node(m=m, dtype=dtype)
        self.sentinel.next = [self.sentinel] * m
        self.sentinel.prev = [self.sentinel] * m

    def append(self, node: Node, index: int) -> None:
        r"""Appends a node to the end of the list at the given index.

        Args:
            node: the new node
            index: the index where the node should be appended.
        """
        last = self.sentinel.prev[index]
        node.next[index] = self.sentinel
        node.prev[index] = last
        # set the last element as the new one
        self.sentinel.prev[index] = node
        last.next[index] = node

    def extend(self, nodes: List[Node], index: int) -> None:
        r"""Extends the list at the given index with the nodes.

        Args:
            nodes: list of nodes to append at the given index.
            index: the index where the nodes should be appended.

        """
        for node in nodes:
            self.append(node=node, index=index)

    def remove(self, node: Node, index: int, bounds: np.ndarray) -> Node:
        r"""Removes and returns 'node' from all lists in [0, 'index'].
        """
        for i in range(index):
            predecessor = node.prev[i]
            successor = node.next[i]
            predecessor.next[i] = successor
            successor.prev[i] = predecessor
        bounds[...] = np.minimum(bounds, node.data)
        return node

    def reinsert(self, node: Node, index: int, bounds: np.ndarray) -> None:
        r"""Re-inserts the node at its original position.

        Re-inserts the node at its original position in all lists in [0, 'index']
        before it was removed. This method assumes that the next and previous
        nodes of the node that is reinserted are in the list.

        Args:
            node: The node
            index: The upper bound on the range of indices
            bounds: A `2 x m`-dim np.array bounds on the objectives

        """
        for i in range(index):
            node.prev[i].next[i] = node
            node.next[i].prev[i] = node
        bounds[...] = np.minimum(bounds, node.data)

class Hypervolume:
    def __init__(self, ref_point):
        self.ref_point = ref_point
        self._ref_point = -ref_point

    def compute(self, pareto_Y):
        pareto_Y = - pareto_Y
        better_than_ref = (pareto_Y <= self._ref_point).all(axis=-1)
        pareto_Y = pareto_Y[better_than_ref]
        pareto_Y = pareto_Y - self._ref_point
        self._initialize_multilist(pareto_Y)
        bounds = np.full_like(self._ref_point, float("-inf"))
        return self._hv_recursive(
            i=self._ref_point.shape[0] - 1, n_pareto=pareto_Y.shape[0], bounds=bounds
        )

    def _hv_recursive(self, i, n_pareto, bounds):
        hvol = 0.0
        sentinel = self.list.sentinel
        if n_pareto == 0:
            return hvol
        elif i == 0:
            return -sentinel.next[0].data[0]
        elif i == 1:
            q = sentinel.next[1]
            h = q.data[0]
            p = q.next[1]
            while p is not sentinel:
                hvol += h * (q.data[1] - p.data[1])
                if p.data[0] < h:
                    h = p.data[0]
                q = p
                p = q.next[1]
            hvol += h * q.data[1]
            return hvol
        else:
            p = sentinel
            q = p.prev[i]
            while q.data is not None:
                if q.ignore < i:
                    q.ignore = 0
                q = q.prev[i]
            q = p.prev[i]
            while n_pareto > 1 and (
                q.data[i] > bounds[i] or q.prev[i].data[i] >= bounds[i]
            ):
                p = q
                self.list.remove(p, i, bounds)
                q = p.prev[i]
                n_pareto -= 1
            q_prev = q.prev[i]
            if n_pareto > 1:
                hvol = q_prev.volume[i] + q_prev.area[i] * (q.data[i] - q_prev.data[i])
            else:
                q.area[0] = 1
                q.area[1 : i + 1] = q.area[:i] * -(q.data[:i])
            q.volume[i] = hvol
            if q.ignore >= i:
                q.area[i] = q_prev.area[i]
            else:
                q.area[i] = self._hv_recursive(i - 1, n_pareto, bounds)
                if q.area[i] <= q_prev.area[i]:
                    q.ignore = i
            while p is not sentinel:
                p_data = p.data[i]
                hvol += q.area[i] * (p_data - q.data[i])
                bounds[i] = p_data
                self.list.reinsert(p, i, bounds)
                n_pareto += 1
                q = p
                p = p.next[i]
                q.volume[i] = hvol
                if q.ignore >= i:
                    q.area[i] = q.prev[i].area[i]
                else:
                    q.area[i] = self._hv_recursive(i - 1, n_pareto, bounds)
                    if q.area[i] <= q.prev[i].area[i]:
                        q.ignore = i
            hvol -= q.area[i] * q.data[i]
            return hvol
    
    def plot(self, ref_point, pareto_Y):
        m = pareto_Y.shape[1]
        bounds = np.full_like(ref_point, float("-inf"))
        self._initialize_multilist(pareto_Y)
        self._plot_recursive(i=m - 1, n_pareto=pareto_Y.shape[0], bounds=bounds, ref_point=ref_point)

    def _plot_recursive(self, i, n_pareto, bounds, ref_point):
        sentinel = self.list.sentinel
        if n_pareto == 0 or i == 0:
            return
        elif i == 1:
            q = sentinel.next[1]
            h = q.data[0]
            p = q.next[1]
            while p is not sentinel:
                plt.fill_betweenx([h, q.data[1]], q.data[0], p.data[0], color='gray', alpha=0.5)
                h = p.data[0]
                q = p
                p = q.next[1]
            plt.fill_betweenx([h, q.data[1]], q.data[0], ref_point[0], color='gray', alpha=0.5)
        else:
            p = sentinel
            q = p.prev[i]
            while q.data is not None:
                if q.ignore < i:
                    q.ignore = 0
                q = q.prev[i]
            q = p.prev[i]
            while n_pareto > 1 and (q.data[i] > bounds[i] or q.prev[i].data[i] >= bounds[i]):
                p = q
                self.list.remove(p, i, bounds)
                q = p.prev[i]
                n_pareto -= 1
            q_prev = q.prev[i]
            if n_pareto > 1:
                plt.fill_between(
                    [q_prev.data[i], q.data[i]],
                    q_prev.data[i],
                    q_prev.data[i] + q_prev.area[i] * (q.data[i] - q_prev.data[i]) / q_prev.volume[i],
                    color='gray',
                    alpha=0.5
                )
            else:
                plt.fill_between(
                    [q_prev.data[i], q.data[i]],
                    q_prev.data[i],
                    ref_point[i],
                    color='gray',
                    alpha=0.5
                )
            if q.ignore >= i:
                q.area[i] = q_prev.area[i]
            else:
                self._plot_recursive(i - 1, n_pareto, bounds, ref_point)
        while p is not sentinel:
            p_data = p.data[i]
            plt.fill_betweenx([p_data, q.data[i]], q.data[i], ref_point[i], color='gray', alpha=0.5)
            bounds[i] = p_data
            self.list.reinsert(p, i, bounds)
            n_pareto += 1
            q = p
            p = p.next[i]
            if q.ignore >= i:
                q.area[i] = q.prev[i].area[i]
            else:
                self._plot_recursive(i - 1, n_pareto, bounds, ref_point)

    def _initialize_multilist(self, pareto_Y):
        m = pareto_Y.shape[-1]
        nodes = [
            Node(m=m, dtype=pareto_Y.dtype, data=point)
            for point in pareto_Y
        ]
        self.list = MultiList(m=m, dtype=pareto_Y.dtype)
        for i in range(m):
            sort_by_dimension(nodes, i)
            self.list.extend(nodes, i)


def sort_by_dimension(nodes: List[Node], i: int) -> None:
    r"""Sorts the list of nodes in-place by the specified objective.

    Args:
        nodes: A list of Nodes
        i: The index of the objective to sort by

    """
    # build a list of tuples of (point[i], node)
    decorated = [(node.data[i], index, node) for index, node in enumerate(nodes)]
    # sort by this value
    decorated.sort()
    # write back to original list
    nodes[:] = [node for (_, _, node) in decorated]


def plot_pareto_hv(pareto_Y, ref_point):
    # Compute Hypervolume
    hv = Hypervolume(ref_point)
    hypervolume_value = hv.compute(pareto_Y)

    # Plot Pareto Front
    fig = plt.figure()
    if pareto_Y.shape[1] == 2:
        plt.scatter(pareto_Y[:, 0], pareto_Y[:, 1], label="Pareto Front", c='blue', marker='o')
        plt.scatter(ref_point[0], ref_point[1], label="Reference Point", c='red', marker='x')
        plt.xlabel("Objective 1")
        plt.ylabel("Objective 2")
    elif pareto_Y.shape[1] == 3:
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(pareto_Y[:, 0], pareto_Y[:, 1], pareto_Y[:, 2], label="Pareto Front", c='blue', marker='o')
        ax.scatter(ref_point[0], ref_point[1], ref_point[2], label="Reference Point", c='red', marker='x')
        ax.set_xlabel("Objective 1")
        ax.set_ylabel("Objective 2")
        ax.set_zlabel("Objective 3")

    # Plot Hypervolume TODO: try a way to visualize hypervolume
    # hv.plot(ref_point, pareto_Y)

    # Display Hypervolume
    plt.title(f'Hypervolume: {hypervolume_value:.4f}')
    plt.legend()
    plt.show()
