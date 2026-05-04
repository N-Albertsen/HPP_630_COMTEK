import numpy as np
from numba import njit


@njit
def bfs_numba(offsets, neighbors, start):
    n = len(offsets) - 1

    visited = np.zeros(n, dtype=np.bool_)
    distance = np.full(n, -1)

    queue = np.empty(n, dtype=np.int64)
    front = 0
    back = 0

    visited[start] = True
    distance[start] = 0
    queue[back] = start
    back += 1

    while front < back:
        u = queue[front]
        front += 1

        for i in range(offsets[u], offsets[u + 1]):
            w = neighbors[i]

            if not visited[w]:
                visited[w] = True
                distance[w] = distance[u] + 1
                queue[back] = w
                back += 1

    return distance