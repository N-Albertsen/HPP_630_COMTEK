"""Exercise 1B.II — Small illustrative graph and level-by-level BFS DAG"""

# 7-node graph
#       0
#      / \
#     1   2
#    /|   |\
#   3 4   5 6
adj = {
    0: [1, 2],
    1: [0, 3, 4],
    2: [0, 5, 6],
    3: [1], 4: [1], 5: [2], 6: [2],
}


def bfs_levels(adj, s):
    """BFS that returns the frontier of each level (the parallel DAG)."""
    dist = {v: None for v in adj}
    dist[s] = 0
    levels = [[s]]
    while True:
        nxt = []
        for u in levels[-1]:
            for w in adj[u]:
                if dist[w] is None:
                    dist[w] = dist[u] + 1
                    nxt.append(w)
        if not nxt:
            return levels
        levels.append(nxt)


if __name__ == "__main__":
    L = bfs_levels(adj, 0)
    print("Level-synchronous BFS from source 0:\n")
    for k, F in enumerate(L):
        tasks = "   ".join(f"explore({v})" for v in F)
        print(f"  L{k}:  {tasks}     ← parallel within level")
        if k < len(L) - 1:
            print(f"         │")
            print(f"         ▼ barrier")
    print(f"\nDepth = {len(L)} levels.   Total vertices = {sum(map(len, L))}.")
