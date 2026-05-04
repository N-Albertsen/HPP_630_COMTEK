from multiprocessing import Pool


def process_node(args):
    u, graph, visited = args
    neighbors = []

    for w in graph[u]:
        if w not in visited:
            neighbors.append(w)

    return neighbors


def multiprocessing_bfs(graph, start, num_processes=4):
    visited = set([start])
    distance = {start: 0}
    frontier = [start]

    level = 0

    while frontier:
        next_frontier = []

        with Pool(processes=num_processes) as pool:
            results = pool.map(
                process_node,
                [(u, graph, visited) for u in frontier]
            )

        for neighbors in results:
            for w in neighbors:
                if w not in visited:
                    visited.add(w)
                    distance[w] = level + 1
                    next_frontier.append(w)

        frontier = next_frontier
        level += 1

    return distance