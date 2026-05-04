from concurrent.futures import ThreadPoolExecutor

def parallel_bfs(graph, start, num_threads=4):
    visited = set([start])
    distance = {start: 0}
    frontier = [start]

    level = 0

    while frontier:
        next_frontier = []

        def process_node(u):
            local = []
            for w in graph[u]:
                if w not in visited:
                    local.append(w)
            return local

        # Parallel processing of frontier
        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            results = executor.map(process_node, frontier)

        for neighbors in results:
            for w in neighbors:
                if w not in visited:
                    visited.add(w)
                    distance[w] = level + 1
                    next_frontier.append(w)

        frontier = next_frontier
        level += 1

    return distance