from collections import deque

def bfs(graph, start):
    visited = set()
    queue = deque([start])
    order = []

    #print(f"Queue: {list(queue)}")

    while queue:
        node = queue.popleft()

        if node not in visited:
            #print(f"Visiting: {node}")

            visited.add(node)
            order.append(node)

            for neighbor in graph[node]:
                if neighbor not in visited:
                    #print(f"Adding neighbor to queue: {neighbor}")
                    queue.append(neighbor)

        #print(f"Queue: {list(queue)}")

    #print("BFS Traversal Order:", order)
    return order