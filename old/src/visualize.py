import networkx as nx
import matplotlib.pyplot as plt
import os

def plot_graph(graph, filename="data/graph.png"):
    if len(graph) > 100:
        print("Graph too large to plot safely. Skipping visualization.")
        return
    
    G = nx.Graph()

    for node in graph:
        for neighbor in graph[node]:
            G.add_edge(node, neighbor)

    plt.figure(figsize=(8, 6))
    nx.draw(G, with_labels=True, node_color='lightblue', node_size=500)

    # Ensure data folder exists
    os.makedirs("data", exist_ok=True)

    plt.title("Graph Visualization")
    plt.savefig(filename)
    plt.close()

    print(f"Graph saved to {filename}")