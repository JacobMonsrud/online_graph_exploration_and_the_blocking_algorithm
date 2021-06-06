from Graph import Graph
from Vertex import Vertex
from Edge import Edge
from TSP import TSP
from GenerateGraphs import GenarateGraphs
import time
import math

if __name__ == '__main__':
    genrator = GenarateGraphs()
    num_of_nodes = 15
    connectivity = 20
    num_of_graphs = 10
    graphs = genrator.generate(num_of_graphs, num_of_nodes, connectivity)
    filename = "main.txt"
    for graph in graphs:
        delta = 2
        print(f"Blocking with {delta=} and {num_of_nodes=}")
        tsp = TSP()
        t1 = time.time()
        optimal_tour, path = tsp.python_tspm_solver(graph)
        t2 = time.time()
        print("tsp time used: ", str(t2-t1))
        t3 = time.time()
        f = open(filename, "w")
        f.write("")
        f.close()
        online_tour = graph.blocking(graph.nodes[0], delta, filename)
        t4 = time.time()
        print("blocking time used: ", str(t4-t3))
        competitive_ratio = online_tour/optimal_tour
        print(f"{online_tour=}")
        print(f"{optimal_tour=}")
        print(f"{competitive_ratio=}\n")
