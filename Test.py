import copy
import random
import unittest
from fractions import Fraction

from Graph import Graph
from Vertex import Vertex
from Edge import Edge
from TSP import TSP
from GenerateGraphs import GenarateGraphs
import math
import numpy as np


class TestGraphClass(unittest.TestCase):

    def setUp(self):
        self.a = Vertex("A")
        self.b = Vertex("B")
        self.c = Vertex("C")
        self.d = Vertex("D")
        self.e = Vertex("E")
        self.f = Vertex("F")
        self.g = Vertex("G")
        self.a_b = Edge(self.a, self.b, 5)
        self.a_c = Edge(self.a, self.c, 2)
        self.b_c = Edge(self.b, self.c, 3)
        self.c_f = Edge(self.c, self.f, 6)
        self.f_e = Edge(self.f, self.e, 3)
        self.d_e = Edge(self.d, self.e, 1)
        self.g_a = Edge(self.a, self.g, 7)
        self.f_d = Edge(self.f, self.d, 1)
        self.e_g = Edge(self.e, self.g, 9)
        self.nodes = [self.a, self.b, self.c, self.d, self.e, self.f, self.g]
        self.edges = [self.a_b, self.a_c, self.b_c, self.c_f, self.f_e, self.d_e, self.g_a, self.f_d, self.e_g]
        self.graph1 = Graph(self.nodes, self.edges)

        self.a2 = Vertex("A")
        self.b2 = Vertex("B")
        self.c2 = Vertex("C")
        self.d2 = Vertex("D")
        self.f2 = Vertex("F")
        self.u2 = Vertex("U")
        self.v2 = Vertex("V")
        self.a_b2 = Edge(self.a2, self.b2, 2)
        self.a_d2 = Edge(self.a2, self.d2, 1)
        self.b_u2 = Edge(self.b2, self.u2, 5)
        self.b_c2 = Edge(self.b2, self.c2, 8)
        self.d_u2 = Edge(self.d2, self.u2, 3)
        self.d_f2 = Edge(self.d2, self.f2, 4)
        self.u_v2 = Edge(self.u2, self.v2, 5)
        self.f_v2 = Edge(self.f2, self.v2, 2)
        nodes = [self.a2, self.b2, self.c2, self.d2, self.f2, self.u2, self.v2]
        edges = [self.a_b2, self.a_d2, self.b_c2, self.b_u2, self.d_u2, self.d_f2, self.u_v2, self.f_v2]
        self.graph2 = Graph(nodes, edges)

    def testInit(self):
        adjList = {self.a: [self.a_b, self.a_c, self.g_a],
                   self.b: [self.a_b, self.b_c],
                   self.c: [self.a_c, self.b_c, self.c_f],
                   self.d: [self.d_e, self.f_d],
                   self.e: [self.d_e, self.f_e, self.e_g],
                   self.f: [self.f_e, self.c_f, self.f_d],
                   self.g: [self.g_a, self.e_g]}

        for edges in adjList.values():
            edges.sort()
        for edges in self.graph1.adjacencyList.values():
            edges.sort()
        self.assertEqual(adjList, self.graph1.adjacencyList)

    def test_initialize_single_source(self):
        self.graph1.initialize_single_source(self.b)
        self.assertEqual(self.a.d, math.inf)
        self.assertEqual(self.c.d, math.inf)
        self.assertEqual(self.f.d, math.inf)
        self.assertEqual(self.e.d, math.inf)
        self.assertEqual(self.g.d, math.inf)
        self.assertEqual(self.b.d, 0)
        self.assertEqual(self.a.pi, None)
        self.assertEqual(self.b.pi, None)
        self.assertEqual(self.d.pi, None)
        self.assertEqual(self.e.pi, None)
        self.assertEqual(self.g.pi, None)

    def test_relax(self):
        self.graph1.initialize_single_source(self.b)
        self.assertEqual(self.a.d, math.inf)
        self.graph1.relax(self.b, self.a, self.a_b)
        self.assertEqual(self.a.d, 5)

    def test_dijkstra(self):
        self.graph1.dijkstra(self.a)
        self.assertEqual(self.c.d, 2)
        self.assertEqual(self.g.d, 7)
        self.graph1.dijkstra(self.b)
        self.assertEqual(self.e.d, 11)
        self.graph1.dijkstra(self.d)
        self.assertEqual(self.f.d, 1)
        self.graph1.dijkstra(self.g)
        self.assertEqual(self.f.d, 11)

    def test_dijkstra_inf_error(self):
        o = Vertex("O")
        a = Vertex("A")
        b = Vertex("B")
        c = Vertex("C")
        d = Vertex("D")
        e = Vertex("E")
        o_a = Edge(o, a, 1)
        o_b = Edge(o, b, 2)
        a_c = Edge(a, c, 3)
        a_d = Edge(a, d, 4)
        b_c = Edge(b, c, 5)
        b_e = Edge(b, e, 8)
        nodes = [o, a, b, c, d, e]
        edges = [o_a, o_b, a_c, a_d, b_c, b_e]
        graph = Graph(nodes, edges)
        # graph.tikz("error.txt")
        graph.dijkstra(d)
        self.assertEqual(15, graph.label_to_node(e.label).d)
        self.assertEqual(15, graph.label_to_node(e.label).d)

    def test_get_known_graph(self):
        self.a.explored = True
        self.b.explored = True
        known_graph: Graph = self.graph1.get_known_graph()
        self.assertEqual([self.a_b, self.a_c, self.b_c, self.g_a].sort(), known_graph.edges.sort())
        self.assertEqual([self.a, self.c, self.b, self.g].sort(), known_graph.nodes.sort())

        adjList = {self.a: [self.a_b, self.a_c, self.g_a],
                   self.b: [self.a_b, self.b_c],
                   self.c: [self.a_c, self.b_c],
                   self.g: [self.g_a]}

        for edges in adjList.values():
            edges.sort()
        for edges in known_graph.adjacencyList.values():
            edges.sort()

        self.assertEqual(adjList, known_graph.adjacencyList)

    def test_dijkstra_only_know_edges(self):
        self.a.explored = True
        self.b.explored = True
        known_graph: Graph = self.graph1.get_known_graph()

        known_graph.dijkstra(self.a)
        self.assertEqual(known_graph.label_to_node(self.c.label).d, 2)
        known_graph.dijkstra(self.g)
        self.assertEqual(known_graph.label_to_node(self.c.label).d, 9)
        self.assertIsNone(known_graph.label_to_node(self.f.label))

    def test_is_blocked(self):
        self.a.explored = True
        self.b.explored = True
        delta = 1.5
        self.assertFalse(self.graph1.is_blocked(self.a_c, delta))
        self.assertTrue(self.graph1.is_blocked(self.g_a, delta))
        # The shortest path from B=u to C=v'=v is b_c, which is shorter than (1+delta)*b_c and there is a shorter boundary edge, namely a_c.
        # Notice that the shortests path goes through the edge which is blocked. Which seems a byte unintuative.
        self.assertTrue(self.graph1.is_blocked(self.b_c, delta))
        self.assertFalse(self.graph1.is_blocked(self.c_f, delta))

    def test_is_blocked_2(self):
        o = Vertex("O")
        a = Vertex("A")
        b = Vertex("B")
        c = Vertex("C")
        d = Vertex("D")
        e = Vertex("E")
        o_a = Edge(o, a, 1)
        o_b = Edge(o, b, 2)
        a_c = Edge(a, c, 3)
        a_d = Edge(a, d, 4)
        b_c = Edge(b, c, 5)
        b_e = Edge(b, e, 8)
        nodes = [o, a, b, c, d, e]
        edges = [o_a, o_b, a_c, a_d, b_c, b_e]
        g = Graph(nodes, edges)
        delta = 0.1
        o.explored = True
        a.explored = True
        b.explored = True
        self.assertTrue(g.is_blocked(a_d, delta))
        self.assertTrue(g.is_blocked(b_e, delta))
        c.explored = True
        self.assertFalse(g.is_blocked(a_d, delta))
        self.assertTrue(g.is_blocked(b_e, delta))

    def test_get_unblocked_boundary_edge_satisfying_while_conditions_none(self):
        self.a.explored = True
        self.b.explored = True
        delta = 1.5
        y = self.b
        empty_set = set()
        unblocked_boundary_edge_satisfying_while_conditions = self.graph1.get_satisfying_edge(y, delta, empty_set)
        self.assertIsNone(unblocked_boundary_edge_satisfying_while_conditions)

    def test_get_unblocked_boundary_edge_satisfying_while_conditions(self):
        self.a.explored = True
        self.b.explored = True
        delta = 1.5
        y = self.a
        empty_set = set()
        unblocked_boundary_edge_satisfying_while_conditions = self.graph1.get_satisfying_edge(y, delta, empty_set)
        self.assertEqual(self.a_c, unblocked_boundary_edge_satisfying_while_conditions)

    def test_blocking(self):
        for i in range(10):
            o = Vertex("O")
            a = Vertex("A")
            b = Vertex("B")
            c = Vertex("C")
            d = Vertex("D")
            e = Vertex("E")
            o_a = Edge(o, a, 1)
            o_b = Edge(o, b, 2)
            a_c = Edge(a, c, 3)
            a_d = Edge(a, d, 4)
            b_c = Edge(b, c, 5)
            b_e = Edge(b, e, 8)
            nodes = [o, a, b, c, d, e]
            edges = [o_a, o_b, a_c, a_d, b_c, b_e]
            g = Graph(nodes, edges)
            delta = 0.1
            filename = "blocking.txt"
            f = open(filename, "w")
            f.write("")
            f.close()
            total_cost = g.blocking(o, delta, filename)
            f = open(filename, "a")
            f.write("\\\\ Total cost: " + str(total_cost) + "\n")
            f.close()
            g.tikz(filename, delta, None, True)
            self.assertEqual(63, total_cost)

    def test_tspm_to_tsp_conversion(self):
        o = Vertex("O")
        a = Vertex("A")
        b = Vertex("B")
        c = Vertex("C")
        d = Vertex("D")
        e = Vertex("E")
        o_a = Edge(o, a, 1)
        o_b = Edge(o, b, 2)
        a_c = Edge(a, c, 3)
        a_d = Edge(a, d, 4)
        b_c = Edge(b, c, 5)
        b_e = Edge(b, e, 8)
        nodes = [o, a, b, c, d, e]
        edges = [o_a, o_b, a_c, a_d, b_c, b_e]
        g = Graph(nodes, edges)

        tsp = TSP()
        tsp_graph = tsp.convert_tspm_to_tsp_graph(g)
        # print(tsp_graph.toString())

        tsp_graph = tsp.convert_tspm_to_tsp_graph(self.graph1)
        # print(tsp_graph.toString())

    def test_tspm(self):
        o = Vertex("O")
        a = Vertex("A")
        b = Vertex("B")
        c = Vertex("C")
        d = Vertex("D")
        e = Vertex("E")
        o_a = Edge(o, a, 1)
        o_b = Edge(o, b, 2)
        a_c = Edge(a, c, 3)
        a_d = Edge(a, d, 4)
        b_c = Edge(b, c, 5)
        b_e = Edge(b, e, 8)
        nodes = [o, a, b, c, d, e]
        edges = [o_a, o_b, a_c, a_d, b_c, b_e]
        g = Graph(nodes, edges)

        tsp = TSP()
        optimal_cost, min_path = tsp.TSPM(g)
        # print("min_path", min_path)
        self.assertEqual(35, optimal_cost)

    def test_tspm2(self):
        tsp = TSP()
        optimal_cost, min_path = tsp.TSPM(self.graph1)
        self.assertEqual(32, optimal_cost)

    def test_blockingShouldBe16Competitive(self):
        o = Vertex("O")
        a = Vertex("A")
        c = Vertex("C")
        b = Vertex("B")
        e = Vertex("E")
        d = Vertex("D")
        o_a = Edge(o, a, 1)
        o_b = Edge(o, b, 2)
        a_c = Edge(a, c, 3)
        a_d = Edge(a, d, 4)
        b_c = Edge(b, c, 5)
        b_e = Edge(b, e, 8)
        nodes = [o, a, b, c, d, e]
        edges = [o_a, o_b, a_c, a_d, b_c, b_e]
        graph = Graph(nodes, edges)

        delta = 0.1
        tsppp = TSP()
        optimal_tour, path = tsppp.TSPM(graph)
        online_tour = graph.blocking(o, delta, "big_test.txt")
        print(f"{online_tour=}")
        print(f"{optimal_tour=}")
        print("Competitive ratio:", online_tour / optimal_tour)
        self.assertLessEqual(online_tour, optimal_tour * 16)

        optimal_tour, path = tsppp.TSPM(self.graph1)
        online_tour = self.graph1.blocking(self.a, delta, "big_test.txt")
        print(f"{online_tour=}")
        print(f"{optimal_tour=}")
        print("Competitive ratio:", online_tour / optimal_tour)
        self.assertLessEqual(online_tour, optimal_tour * 16)

    def test_graph_with_two_edges_in_p_not_in_MST(self):
        a = Vertex("A")
        b = Vertex("B")
        c = Vertex("C")
        d = Vertex("D")
        f = Vertex("F")
        u = Vertex("U")
        v = Vertex("V")
        a_b = Edge(a, b, 2)
        a_d = Edge(a, d, 1)
        b_u = Edge(b, u, 5)
        b_c = Edge(b, c, 8)
        d_u = Edge(d, u, 3)
        d_f = Edge(d, f, 4)
        u_v = Edge(u, v, 5)
        f_v = Edge(f, v, 2)
        nodes = [a, b, c, d, f, u, v]
        edges = [a_b, a_d, b_c, b_u, d_u, d_f, u_v, f_v]
        graph = Graph(nodes, edges)

        delta = 0.1
        filename = "blocking.txt"
        f = open(filename, "w")
        f.write("")
        f.close()
        total_cost = graph.blocking(a, delta, filename)
        f = open(filename, "a")
        f.write("\\\\ Total cost: " + str(total_cost) + "\n")
        graph.tikz("blocking.txt", delta, final=True)
        f.close()

    def test_graph_generation(self):
        generator = GenarateGraphs()
        graphs = generator.generate(1, 10, 20)
        filename = "gen_graph.txt"
        f = open(filename, "w")
        f.write("")
        f.close()
        f = open(filename, "a")
        graphs[0].blocking(graphs[0].nodes[0], 2, filename)
        graphs[0].tikz("gen_graph.txt", 0.1, None, True)
        f.close()

    def test_lines_intersect(self):
        gen = GenarateGraphs()

        p1 = (4, 4)
        p2 = (8, 8)
        q1 = (4, 8)
        q2 = (8, 4)
        z1 = (2, 10)
        z2 = (10, 6)
        u1 = (6, 8)
        u2 = (10, 8)

        line1 = (p1, p2)
        line2 = (q1, q2)
        intersect1 = gen.lines_intersect(line1, line2)
        self.assertTrue(intersect1)

        line3 = (p1, q1)
        line4 = (p2, q2)
        intersect2 = gen.lines_intersect(line3, line4)
        self.assertFalse(intersect2)

        line5 = (z1, z2)
        line6 = (p2, q1)
        intersect3 = gen.lines_intersect(line3, line5)
        intersect4 = gen.lines_intersect(line6, line5)
        self.assertFalse(intersect3)
        self.assertTrue(intersect4)

        line7 = (u1, u2)
        intersect5 = gen.lines_intersect(line6, line7)
        self.assertTrue(intersect5)

        intersect6 = gen.lines_intersect(line1, line7)
        self.assertTrue(intersect6)

        intersect7 = gen.lines_intersect(line1, line3)
        self.assertFalse(intersect7)

        lines = [(p1, p2), (z1, z2), (u1, u2)]

        inter = gen.line_intersect_list_of_lines((q1, q2), lines)
        self.assertTrue(inter)

        lines = [(z1, z2), (u1, u2)]

        inter = gen.line_intersect_list_of_lines((q1, q2), lines)
        self.assertFalse(inter)

    def test_parrellel_lines_intersect(self):
        gen = GenarateGraphs()
        p1 = (8, 8)
        p2 = (4, 8)
        u1 = (6, 8)
        u2 = (10, 8)
        line1 = (p1, p2)
        line2 = (u1, u2)
        intersect = gen.parrellel_lines_intersect(line1, line2)
        self.assertTrue(intersect)

        q1 = (4, 4)
        q2 = (12, 12)
        z1 = (2, 2)
        line3 = (z1, p1)
        line4 = (q1, q2)
        intersect2 = gen.parrellel_lines_intersect(line3, line4)
        self.assertTrue(intersect2)

        line5 = (z1, q1)
        intersect3 = gen.parrellel_lines_intersect(line5, line4)
        self.assertFalse(intersect3)

    def test_mst(self):
        self.graph1.tikz("mst_graph.txt", delta=2)
        mst = self.graph1.mst_prim(self.d)
        expected = {self.a_c, self.b_c, self.c_f, self.f_d, self.d_e, self.g_a}
        self.assertEqual(expected, mst)
        self.graph2.tikz("mst_graph2.txt", delta=2)
        mst2 = self.graph2.mst_prim(self.a2)
        expected2 = {self.a_b2, self.b_c2, self.a_d2, self.f_v2, self.d_u2, self.d_f2}
        self.assertEqual(expected2, mst2)

    def test_python_solver(self):
        tsp = TSP()
        cost, path = tsp.python_tspm_solver(self.graph1)

    def test_tikz_mst_p(self):
        delta = 0.1
        filename = "blocking.txt"
        f = open(filename, "w")
        f.write("")
        f.close()
        total_cost = self.graph1.blocking(self.a, delta, filename)
        f = open(filename, "a")
        f.write("\\\\ Total cost: " + str(total_cost) + "\n")
        f.close()
        self.graph1.tikz(filename, delta, None, True)

    def test_get_P(self):
        file = "in_p.txt"
        f = open(file, "w")
        f.write("")
        f.close()
        self.graph2.blocking(self.a2, 0.1, "in_p.txt")
        self.graph2.tikz(file, 0.1, None, True)
        P = self.graph2.get_P()
        expected = {self.a_b2, self.b_c2, self.a_d2, self.d_u2, self.u_v2, self.f_v2}
        self.assertEqual(expected, P)

    def test_if_two_edges_in_P_not_in_mst(self):
        generator = GenarateGraphs()
        file_counter = 280
        delta = 2
        for graph in generator.generate(1, 10, 30):
            graph.blocking(graph.nodes[0], delta, None)
            if 1 < len(graph.get_P().difference(graph.mst_prim(graph.nodes[-1]))):
                file = "figs/two_edges_in_P_not_in_mst" + str(file_counter) + ".txt"
                file_img = "figs/two_edges_in_P_not_in_mst" + str(file_counter) + ".pdf"
                file_counter += 1
                graph.tikz(file, delta, None, True)
                graph.draw_spacial_graph(file_img)
            #self.assertGreaterEqual(1, len(graph.get_P().difference(graph.mst_prim(graph.nodes[-1]))))

    def test_counter_example(self):
        n0 = Vertex("0")
        n1 = Vertex("1")
        n2 = Vertex("2")
        n3 = Vertex("3")
        n4 = Vertex("4")
        n5 = Vertex("5")
        n6 = Vertex("6")
        n7 = Vertex("7")
        n8 = Vertex("8")
        n9 = Vertex("9")
        n10 = Vertex("10")
        n11 = Vertex("11")
        n12 = Vertex("12")
        n13 = Vertex("13")
        n14 = Vertex("14")
        n15 = Vertex("15")
        n16 = Vertex("16")
        n17 = Vertex("17")
        n18 = Vertex("18")
        n19 = Vertex("19")

        e1 = Edge(n1, n0, 8622)
        e2 = Edge(n2, n0, 5343)
        e3 = Edge(n3, n1, 3083)
        e4 = Edge(n4, n1, 3371)
        e5 = Edge(n5, n2, 6690)
        e6 = Edge(n6, n1, 555)
        e7 = Edge(n7, n2, 3372)
        e8 = Edge(n8, n4, 6207)
        e9 = Edge(n9, n4, 6368)
        e10 = Edge(n10, n2, 3108)
        e11 = Edge(n11, n8, 2524)
        e12 = Edge(n12, n4, 4495)
        e13 = Edge(n13, n0, 3942)
        e14 = Edge(n14, n12, 2133)
        e15 = Edge(n15, n4, 5523)
        #e16 = Edge(n16, n6, 5335)
        e17 = Edge(n17, n0, 1615)
        e18 = Edge(n18, n2, 7532)
        e19 = Edge(n19, n2, 1895)
        e20 = Edge(n1, n15, 3622)
        e21 = Edge(n1, n16, 9062)
        e22 = Edge(n2, n8, 6591)
        e23 = Edge(n2, n17, 5343)
        e24 = Edge(n4, n14, 1285)
        e25 = Edge(n5, n17, 468)
        e26 = Edge(n6, n16, 9818)
        e27 = Edge(n7, n13, 8347)
        e28 = Edge(n9, n15, 9906)
        e29 = Edge(n12, n5, 8784)
        e30 = Edge(n13, n1, 8396)
        e31 = Edge(n13, n16, 7036)

        nodes = [n0, n1, n2, n3, n4, n5, n6, n7, n8, n9, n10, n11, n12, n13, n14, n15, n16, n17, n18, n19]
        edges = [e1, e2, e3, e4, e5, e6, e7, e8, e9, e10, e11, e12, e13, e14, e15, e17, e18, e19, e20, e21, e22, e23, e24, e25, e26, e27, e28, e29, e30, e31]
        graph = Graph(nodes, edges)

        file = "cc.txt"
        f = open(file, "w")
        f.close()

        graph.blocking(n0, 2, file)
        f = open(file, "w")
        f.close()
        graph.tikz(file, 2, None, True)
        #graph.mst_scipy()
        #p = graph.get_P()


    def test_scipy_mst(self):
        self.graph2.mst_scipy()

    def test_plot_graph(self):
        gen = GenarateGraphs()
        graphs = gen.generate(2, 20, 10)
        graphs[0].blocking(graphs[0].nodes[0], 0.01, None)

        file = "figs/two.txt"
        file_img = "figs/two.pdf"
        graphs[0].draw_spacial_graph(file_img)
        graphs[0].tikz(file, 2, None, True)


    def test_claim1(self):
        n0 = Vertex("0")
        n1 = Vertex("1")
        n2 = Vertex("2")
        n3 = Vertex("3")
        n4 = Vertex("4")
        n5 = Vertex("5")

        e0 = Edge(n1, n0, 4208)
        e1 = Edge(n2, n0, 3770)
        e2 = Edge(n3, n0, 3763)
        e3 = Edge(n4, n2, 7834)
        e4 = Edge(n5, n0, 6391)
        e5 = Edge(n4, n0, 8093)
        e6 = Edge(n4, n1, 4447)
        e7 = Edge(n5, n2, 2723)
        e8 = Edge(n5, n3, 3988)

        nodes = [n0, n1, n2, n3, n4, n5]
        edges = [e0, e1, e2, e2, e3, e4, e5, e6, e7, e8]

        g = Graph(nodes, edges)
        g.blocking(n0, 0.0001, "claim1.txt")
        g.tikz("claim1.txt", 0.0001, None, True)

    def test_claim1(self):
        n0 = Vertex("0")
        n1 = Vertex("1")
        n2 = Vertex("2")
        n3 = Vertex("3")
        n4 = Vertex("4")
        n5 = Vertex("5")

        e0 = Edge(n1, n0, 3)
        e1 = Edge(n2, n0, 8)
        e2 = Edge(n3, n2, 1)
        e3 = Edge(n4, n2, 9)
        e4 = Edge(n5, n1, 5)
        e5 = Edge(n1, n3, 7)
        e6 = Edge(n4, n0, 4)
        e7 = Edge(n4, n5, 2)

        nodes = [n0, n1, n2, n3, n4, n5]
        edges = [e0, e1, e2, e3, e4, e5, e6, e7]
        delta = 0.15
        filename = "claim1.txt"
        f = open(filename, "w");f.write("");f.close()
        g = Graph(nodes, edges)
        g.blocking(n0, delta, filename)
        g.tikz(filename, delta, None, True)

    def test_comp_ratio(self):
        n0 = Vertex("0")
        n1 = Vertex("1")
        n2 = Vertex("2")
        n3 = Vertex("3")
        n4 = Vertex("4")
        n5 = Vertex("5")
        n6 = Vertex("6")
        n7 = Vertex("7")
        n8 = Vertex("8")
        n9 = Vertex("9")

        e0_1 = Edge(n0, n1 , 1)
        e1_3 = Edge(n1, n3, 2.01)
        e3_5 = Edge(n3, n5, 2.35)
        e5_7 = Edge(n5, n7, 5.5)
        e7_9 = Edge(n7, n9, 9.78)
        e0_2 = Edge(n0, n2, 2)
        e2_4 = Edge(n2, n4, 2.02)
        e4_6 = Edge(n4, n6, 4.13)
        e6_8 = Edge(n6, n8, 7.34)
        e8_9 = Edge(n8, n9, 13.04)
        nodes = [n0, n1, n2, n3, n4, n5, n6, n7, n8, n9]
        edges = [e0_1, e1_3, e3_5, e5_7, e7_9, e0_2, e2_4, e4_6, e6_8, e8_9]
        graph = Graph(nodes, edges)

        delta = 2
        filename = "test_comp_16.txt"
        f = open(filename, "w");f.write("");f.close()
        online_cost = graph.blocking(n0, delta, filename)
        tsp = TSP();
        offline_cost, perm = tsp.python_tspm_solver(graph)
        f = open(filename, "a")
        f.write("\\\\ Total online cost: " + str(online_cost) + "\n")
        f.write("\\\\ Total offline cost: " + str(offline_cost) + "\n")
        f.write("\\\\ Competitive ratio: " + str(online_cost / offline_cost) + "\n")
        graph.tikz(filename, delta, final=True)
        f.close()

    def test_evil_graph(self):
        for size in range(3,4):
            nodes = []
            edges = []
            edges_weights = [Fraction(1), Fraction(2), Fraction(201,100), Fraction(202,100)]
            for i in range(size*10):
                sum_w = 0
                for w in edges_weights:
                    sum_w = sum_w + w
                weight = sum_w / Fraction(3) + Fraction(1)
                ratio = sum_w / weight
                # print("ratio", float(ratio))
                edges_weights.append(weight)

            # find the magic ratio
            n = len(edges_weights)
            acc = 0
            for i, e in enumerate(edges_weights):
                if (i+1) % 2 == 0:
                    num_of_traversals = n - (i+1)
                else:
                    num_of_traversals = n - i
                # print("i:", i)
                # print("num_of_traversals:", num_of_traversals)
                # print("num_of_traversals * e:", float(num_of_traversals * e))
                acc += num_of_traversals * e

            l_ctr = -1
            r_ctr = 1
            for i, weight in enumerate(edges_weights):
                nodes.append(Vertex(str(i)))


            for weight in edges_weights:
                l_label = str(max(l_ctr, 0))
                r_label = str(min(r_ctr,len(edges_weights)-1))
                # print("r_label", r_label)
                # print("l_label", l_label)
                node1 = None
                node2 = None
                for node in nodes:

                    if node.label == l_label:
                        # print("node.lable", node.label)
                        node1 = node
                    elif node.label == r_label:
                        # print("node.lable", node.label)
                        node2 = node
                edges.append(Edge(node1, node2, weight))
                l_ctr += 1
                r_ctr += 1

            graph = Graph(nodes, edges)

            delta = 2

            filename = "test_comp_16.txt"
            f = open(filename, "w");f.write("");f.close()
            #online_cost = graph.blocking(nodes[0], delta, filename)
            online_cost = graph.blocking(nodes[0], delta, None)
            offline_cost = sum(edges_weights)
            f = open(filename, "a")
            graph.tikz(filename, delta, final=True)
            f.write("\\\\ Total online cost: " + str(online_cost) + "\n")
            f.write("\\\\ Total offline cost: " + str(offline_cost) + "\n")
            f.write("\\\\ Competitive ratio: " + str(online_cost / offline_cost) + "\n")
            #print("Offline cost: ", float(offline_cost))
            #print("Blocking: ", float(online_cost))
            #print("Calculated: ", float(acc))
            print("Size " + str(size * 10) + " Competitive ratio: " + str(float(online_cost / offline_cost)))
            #print("Acc ratio: " + str(float(acc / offline_cost)) + "\n")
            f.close()

    def test_experiments_delta(self):
        f = open("exp_delta.txt", "w")
        f.write("")
        f.close()
        f = open("exp_delta.txt", "a")
        delta = [0.0001, 0.1, 1, 2, 5, 10]
        generator = GenarateGraphs()
        graphs = generator.generate(10, 10, 30)
        for graph in graphs:
            tsp = TSP()
            tsp_cost, perm = tsp.python_tspm_solver(graph)
            for d in delta:
                graph_copy = copy.deepcopy(graph)
                blocking_cost = graph_copy.blocking(graph_copy.nodes[0], d, None)
                comp_ratio = blocking_cost / tsp_cost
                f.write("delta="+str(d)+" tsp="+str(tsp_cost)+" blocking="+str(blocking_cost)+" comp_ratio="+str(comp_ratio)+" size="+str(len(graph.nodes))+" edges="+str(len(graph.edges))+" weigths=uniform connectivity=30\n")
        f.close()

    def test_experiments_random_delta(self):
        f = open("exp_random_delta.txt", "w")
        f.write("")
        f.close()
        f = open("exp_random_delta.txt", "a")
        size = 15
        generator = GenarateGraphs()
        graphs = generator.generate(140, size, 30)
        for graph in graphs:
            tsp = TSP()
            tsp_cost, perm = tsp.python_tspm_solver(graph)
            for d in range(1):
                delta = random.uniform(0.00000001, 50)
                graph_copy = copy.deepcopy(graph)
                blocking_cost = graph_copy.blocking(graph_copy.nodes[0], delta, None)
                comp_ratio = blocking_cost / tsp_cost
                f.write("delta="+str(delta)+" tsp="+str(tsp_cost)+" blocking="+str(blocking_cost)+" comp_ratio="+str(comp_ratio)+" size="+str(len(graph.nodes))+" edges="+str(len(graph.edges))+" weigths=uniform connectivity=30\n")
        f.close()

    def test_experiments_connectivity(self):
        f = open("exp_connectivity.txt", "w")
        f.write("")
        f.close()
        f = open("exp_connectivity.txt", "a")
        connectivity = [0, 20, 30, 50, 75, 100]
        generator = GenarateGraphs()
        delta = 2
        for c in connectivity:
            graphs = generator.generate(0, 15, c)
            for graph in graphs:
                tsp = TSP()
                tsp_cost, perm = tsp.python_tspm_solver(graph)
                blocking_cost = graph.blocking(graph.nodes[0], delta, None)
                comp_ratio = blocking_cost / tsp_cost
                f.write("delta="+str(delta)+" tsp="+str(tsp_cost)+" blocking="+str(blocking_cost)+" comp_ratio="+str(comp_ratio)+" size="+str(len(graph.nodes))+" edges="+str(len(graph.edges))+" weigths=uniform connectivity="+str(c)+"\n")
        f.close()

    def test_experiments_size(self):
        f = open("exp_size.txt", "w")
        f.write("")
        f.close()
        f = open("exp_size.txt", "a")
        size = [4, 8, 12, 16, 18, 19]
        generator = GenarateGraphs()
        delta = 2
        for s in size:
            graphs = generator.generate(0, s, 30)
            for graph in graphs:
                tsp = TSP()
                tsp_cost, perm = tsp.python_tspm_solver(graph)
                blocking_cost = graph.blocking(graph.nodes[0], delta, None)
                comp_ratio = blocking_cost / tsp_cost
                f.write("delta="+str(delta)+" tsp="+str(tsp_cost)+" blocking="+str(blocking_cost)+" comp_ratio="+str(comp_ratio)+" size="+str(len(graph.nodes))+" edges="+str(len(graph.edges))+" weigths=uniform connectivity=30\n")
        f.close()

    def test_experiments_weights_linear(self):
        f = open("exp_weights_linear.txt", "w");f.write("");f.close();f = open("exp_weights_linear.txt", "a")
        weights = set()
        delta = 2
        size = 10
        num_of_edges_upperbound = math.ceil((size * (size + 1)) / 2)
        for w in range(num_of_edges_upperbound):
            weights.add(w)
        generator = GenarateGraphs()
        graphs = generator.generate(10, size, 30, weights)
        for graph in graphs:
            tsp = TSP()
            tsp_cost, perm = tsp.python_tspm_solver(graph)
            blocking_cost = graph.blocking(graph.nodes[0], delta, None)
            comp_ratio = blocking_cost / tsp_cost
            f.write("delta="+str(delta)+" tsp="+str(tsp_cost)+" blocking="+str(blocking_cost)+" comp_ratio="+str(comp_ratio)+" size="+str(len(graph.nodes))+" edges="+str(len(graph.edges))+" weigths=linear connectivity=30\n")
        f.close()

    def test_experiments_weights_poly(self):
        f = open("exp_weights_poly.txt", "w");f.write("");f.close();f = open("exp_weights_poly.txt", "a")
        weights = set()
        delta = 2
        size = 10
        num_of_edges_upperbound = math.ceil((size * (size + 1)) / 2)
        for w in range(num_of_edges_upperbound):
            weights.add(w^2)
        generator = GenarateGraphs()
        graphs = generator.generate(10, size, 30, weights)
        for graph in graphs:
            tsp = TSP()
            tsp_cost, perm = tsp.python_tspm_solver(graph)
            blocking_cost = graph.blocking(graph.nodes[0], delta, None)
            comp_ratio = blocking_cost / tsp_cost
            f.write("delta="+str(delta)+" tsp="+str(tsp_cost)+" blocking="+str(blocking_cost)+" comp_ratio="+str(comp_ratio)+" size="+str(len(graph.nodes))+" edges="+str(len(graph.edges))+" weigths=poly connectivity=30\n")
        f.close()

    def test_experiments_weights_exp(self):
        f = open("exp_weights_exp.txt", "w");f.write("");f.close();f = open("exp_weights_exp.txt", "a")
        weights = set()
        #weights =[]
        delta = 2
        size = 15
        num_of_edges_upperbound = math.ceil((size * (size + 1)) / 2)
        for w in range(num_of_edges_upperbound):
            weights.add(math.pow(1.2, float(w)))
            #weights.append(math.pow(1.2, float(w)))
        generator = GenarateGraphs()
        graphs = generator.generate(15, size, 30, weights)
        for graph in graphs:
            tsp = TSP()
            tsp_cost, perm = tsp.python_tspm_solver(graph)
            blocking_cost = graph.blocking(graph.nodes[0], delta, None)
            comp_ratio = blocking_cost / tsp_cost
            f.write("delta="+str(delta)+" tsp="+str(tsp_cost)+" blocking="+str(blocking_cost)+" comp_ratio="+str(comp_ratio)+" size="+str(len(graph.nodes))+" edges="+str(len(graph.edges))+" weigths=exp connectivity=30\n")
        f.close()

    def test_calc(self):
        f = open("exp_size_random.txt", "r")
        sum = 0
        total = 0
        for line in f.readlines():
            attributes = line.split(" ")
            size = 0
            comp = 0
            for at in attributes:
                if "size" in at:
                    size_arr = at.split("=")
                    size = int(size_arr[1])
                if "comp_ratio" in at:
                    comp_arr = at.split("=")
                    comp = float(comp_arr[1])
            if size < 21 and size > 12:
                sum += comp
                total += 1
        print(str(sum/total))


    def test_format_me_pls(self):
        f = open("format_these_motherfuckers.txt", "r")
        f2 = open("format_these_motherfuckers_new.txt", "a")
        for line in f.readlines():
            arr = line.split("$")
            print(arr[1])
            if "/" in arr[1]:
                values = arr[1].split("/")
                arr[1] = str(round(int(values[0]) / int(values[1]), 1))
            s = ''.join(map(str, arr))
            f2.write(s)

if __name__ == '__main__':
    unittest.main()
