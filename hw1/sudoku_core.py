from pysat.formula import CNF
from pysat.solvers import MinisatGH
from ortools.sat.python import cp_model
import clingo
import gurobipy as gp
from gurobipy import GRB
import itertools
import numpy as np
import math

###
# Propagation function to be used in the recursive sudoku solver
###


def propagate(sudoku_possible_values, k):
    """
    Propagate function which given possible values list for sudoku, filters
    impossible values from list based on sudoku rules. Edges here correspond
    to undirected edge between cells in same row, column or k*k block.
    """

    num_vertices = (k*k)*(k*k)  # 81 for k = 3= 3
    # Create 2d np array of vertice numbers of 0 until num_vertices-1 for edges
    vertices = np.arange(num_vertices).reshape(k*k, k*k)
    edges = get_edges(vertices, k)  # 810 for k

    while len(edges) > 1:

        new_edges = []
        for (v1, v2) in edges:
            # Obtain row and column values for vertex number v2
            row2 = (v2) // (k*k)
            col2 = (v2) % (k*k)
            # Possible values for v2 vertex
            p2 = sudoku_possible_values[row2][col2]

            # Also for edges the other way around
            row1 = (v1) // (k*k)
            col1 = (v1) % (k*k)
            p1 = sudoku_possible_values[row1][col1]

            # Remove possibilities if possibilties of v1 vertex (=p1) only has
            # one possible value and value is in the possibilities of its edge
            # v2 (=p2)
            if len(p1) == 1 and p1[0] in p2:
                sudoku_possible_values[row2][col2].remove(p1[0])

                if len(sudoku_possible_values[row2][col2]) == 1:
                    new_edges.append((v1, v2))

            if len(p2) == 1 and p2[0] in p1:
                sudoku_possible_values[row1][col1].remove(p2[0])

                # Store if other vertex now becomes case with one possibility
                if len(sudoku_possible_values[row1][col1]) == 1:
                    new_edges.append((v1, v2))

        edges = new_edges

    return sudoku_possible_values

###
# Solver that uses SAT encoding
###


def solve_sudoku_SAT(sudoku, k):
    """
    Function that solves given sudoku list with SAT encoding. Edges correspond
    to undirected edge between cells in same row, column or k*k block.
    """

    formula = CNF()

    num_vertices = (k*k)*(k*k)  # 81 for k = 3
    # Nd array representation of sudoku
    graph = np.array(sudoku).reshape(num_vertices, 1)
    # Create 2d np array of vertice numbers of 0 until num_vertices-1 for edges
    vertices = np.arange(num_vertices).reshape(k*k, k*k)
    edges = get_edges(vertices, k)
    vertices = vertices.reshape(num_vertices, 1)
    # Number of possible values in sudoku which we refer to as numbers (nums/n)
    num_nums = k*k

    # Return propositional variable for each vertex and num combination
    def var_v_n(v, n):
        return (v*num_nums)+n

    # Each cell contains a value between 1 and k*k
    for v in range(num_vertices):

        # Clause to ensure at least one num can be assigned to one vertex
        clause = [var_v_n(v, n) for n in range(1, num_nums+1)]
        formula.append(clause)

        for n1 in range(1, num_nums+1):
            for n2 in range(n1+1, num_nums+1):
                # Clause ensures at most one num can be assigned to one vertex
                clause = [-1*var_v_n(v, n1), -1*var_v_n(v, n2)]
                formula.append(clause)

        # If a cell contains u in input, cell in solution must contain u
        if graph[v] != 0:
            cell = var_v_n(vertices[v].item(), graph[v].item())
            formula.append([cell])

    # Each two different cells in same row,col,block must contain diff values
    for (v1, v2) in edges:
        for n in range(1, num_nums+1):
            # Num assigned to vertex 1 of edge should be true, OR num assigned
            # to other vertex of edge should be true
            clause = [-var_v_n(v1.item(), n), -var_v_n(v2.item(), n)]
            formula.append(clause)

    solver = MinisatGH()
    solver.append_formula(formula)

    answer = solver.solve()
    if answer:
        model = solver.get_model()
        # Fill in sudoku graph with answer
        for v in range(num_vertices):
            for n in range(1, num_nums+1):
                if var_v_n(v, n) in model:
                    graph[v] = n
        graph = graph.reshape(k*k, k*k).tolist()
    else:
        graph = None

    return graph

###
# Solver that uses CSP encoding
###


def solve_sudoku_CSP(sudoku, k):
    """
    Function that solves given sudoku list with CSP encoding. Edges correspond
    to undirected edge between cells in same row, column or k*k block.
    """

    model = cp_model.CpModel()

    num_vertices = (k*k)*(k*k)  # 81 for k = 3
    # Nd array representation of sudoku
    graph = np.array(sudoku).reshape(num_vertices, 1)
    # Create 2d np array of vertice numbers of 0 until num_vertices-1 for edges
    vertices = np.arange(num_vertices).reshape(k*k, k*k)
    edges = get_edges(vertices, k)
    vertices = vertices.reshape(num_vertices, 1)
    # Number of possible values in sudoku which we refer to as numbers (nums/n)
    num_nums = k*k

    vars = dict()
    for v in range(num_vertices):
        # Add variables n_i for each vertex with domains 1 until k*k
        # correspond to possible nums for each vertex
        vars[v] = model.NewIntVar(1, num_nums, "n{}".format(v))

        # If a cell contains u in input, cell in solution must contain u
        if graph[v] != 0:
            model.Add(vars[v] == graph[v].item())

    for (v1, v2) in edges:
        # Each 2 different cells in same row,col,block must contain diff values
        model.Add(vars[v1] != vars[v2])

    solver = cp_model.CpSolver()
    answer = solver.Solve(model)

    if answer == cp_model.FEASIBLE:
        # Fill in sudoku graph with answer
        for v in range(num_vertices):
            n = solver.Value(vars[v])
            graph[v] = n
        graph = graph.reshape(k*k, k*k).tolist()
    else:
        graph = None

    return graph

###
# Solver that uses ASP encoding
###


def solve_sudoku_ASP(sudoku, k):
    """
    Function that solves given sudoku list with ASP encoding. Edges correspond
    to undirected edge between cells in same row, column or k*k block.
    """

    num_vertices = (k*k)*(k*k)  # 81 for k = 3
    # Nd array representation of sudoku
    graph = np.array(sudoku).reshape(num_vertices, 1)
    # Create 2d np array of vertice numbers of 0 until num_vertices-1 for edges
    vertices = np.arange(num_vertices).reshape(k*k, k*k)
    edges = get_edges(vertices, k)
    vertices = vertices.reshape(num_vertices, 1)
    # Number of possible values in sudoku which we refer to as numbers (nums/n)
    num_nums = k*k

    asp_code = ""
    # Encode our facts as predicates vertex/1, value/2 which is possible values
    # for each vertex and edges/2
    for v in range(num_vertices):
        asp_code += """vertex(v{}).\n""".format(v)

        # If a cell contains u in input, cell in solution must contain u
        if graph[v].item() != 0:
            asp_code += """value(v{}, {}).\n""".format(v, graph[v].item())
        else:
            for n1 in range(1, num_nums+1):
                asp_code += """value(v{},{}).\n""".format(v, n1)

    for (v1, v2) in edges:
        asp_code += """edge(v{},v{}).\n""".format(v1, v2)

    # Universily quantified variables; number for each vertex, already filled
    # in value must also hold
    for n1 in range(1, num_nums+1):
        asp_code += """num(V,{}) :- vertex(V), value(V, {})""".format(n1, n1)
        for n2 in range(1, num_nums+1):
            if n1 != n2:
                asp_code += """, not num(V,{})""".format(n2)

        asp_code += """.\n"""

    # Each two different cells in same row,col,block must contain diff values
    asp_code += """:- edge(V1, V2), num(V1,N), num(V2,N).\n"""

    # Only interested in the number predicate
    asp_code += """#show num/2.\n"""

    control = clingo.Control()
    control.add("base", [], asp_code)
    control.ground([("base", [])])

    def on_model(model):
        global output
        output = model.symbols(shown=True)

    control.configuration.solve.models = 1
    answer = control.solve(on_model=on_model)

    if answer.satisfiable:
        # Fill in sudoku graph with answer
        for i in range(len(output)):
            v = int(str(output[i]).split(",")[0].split("v")[1])
            n = int(str(output[i]).split(",")[-1][:-1])
            graph[v] = n
        graph = graph.reshape(k*k, k*k).tolist()
    else:
        graph = None
    return graph

###
# Solver that uses ILP encoding
###


def solve_sudoku_ILP(sudoku, k):
    """
    Function that solves given sudoku list with ILP encoding. Edges correspond
    to undirected edge between cells in same row, column or k*k block.
    """

    model = gp.Model()

    num_vertices = (k*k)*(k*k)  # 81 for k = 3
    # Nd array representation of sudoku
    graph = np.array(sudoku).reshape(num_vertices, 1)
    # Create 2d np array of vertice numbers of 0 until num_vertices-1 for edges
    vertices = np.arange(num_vertices).reshape(k*k, k*k)
    edges = get_edges(vertices, k)
    vertices = vertices.reshape(num_vertices, 1)
    # Number of possible values in sudoku which we refer to as numbers (nums/n)
    num_nums = k*k

    vars = dict()
    for v in range(num_vertices):
        for n in range(1, num_nums+1):
            # Add Binary variables for each vertex, number combination that
            # indicate num assigned or not
            vars[(v, n)] = model.addVar(vtype=GRB.BINARY,
                                        name="x({},{})".format(v, n))

        # Constraint to ensure exactly one num assigned to one vertex
        # Sum should be one
        model.addConstr(gp.quicksum([vars[(v, n)] for n in range(
            1, num_nums+1)]) == 1)

        # If a cell contains u in input, cell in solution must contain u
        if graph[v] != 0:
            model.addConstr(vars[(v, graph[v].item())] == 1)

    # Each two different cells in same row,col,block must contain diff values
    for (v1, v2) in edges:
        for n in range(1, num_nums+1):
            # Num can not be assigned to both nodes as sum not larger than one
            model.addConstr(vars[(v1, n)] + vars[(v2, n)] <= 1)

    model.optimize()
    if model.status == GRB.OPTIMAL:
        # Fill in sudoku graph with answer
        for v in range(num_vertices):
            for n in range(1, num_nums+1):
                if vars[(v, n)].x == 1:
                    graph[v] = n
        graph = graph.reshape(k*k, k*k).tolist()
    else:
        graph = None

    return graph


def get_edges(vertices, k):
    edges = []
    # Obtain edges between vertices in rows and in columns by permuting
    for row_col in range(k*k):

        col_vertices = vertices[:, row_col]
        row_vertices = vertices[row_col, :]
        col_edges = list(itertools.permutations(row_vertices, 2))
        edges.extend(col_edges)
        row_edges = list(itertools.permutations(col_vertices, 2))
        edges.extend(row_edges)

    # Obtain edges for each block of k*k by taking steps of k*k
    for i in range(0, k*k, k):
        for j in range(0, k*k, k):
            # Permute from array of vertices in current block
            block = np.ndarray.flatten(vertices[i:i+k, j:j+k])
            edges.extend(list(itertools.permutations(block, 2)))

    # Convert to set to get undirected graph; we do not need edges both ways
    edges = list(set([tuple(sorted(e)) for e in edges]))
    return edges
