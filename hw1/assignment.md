# Homework assignment 1

In this assignment, you will program several algorithms to solve a search problem—namely, to solve Sudoku's on a `(k*k)^2` size grid, for varying sizes of `k`. You will use several problem solving and search methods.

The homework assignment consists of several parts (*a–e*).

You will finish a Python program ([sudoku.py](sudoku.py)) by
implementing the core algorithms (in [sudoku_core.py](sudoku_core.py)) for the program.

## Instructions

1. Read the description of assignments 1.a–1.e (below).
1. Provide your implementation of the functions `propagate()`, `solve_sudoku_SAT()`, `solve_sudoku_CSP()`, `solve_sudoku_ASP()` and `solve_sudoku_ILP()` in (your version of) the file `sudoku_core.py`.
(You can test your implementation using [sudoku.py](sudoku.py), that refers to [sudoku_core.py](sudoku_core.py), and using the files in [inputs](./inputs).)
1. Explain clearly what you did, and how your encodings work, in comments in `sudoku_core.py`.
1. Submit your version of the file `sudoku_core.py` (with your implementation and comments) via [Canvas](https://canvas.uva.nl/courses/10768).

---

## Sudoku's

Here follows a description of *inputs* and *solutions* for Sudoku's.
These notions are the same as for the [Sudoku's](https://en.wikipedia.org/wiki/Sudoku) that you might have seen elsewhere, generalized to arbitrary values of `k` (typically, `k=3`).

An *input* consists of a `k`-by-`k` grid of blocks,
where each consisting of a `k`-by-`k` grid of cells,
forming a `k*k`-by-`k*k` grid of cells.
Each cell can (but must not) contain a value between `1` and `k*k`.

An example is the following grid, for `k=3`, written out
as `k*k` lines, each consisting of `k*k` numbers, separated by spaces.
Here the value `0` represents that a cell does not contain
a value between `1` and `k*k`.

```
0 0 0 2 6 0 7 0 1
6 8 0 0 7 0 0 9 0
1 9 0 0 0 4 5 0 0
8 2 0 1 0 0 0 4 0
0 0 4 6 0 2 9 0 0
0 5 0 0 0 3 0 2 8
0 0 9 3 0 0 0 7 4
0 4 0 0 5 0 0 3 6
7 0 3 0 1 8 0 0 0
```

Here is the same input, where now the `k*k` blocks are separated from
each other with whitespace.

```
0 0 0   2 6 0   7 0 1
6 8 0   0 7 0   0 9 0
1 9 0   0 0 4   5 0 0

8 2 0   1 0 0   0 4 0
0 0 4   6 0 2   9 0 0
0 5 0   0 0 3   0 2 8

0 0 9   3 0 0   0 7 4
0 4 0   0 5 0   0 3 6
7 0 3   0 1 8   0 0 0
```

A *solution* for a sudoku input is a `k*k`-by-`k*k` grid (for the same `k`),
where:
1. each cell contains a value between `1` and `k*k`,
1. if a cell *(i,j)* contains a value `u` in the input,
  then the cell *(i,j)* in the solution must contain the same value `u`,
1. each two different cells in the same row must contain different values,
1. each two different cells in the same column must contain different values, and
1. each two different cells in the same `k*k` block must contain different values.

For example, a solution for the input mentioned above is the following:

```
4 3 5 2 6 9 7 8 1
6 8 2 5 7 1 4 9 3
1 9 7 8 3 4 5 6 2
8 2 6 1 9 5 3 4 7
3 7 4 6 8 2 9 1 5
9 5 1 7 4 3 6 2 8
5 1 9 3 2 6 8 7 4
2 4 8 9 5 7 1 3 6
7 6 3 4 1 8 2 5 9
```

## Working of `sudoku.py`

The program `sudoku.py` provides functionality to find a solution for
a sudoku input (and you will add to this functionality in this assignment).

Here follows a description of the functionality that is already present in `sudoku.py`,
and of the functionality that you will add.

### Representing sudoku inputs and solutions

Sudoku inputs are represented as a list of lists.
For example, the sudoku input above is represented as:

```
input =
  [[0, 0, 0, 2, 6, 0, 7, 0, 1],
   [6, 8, 0, 0, 7, 0, 0, 9, 0],
   [1, 9, 0, 0, 0, 4, 5, 0, 0],
   [8, 2, 0, 1, 0, 0, 0, 4, 0],
   [0, 0, 4, 6, 0, 2, 9, 0, 0],
   [0, 5, 0, 0, 0, 3, 0, 2, 8],
   [0, 0, 9, 3, 0, 0, 0, 7, 4],
   [0, 4, 0, 0, 5, 0, 0, 3, 6],
   [7, 0, 3, 0, 1, 8, 0, 0, 0]];
```

Solutions are represented similarly, for example:

```
solution =
  [[4, 3, 5, 2, 6, 9, 7, 8, 1],
   [6, 8, 2, 5, 7, 1, 4, 9, 3],
   [1, 9, 7, 8, 3, 4, 5, 6, 2],
   [8, 2, 6, 1, 9, 5, 3, 4, 7],
   [3, 7, 4, 6, 8, 2, 9, 1, 5],
   [9, 5, 1, 7, 4, 3, 6, 2, 8],
   [5, 1, 9, 3, 2, 6, 8, 7, 4],
   [2, 4, 8, 9, 5, 7, 1, 3, 6],
   [7, 6, 3, 4, 1, 8, 2, 5, 9]];
```

### Solving algorithms

The program `sudoku.py` provides structure for five different algorithms
to find a solution for a given input:
- A recursive search algorithm, that employs some *propagation* at each of the nodes in the search tree.
  - Abbreviation: `prop`
- A search algorithm that encodes the task of finding a solution for the sudoku input as a *propositional CNF formula*, and that uses a SAT solver to find the solution.
  - Abbreviation: `sat`
- A search algorithm that encodes the task of finding a solution for the sudoku input as a *constraint satisfaction problem*, and that uses a CSP solver to find the solution.
  - Abbreviation: `csp`
- A search algorithm that encodes the task of finding a solution for the sudoku input as an *answer set program*, and that uses an ASP solver to find the solution.
  - Abbreviation: `asp`
- A search algorithm that encodes the task of finding a solution for the sudoku input as a *integer logic program*, and that uses an ILP solver to find the solution.
  - Abbreviation: `ilp`

### Calling `sudoku.py`

The program `sudoku.py` can be used by calling:
```
% python sudoku.py -i INPUT -s SOLVER
```
where `INPUT` is replaced by a file containing a sudoku input,
and where `SOLVER` is replaced by one of `prop`/`sat`/`csp`/`asp`/`ilp`.

You can add the optional flag `-v` to show more information during the process of solving (enable verbose mode).

For example:
```
% python sudoku.py -i inputs/easy3.sudoku -s prop

+-------+-------+-------+
| 4 3 5 | 2 6 9 | 7 8 1 |
| 6 8 2 | 5 7 1 | 4 9 3 |
| 1 9 7 | 8 3 4 | 5 6 2 |
+-------+-------+-------+
| 8 2 6 | 1 9 5 | 3 4 7 |
| 3 7 4 | 6 8 2 | 9 1 5 |
| 9 5 1 | 7 4 3 | 6 2 8 |
+-------+-------+-------+
| 5 1 9 | 3 2 6 | 8 7 4 |
| 2 4 8 | 9 5 7 | 1 3 6 |
| 7 6 3 | 4 1 8 | 2 5 9 |
+-------+-------+-------+
```
Or:
```
% python sudoku.py -i inputs/easy3.sudoku -s prop -v

Reading sudoku from inputs/easy3.sudoku..
Input sudoku:
+-------+-------+-------+
|       | 2 6   | 7   1 |
| 6 8   |   7   |   9   |
| 1 9   |     4 | 5     |
+-------+-------+-------+
| 8 2   | 1     |   4   |
|     4 | 6   2 | 9     |
|   5   |     3 |   2 8 |
+-------+-------+-------+
|     9 | 3     |   7 4 |
|   4   |   5   |   3 6 |
| 7   3 |   1 8 |       |
+-------+-------+-------+
Solving sudoku using recursion and propagation..
Did recursive solving with propagation in 0.02 seconds
+-------+-------+-------+
| 4 3 5 | 2 6 9 | 7 8 1 |
| 6 8 2 | 5 7 1 | 4 9 3 |
| 1 9 7 | 8 3 4 | 5 6 2 |
+-------+-------+-------+
| 8 2 6 | 1 9 5 | 3 4 7 |
| 3 7 4 | 6 8 2 | 9 1 5 |
| 9 5 1 | 7 4 3 | 6 2 8 |
+-------+-------+-------+
| 5 1 9 | 3 2 6 | 8 7 4 |
| 2 4 8 | 9 5 7 | 1 3 6 |
| 7 6 3 | 4 1 8 | 2 5 9 |
+-------+-------+-------+
```

When the program finds no solution, it will output `NO SOLUTION FOUND`, e.g.:
```
% python sudoku.py -i inputs/nosol3.sudoku -s prop

NO SOLUTION FOUND
```

Note that this can be correct (if no solution exists, as is the case for `inputs/nosol3.sudoku`),
or incorrect (if the algorithms are not implemented correctly).

---

## Assignment 1.a: `propagate()`

The recursive search algorithm is already implemented in `sudoku.py` (as `solve_sudoku_prop()`). However, no form of propagation is implemented: `solve_sudoku_prop()` just returns its input `sudoku_possible_values`.

Implement some propagation—that is, change `propagate()` in `sudoku_core.py` so that it does something intelligent.

Make sure that with your implemented `propagate()`, the `prop` algorithm in `sudoku.py` is able to solve `3x3` sudoku inputs efficiently (say, in a matter of seconds). You can use `inputs/hard3.sudoku` to test this.

### Use of `sudoku_possible_values`

One of the arguments given to `propagate()` is `sudoku_possible_values`.
This contains a list of lists of lists.
Each innermost list corresponds to a cell *(i,j)* in the grid (similarly to the way that sudoku inputs and solutions are represented as a list of lists).
However, instead of a single integer per cell, `sudoku_possible_values`
contains a list per cell.
This list contains the values that the search algorithms has not yet ruled out
for cell *(i,j)*.

The idea of `propagate()` is that it inspects the possible values for the different
cells in the grid, that it filters out some values that can be concluded to be impossible
(based on the possible values remaining for other cells),
and that it returns the filtered version of `sudoku_possible_values`.

---

## Assignment 1.b: `solve_sudoku_SAT()`

Implement the `sat` algorithm—that is, change `solve_sudoku_SAT()`
in `sudoku_core.py` so that:
- It returns a solution (in list of lists representation) for a sudoku input,
if a solution exists.
- It finds solutions by encoding the problem into SAT and calling a SAT solver.

For an example of how to call a SAT solver in Python,
see [this example](../examples/sat.ipynb).

---

## Assignment 1.c: `solve_sudoku_CSP()`

Implement the `csp` algorithm—that is, change `solve_sudoku_CSP()`
in `sudoku_core.py` so that:
- It returns a solution (in list of lists representation) for a sudoku input,
if a solution exists.
- It finds solutions by encoding the problem into CSP and calling a CSP solver.

For an example of how to call a CSP solver in Python,
see [this example](../examples/csp.ipynb).

---

## Assignment 1.d: `solve_sudoku_ASP()`

Implement the `asp` algorithm—that is, change `solve_sudoku_ASP()`
in `sudoku_core.py` so that:
- It returns a solution (in list of lists representation) for a sudoku input,
if a solution exists.
- It finds solutions by encoding the problem into ASP and calling an ASP solver.

For an example of how to call an ASP solver in Python,
see [this example](../examples/asp.ipynb).

---

## Assignment 1.e: `solve_sudoku_ILP()`

Implement the `ilp` algorithm—that is, change `solve_sudoku_ILP()`
in `sudoku_core.py` so that:
- It returns a solution (in list of lists representation) for a sudoku input,
if a solution exists.
- It finds solutions by encoding the problem into ILP and calling an ILP solver.

For an example of how to call an ILP solver in Python,
see [this example](../examples/ilp.ipynb).

---

## Evaluation

To get a passing grade for this homework assignment:
- You must follow all instructions, and document your code clearly.
- Your implementation of the solver `prop` must be able to solve sudoku inputs for `k=4` within a matter of seconds (on a typical modern laptop).
- You must have implemented at least one of the solvers `sat`, `csp`, `asp` and `ilp` and your implementation must be able to solve sudoku inputs for `k=5` within less than 30 seconds (on a typical modern laptop).

To get full points for this homework assignment, in addition:
- Your implementation of all of the solvers `sat`, `csp`, `asp` and `ilp` must be able to solve sudoku inputs for `k=5` within less than 30 seconds (on a typical modern laptop).

There are various examples available on the internet that encode the problem of solving Sudoku's into SAT/CSP/ASP/ILP.
You may use these for inspiration.
However, if you do so, you must give clear references, and you must clearly indicate what you did differently.
It is definitely not allowed to submit a solution copied from the internet.
