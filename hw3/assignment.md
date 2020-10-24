# Homework assignment 3

In this assignment, you will use the problem solving paradigm of answer set programming to solve the problem of automated reasoning (in a particular setting).

You will finish a Python program ([asp_planner.py](asp_planner.py)) by
implementing the core algorithm (in [asp_planner_core.py](asp_planner_core.py)) for the program.

## Instructions

1. Read the description of the assignments.
1. Provide your implementation of the function `solve_planning_problem_using_ASP()`, that uses the `clingo` package for answer set programming.
(You can test your implementation using [asp_planner.py](asp_planner.py), that refers to [asp_planner_core.py](asp_planner_core.py), and using the files in [inputs](./inputs).)
1. Explain clearly what you did, and how your encoding work, in comments in `asp_planner_core.py`.
1. Submit your version of the file `asp_planner_core.py` (with your implementation and comments) via [Canvas](https://canvas.uva.nl/courses/10768).

---

## Planning

Here follows a description of the automated planning setting that your function `solve_planning_problem_using_ASP()` should be able to deal with,
as well as the data structures used to represent planning problems and actions (as defined in [planning.py](planning.py)).

### Expressions

To represent (quantifier-free first-order logic) expressions, we use the class `Expr`.
Each instance `e` of `Expr` contains an operator (`e.op`)
and arguments (`e.args`). The operator is a string specifying the name of the predicate (or logical operator),
and the arguments are specified by a tuple containing the arguments of the predicate (or logical operator). For the arguments of `e` we use (other) instances of the class `Expr`.

We also use the function `expr()`, which gives us an convenient method to define `Expr` objects from a string. For example, `expr('Predicate(A,B)')` returns an `Expr` object with operator `'Predicate'` and two arguments (expressions with operators `'A'` and `'B'`, respectively, and no arguments).

We can also use logical operators to build expressions. For example, we can use `expr('A & B')` to get an expression with operator `'&'` (representing logical conjunction) and two arguments (expressions formed by `expr('A')` and `expr('B')`). For logical disjunction, we use `'|'`. Negation is represented by the unary operator `'~'`, e.g., as in `expr('~A')`.

We use expressions formed using strings that start with an uppercase letter for predicates and propositions (which can be seen as 0-ary predicates), and we use expressions formed using strings that start with a lowercase letter for variables (e.g., `expr('x')`).

### Planning problems

Planning problems are represented as instances of the class `PlanningProblem`.
Each such instance specifies an *initial state*, *goals*, and a set of possible *actions*.

States are represented as a list of expressions, representing all statements (without variables, and without logical operators) that hold in the state. For example, `[expr('Near(A,B)'), expr('Near(C,D)')]` represents a state where the statements `Near(A,B)` and `Near(C,D)` are true (and all other expressions are false).

The initial state of a `PlanningProblem` instance is given as a list of expressions representing this state.

The goals of a `PlanningProblem` instance are also given as a list of expressions (without binary logical operators),
representing the statements that should be true in a state for the goals to be reached—but additional statements are allowed to be true. Goals may also contain negated statements. For example, the goals represented by `[expr('Near(A,B)'), expr('~Near(C,D)')]` are satisfied in a state where the statement `Near(A,B)` is true and the statement `Near(C,D)` is false (regardless of the truth of other statements).

The actions of a `PlanningProblem` instance are given as a list of `Action` instances, representing the actions available in the planning problem.

#### Actions

Actions are represented as instances of the class `Action`.
Each such action has a name, and specifies a *precondition*, and *effects*.

An action has a (predicate) name, and some variables as arguments.

The preconditions of an action are represented as a list of expressions (without binary logical operators), that represent the statements that should be true or false for an action to be applicable. These expressions may contain variables that are named in the action.

The effects of an action are represented as a list of expressions (without binary logical operators), that represent which statements should become true or false after applying this action. Again, these expressions may contain variables that are named in the action.

The variables that are used in an action will be instantiated by other expressions (containing no variables).

When constructing an action, you can specify a string that expresses a conjunction of statements for the precondition and effects.

For example, the following specifies an action named `Move`, with variables `x`, `a` and `b`. The preconditions of this action specify that `Object(x)`, `Location(a)`, `Location(b)`, and `At(x,a)` should be true (after instantiating the variables) and that `At(x,b)` should be false
in any state where this action is applicable.
The effects state that after applying this action `At(x,b)` becomes true
and `At(x,a)` becomes false (after instantiating the variables).

```python
action0 = Action(
  'Move(x, a, b)',
  precond='Object(x) & Location(a) & Location(b) & At(x, a) & ~At(x, b)',
  effect='At(x, b) & ~At(x, a)'
);
```

### Plans

A plan consists of a list of expressions that each specify an application of an action (or an action instance).
Such expressions contain the name of an action, and (variable-free) expressions in place of the arguments named in the action.

For example, the expression `expr('Move(Book, Room1, Room2)')` specifies an instance of the action `Move(x, a, b)` that we described above, where `x` is instantiated by `Book`, `a` by `Room1`, and `b` by `Room2`.

### An example

Putting this all together, let's consider a [simple example](example.ipynb)
where we create a planning problem,
together with a plan for this planning problem that achieves the goals in the planning problem.

### Some conditions on planning problems that we assume

For this exercise, you may assume that all planning problems satisfy the following conditions:
- Each variable named in an action occurs in a non-negated expression in the precondition of the action.
- Each variable named in the goals occurs in a non-negated expression in the goals.

## Calling `asp_planner.py`

The program `asp_planner.py` can be used by calling:
```
% python asp_planner.py -i INPUT
```
where `INPUT` is replaced by a file containing a planning problem input.

You can add the optional flag `-v` to show more information during the process of solving (enable verbose mode).

For example:
```
% python asp_planner.py -i inputs/easy0.planning

[LeftSock, RightSock, RightShoe, LeftShoe]
```
Or:
```
% python asp_planner.py -i inputs/easy0.planning -v

Reading planning problem and bound on plan length from inputs/easy0.planning..
Planning problem:
--- Planning problem ---
Initial: []
Goals: [RightShoeOn, LeftShoeOn]
Actions:
  * RightShoe
    precondition: [RightSockOn]
    effect: [RightShoeOn]
  * RightSock
    precondition: []
    effect: [RightSockOn]
  * LeftShoe
    precondition: [LeftSockOn]
    effect: [LeftShoeOn]
  * LeftSock
    precondition: []
    effect: [LeftSockOn]
---
Upper bound on plan length: 5
Solving planning problem using ASP encoding..
Did ASP encoding & solving in 0.01 seconds
Correct plan found:
[LeftSock, RightSock, RightShoe, LeftShoe]
```

When the program finds no solution, it will output `NO PLAN FOUND`, e.g.:
```
% python asp_planner.py -i inputs/nosol0.planning

NO PLAN FOUND
```

Note that this can be correct (if no solution exists, as is the case for `inputs/nosol0.planning`),
or incorrect (if the algorithm is not implemented correctly).

### Additional functions in `asp_planner.py`

The file [asp_planner.py](asp_planner.py) additionally contains auxiliary functions, including functions that:
- read `PlanningProblem` instances (as well as an upper bound on the plan length) from a file (`read_problem_from_file()`),
- write `PlanningProblems` instances (as well as an upper bound on the plan length) to a file (`write_planning_problem_to_file()`), and
- verify that a given plan is correct for a given planning problem (`verify_plan()`).

---

## Assignment: `solve_planning_problem_using_ASP()`

Implement the function `solve_planning_problem_using_ASP()` in [asp_planner_core.py](asp_planner_core.py).
This function takes as arguments `planning_problem` (a planning problem, in the form of an instance of the `PlanningProblem` class) and an integer `t_max` (specifying an upper bound on the length of plans that the function `solve_planning_problem_using_ASP()` should consider).

The function `solve_planning_problem_using_ASP()` should return a plan that, when applied to the initial state in `planning_problem` should result in a state that satisfies the goals in `planning_problem`, that:
- is of length at most `t_max`, and
- is of the shortest length of all such plans,

if such a plan exists.
If no such plan (of length at most `t_max`) exists,
the function `solve_planning_problem_using_ASP()` should return `None`.

In other words, the function `solve_planning_problem_using_ASP()` should find a shortest plan for the planning problem, taking into account the maximum plan length `t_max`.

Moreover, your implementation of `solve_planning_problem_using_ASP()` should solve this problem by:
1. encoding the problem into answer set programming,
1. using the clingo python package to find an optimal answer set for the encoding, and
1. translating the output given by clingo into a solution (a shortest plan).

<!--
---

## Evaluation

To get a passing grade for this homework assignment:
- You must follow all instructions, and document your code clearly.
- Your implementation of the solver `prop` must be able to solve sudoku inputs for `k=4` within a matter of seconds (on a typical modern laptop).
- You must have implemented at least one of the solvers `sat`, `csp`, `asp` and `ilp` and your implementation must be able to solve sudoku inputs for `k=5` within less than 30 seconds (on a typical modern laptop).

To get full points for this homework assignment, in addition:
- Your implementation of all of the solvers `sat`, `csp`, `asp` and `ilp` must be able to solve sudoku inputs for `k=5` within less than 30 seconds (on a typical modern laptop).

You may use solutions for similar problems that you find on the internet for inspiration.
However, if you do so, you must give clear references, and you must clearly indicate what you did differently.
It is definitely not allowed to submit a solution copied from the internet.
-->
