# Symbolic Systems 1 (UvA, MSc AI, June 2020)

Contents:
1. [Topics](#topics)
1. [Examples](#examples)
1. [Homework assignments](#homework-assignments)

---

## Topics

Here is a short overview of the different topics that we will cover in the course,
together with pointers to reading material.

### (A) Problem solving & search

Subtopics:
- Problem solving and search in rational agents
- Propositional logic and SAT solving
- Constraint programming
- Integer linear programming

Reading material:
- Sections 3.1–3.4 of [[Russell, Norvig, 2016]](#aima).
- Sections 6.1–6.3 of [[Russell, Norvig, 2016]](#aima).
- Sections 7.1–7.6.1 of [[Russell, Norvig, 2016]](#aima).
- Appendix A.1 of [[Russell, Norvig, 2016]](#aima).
- Sections 2.0–2.2.2 of [[Van Harmelen, Lifschitz, Porter, 2008]](#hokr).

### (B) Non-monotonic reasoning and answer set programming

Subtopics:
- Non-monotonic reasoning
- Default logic
- Answer set programming (ASP)
- Problem solving using ASP

Reading material:
- Sections 6.0–6.2 of [[Van Harmelen, Lifschitz, Porter, 2008]](#hokr).
- Sections 7.1–7.3.1 and 7.3.4–7.4 of [[Van Harmelen, Lifschitz, Porter, 2008]](#hokr).
- Sections 2.0–2.3 of [[Gebser, Kaminski, Kaufmann, Schaub, 2012]](#assip).
- Chapter 3 of [[Gebser, Kaminski, Kaufmann, Schaub, 2012]](#assip).

### (C) Automated planning

Subtopics:
- Classical planning
- Planning using problem solving (in particular: ASP)
- Extensions of classical planning

Reading material:
- Sections 10.0–10.2 and 10.4 of [[Russell, Norvig, 2016]](#aima).
- Section 11.3 of [[Russell, Norvig, 2016]](#aima).
- Chapter 8 of [[Gebser, Kaminski, Kaufmann, Schaub, 2012]](#assip)—for the homework assignment.

### (D) Description logics and OWL

Subtopics:
- Description logics
- Web Ontology Language (OWL)
- Modelling and reasoning with ontologies

Reading material:
- Sections 8.0–8.2 of [[Russell, Norvig, 2016]](#aima).
- Sections 12.1–12.2 of [[Russell, Norvig, 2016]](#aima).
- Sections 3.0–3.2 and 3.7 of [[Van Harmelen, Lifschitz, Porter, 2008]](#hokr).

### References:

<a name="aima"></a>
- **[Russell, Norvig, 2016]**: Stuart Russell, and Peter Norvig. [*Artificial Intelligence: A Modern Approach (3rd Ed.)*](http://aima.cs.berkeley.edu/), Prentice Hall, 2016.
<a name="hokr"></a>
- **[Van Harmelen, Lifschitz, Porter, 2008]**: Frank van Harmelen, Vladimir Lifschitz, and Bruce Porter (Eds.). [*Handbook of Knowledge Representation*](https://dai.fmph.uniba.sk/~sefranek/kri/handbook/), Elsevier, 2008.
<a name="assip"></a>
- **[Gebser, Kaminski, Kaufmann, Schaub, 2012]**: Martin Gebser, Roland Kaminski, Benjamin Kaufmann, and Torsten Schaub. [*Answer Set Solving in Practice*](https://potassco.org/book/), Morgan & Claypool, 2012.

---

## Examples

Here are some Python notebooks that illustrate how to use various of the techniques that we cover in the course.

- How to call solvers for different search problems from Python:
  - [SAT solving in Python](examples/sat.ipynb)
  - [ASP solving in Python](examples/asp.ipynb)
  - [CSP solving in Python](examples/csp.ipynb)
  - [ILP solving in Python](examples/ilp.ipynb)
- [A very short guide to integer linear programming](examples/guide-to-ilp.ipynb)
- [A short guide to answer set programming (using clingo and Python)](examples/guide-to-asp.ipynb)
- [How to solve 3-Coloring using SAT solving](examples/3coloring-sat.ipynb)
- [How to solve 3-Coloring using CSP solving](examples/3coloring-csp.ipynb)
- [How to solve 3-Coloring using ILP solving](examples/3coloring-ilp.ipynb)
- [How to solve 3-Coloring using answer set programming](examples/3coloring-asp.ipynb)
- [How to solve Dominating Set using answer set programming](examples/ds-asp.ipynb)

---

## Homework assignments

- [Assignment 1](hw1/assignment.md) (programming)
- [Assignment 2](hw2/assignment.md)
- [Assignment 3](hw3/assignment.md) (programming)

---


