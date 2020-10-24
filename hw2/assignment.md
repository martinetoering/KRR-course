# Homework assignment 2

In this set of homework assignments, you will answer some theoretical questions about *answer set programming* and *monotonic and non-monotonic reasoning*.

## Instructions

1. Read the description of the assignments.
1. Provide your answers to the questions, preferably typeset with LaTeX and using [this template](../templates/homework.tex).
1. Submit your solutions via [Canvas](https://canvas.uva.nl/courses/10768).

The difficulty of the assignments is indicated with stars (&star;'s): the more stars, the harder the assignment.

---

## Assignment 1 (&star;): monotonic reasoning using SAT

In this exercise, you will explain how to use an algorithm for SAT (that is, satisfiability of propositional logic formulas in CNF) to perform propositional reasoning.
To make this concrete, for this exercise you may consider propositional reasoning to be the following computational task.

**Computational task (propositional reasoning):** Given a set &Phi; of propositional logic formulas, and another formula &varphi;, decide whether &varphi; is logically entailed by &Phi; (written &Phi; &models; &varphi;)—that is, whether for all truth assignments &alpha; that make &Phi; true it holds that &alpha; also makes &varphi; true.

*Assignment:* Describe how to solve this computational task efficiently under the assumption that you have an efficient algorithm for deciding the satisfiability of propositional CNF formulas.
(In other words: reduce the task of propositional reasoning to deciding SAT for propositional CNF formulas.)

*Remarks:*
- Note that the formulas in &Phi; and the formula &varphi; may be propositional logic formulas of arbitrarily complex structure, where the satisfiability algorithm that you may use only works for formulas in conjunctive normal form.
- Describe an algorithm for the task of propositional reasoning in enough detail so that one of your class mates can understand how and why it works, and avoid spelling out unnecessary details. (This can be a tricky balance.)
- If the working of your algorithm is based on theoretical results (e.g., theoretical results from the reading material for the course), clearly explain which results.

---

## Assignment 2 (&star;): monotonic reasoning using ASP

In this exercise, you will explain how to use an algorithm for answer set programming to perform propositional reasoning.

*Assignment:* Describe how to solve the computational task of propositional reasoning (see Assignment 1) efficiently under the assumption that you have an efficient algorithm for deciding if a given answer set program has at least one answer set.
(In other words: reduce the task of propositional reasoning to answer set programming.)

(The same remarks hold for this assignment as for Assignment 1.)

---

## Assignment 3 (&star;&star;): encoding default logic in ASP

In this exercise, you will explain how to use an algorithm for answer set programming to perform non-monotonic reasoning in a particular fragment of default logic.

Consider default rules of the following form: `P : C / C`, where both `P` and `C` are conjunctions of propositional literals. (Such default rules are called *propositional disjunction-free normal default rules*.) Moreover, consider a default theory `(W,D)`, where `W` is a set of propositional literals and `D` is a set of disjunction-free normal default rules. We call such a default theory a *PDFN default theory*.

For example, the following specifies a PDFN default theory `(W,D)` (where logical conjunction is written as `&` and negation as `~`):
```
W = p & q
D = {
  q : r / r,
  p & r : ~s & q / ~s & q
}
```

Then consider the following reasoning task:

**Computational task (PDFN reasoning):** Given a PDFN default theory `(W,D)` and a propositional literal `l`, decide whether there is an extension of `(W,D)` that includes `l`.

*Assignment:* Describe how to solve the computational task of PDFN reasoning efficiently under the assumption that you have an efficient algorithm for deciding if a given answer set program has at least one answer set.
(In other words: reduce the task of PDFN reasoning to answer set programming.)

*Remarks:*
- Describe an algorithm for the task of propositional reasoning in enough detail so that one of your class mates can understand how and why it works, and avoid spelling out unnecessary details. (This can be a tricky balance.)
- If the working of your algorithm is based on results (e.g., from the reading material for the course), clearly explain which results.

*Hints:*
- Begin with the example of a PDFN default theory `(W,D)` that is mentioned above. Figure out what its extensions are. Encode this PDFN default theory into an answer set program `P`, in such a way that the answer sets of `P` are in direct correspondence to the extensions of `(W,D)`. Then generalize your approach to arbitrary PDFN default theories.
- For this assignment, think of answer set programming as a problem solving paradigm (rather than a paradigm for non-monotonic reasoning).

---

## Assignment 4 (&star;&star;): expressing cardinality rules in basic ASP

In this assignment, you will show that cardinality rules are not a 'true' language extension for basic answer set programming. In particular, you will show how to encode the following using an ASP program without cardinality rules (and without choice rules, aggregates, etc), for every value of `n`, `m`, and `u`:

```
#const n=2.
#const m=3.
#const u=4.

n { item(1..u) } m.
```

*Assignment:* Give an answer set program `P` (in the basic language of answer set programming) that when combined with the line `#show item/1.` yields exactly the same answer sets as the answer set program consisting of the line `n { item(1..u) } m.`, for each combination of declared values for the constants `n`, `m` and `u`. Explain how your answer set program `P` works. The basic language for answer set programming only includes rules of the form `a :- b1, ..., bn, not c1, ..., not cm.` (this includes facts `a.` and constraints `:- b1, ..., bn, not c1, ..., not cm.`, for example), and does not include cardinality rules, choice rules, aggregates, etc.

*Hints:*
- Begin with example values for `n`, `m` and `u`. For example, with `#const n=2.`, `#const m=3.` and `#const u=4.` Then, make sure that your program works also for other values of `n`, `m` and `u`.

---

## Assignment 5 (&star;&star;&star;): disjunction is a true extension of ASP

In this assignment, you will show that disjunction in the head of rules is a 'true' language extension for basic answer set programming. In other words, you will show that answer set programs with disjunction in the head of rules (e.g., with rules of the form `a ; b :- c.`) are more powerful than answer set programs without disjunction in the head of rules.

*Assignment:*
Give an example of a problem that (a) can be encoded effectively as the problem of finding an answer set of a program with disjunction in the head of rules, but that (b) cannot be encoded effectively as the problem of finding  an answer set of a program without disjunction in the head of rules.

*Remarks:*
- Explain clearly (i) what the problem is, (ii) how you can encode the problem effectively into answer set programming with disjunction in the head of rules, and (iii) why this cannot be encoded effectively into answer set programming without disjunction in the head of rules.
- With *effectively* we mean that there is, say, no exponential blow-up in the answer set program and that the encoding does not take exponential time to compute.
- For this assignment, you may assume that the problem of deciding if a propositional logic formula is satisfiable (i.e., the SAT problem) takes exponential time. Note, however, that the SAT problem can be effectively encoded as an answer set program without disjunction in the heads of rules, so the SAT problem itself is not a good answer for this assignment.

*Hints:*
- Search the web for an ASP modelling technique called *saturation*.

---

## Evaluation

Each of the assignments (1–5) is worth 2 points, yielding a total of 10 points.
