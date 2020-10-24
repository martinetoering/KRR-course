import clingo
import itertools
import numpy as np
import math

asp_code = ""

n = 2
m = 4
u = 3

# asp_code = """#const n={}.
# #const m={}.
# #const u={}.
# """.format(n, m, u)

# asp_code += """n { item(1..u) } m.
# """

# asp_code += """given_item(1..u).
# not_item(I):- not item(I), given_item(I).
# item(I):- not not_item(I), given_item(I).
# ctr(I, K+1):- ctr(I-1, K), item(I).
# ctr(I, K):- ctr(I-1, K), not_item(I).
# :- ctr(u, K), K > m.
# :- ctr(u, K), K < n.
# ctr(0, 0).
# """

###################################


asp_code += """binary(1..10).
{ choice(X): binary(X)}.
#show choice/1.\n
"""

print(asp_code)

control = clingo.Control()
control.add("base", [], asp_code)
control.ground([("base", [])])

def on_model(model):
    print(model.symbols(shown=True))

control.configuration.solve.models = 0;
answer = control.solve(on_model=on_model)

if answer.satisfiable:
    print("satisfiable:")
else:
    print("unsatisfiable")











################################ 




# asp_code += """b.
# item(1..u) :- b.
# c :- d, not e.
# d:- ctrl(1, n).
# e:- ctru(1, m+1).
# :- b, not c.
# """


# asp_code += """ctrl(I, 1) :- item(I), I >= n, I <= u.
# """
# asp_code += """ctrl(I, 1) :- not item(I-u), I >= (max(n, u+1)), I <= u+v."""

# asp_code += """ctrl(I, K) :- ctrl(I, K-1), item(I), K > 1, K <= n, I >= n-I+1, I <= u.
# """
# asp_code += """ctrl(I, K) :- ctrl(I+1, K), not item(I), K > 1, K <= n, I >= n-I+1, I <= u.
# """

# asp_code += """ctrl(I, K) :- ctrl(I, K-1), not item(I-u), K > 1, K <= n, I >= max(n, u+1), I <= u+v-K+1.
# """
# asp_code += """ctrl(I, K) :- ctrl(I+1, K), not not item(I-u), K > 1, K <= n, I >= max(n, u+1), I < u+v-I+1.
# """





# asp_code += """ctru(J, 1) :- item(J), J >= m, J <= u.
# """
# asp_code += """ctru(J, 1) :- not item(J-u), J >= (max(m, u+1)), J <= u+v."""

# asp_code += """ctru(J, K) :- ctru(J, K-1), item(J), K > 1, K <= m, J >= n-J+1, J <= u.
# """
# asp_code += """ctru(J, K) :- ctru(J+1, K), not item(J), K > 1, K <= m, J >= m-J+1, J <= u.
# """


# asp_code += """ctru(J, K) :- ctru(J, K-1), not item(J-u), K > 1, K <= m, J >= max(m, u+1), J <= u+v-K+1.
# """
# asp_code += """ctru(J, K) :- ctru(J+1, K), not not item(J-u), K > 1, K <= m, J >= max(m, J+1), J < u+v-J+1.
# """



