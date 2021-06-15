# SAT strategy
import numpy as np
import pysat
from pysat.solvers import Minisat22
from pysat.pb import PBEnc
from itertools import product

from utils import compare_dice, collapse_values, natural_faces

# =============================================================================
# Three dice sets
# =============================================================================
d = 3
score = 5
A_vars = ["A%i" % i for i in range(1, d + 1)]
B_vars = ["B%i" % i for i in range(1, d + 1)]
C_vars = ["C%i" % i for i in range(1, d + 1)]

AB_vars = list(product(A_vars, B_vars))
BC_vars = list(product(B_vars, C_vars))
CA_vars = list(product(C_vars, A_vars))

vars = dict((v, k) for k, v in enumerate(AB_vars + BC_vars + CA_vars, 1))
vpool = pysat.formula.IDPool(start_from=len(vars) + 1)

# Cardinality clauses
cardinality_clauses = []
cnf = PBEnc.equals(
    lits=[vars[v] for v in AB_vars], bound=score, vpool=vpool, encoding=0
)
cardinality_clauses += cnf.clauses
cnf = PBEnc.equals(
    lits=[vars[v] for v in BC_vars], bound=score, vpool=vpool, encoding=0
)
cardinality_clauses += cnf.clauses
cnf = PBEnc.equals(
    lits=[vars[v] for v in CA_vars], bound=score, vpool=vpool, encoding=0
)
cardinality_clauses += cnf.clauses

# Sorting clauses
sorting_clauses = []

## Horizonal implications
## These clauses caputure the implication that if (A3 > B2) then (A3 > B1)
for x, y in (("A", "B"), ("B", "C"), ("C", "A")):
    for i in range(1, (d + 1)):
        for j in range(2, (d + 1)):
            v1 = vars[(x + ("%i" % i), y + ("%i" % j))]
            v2 = vars[(x + ("%i" % i), y + ("%i" % (j - 1)))]
            sorting_clauses.append([-v1, v2])

## Vertical implications
## These clauses capture the implication that if (A2 > B3) (then A3 > B3)
for x, y in (("A", "B"), ("B", "C"), ("C", "A")):
    for i in range(1, d):
        for j in range(1, (d + 1)):
            v1 = vars[(x + ("%i" % i), y + ("%i" % j))]
            v2 = vars[(x + ("%i" % (i + 1)), y + ("%i" % j))]
            sorting_clauses.append([-v1, v2])

# Transitivity Clauses
transitivity_clauses = []
for i in range(1, (d + 1)):
    for j in range(1, (d + 1)):
        for k in range(1, (d + 1)):
            v1 = vars[("A%i" % i, "B%i" % j)]
            v2 = vars[("B%i" % j, "C%i" % k)]
            v3 = vars[("C%i" % k, "A%i" % i)]
            transitivity_clauses.append([v1, v2, v3])
            transitivity_clauses.append([-v1, -v2, -v3])


m = Minisat22()
for clause in cardinality_clauses:
    m.add_clause(clause)

for clause in sorting_clauses:
    m.add_clause(clause)

for clause in transitivity_clauses:
    m.add_clause(clause)

print(m.solve())
res = m.get_model()
print(res)

elim = [-1 * r for r in res[:27]]
m.add_clause(elim)

# =============================================================================
# Four dice sets
# =============================================================================
d = 4
score = 10
cross_score = 8
A_vars = ["A%i" % i for i in range(1, d + 1)]
B_vars = ["B%i" % i for i in range(1, d + 1)]
C_vars = ["C%i" % i for i in range(1, d + 1)]
D_vars = ["D%i" % i for i in range(1, d + 1)]


AB_vars = list(product(A_vars, B_vars))
BC_vars = list(product(B_vars, C_vars))
CD_vars = list(product(C_vars, D_vars))
DA_vars = list(product(D_vars, A_vars))

CA_vars = list(product(C_vars, A_vars))
DB_vars = list(product(D_vars, B_vars))
AC_vars = list(product(A_vars, C_vars))
BD_vars = list(product(B_vars, D_vars))

var_list = AB_vars + BC_vars + CD_vars + DA_vars + CA_vars + DB_vars + AC_vars + BD_vars
vars = dict((v, k) for k, v in enumerate(var_list, 1))
vpool = pysat.formula.IDPool(start_from=len(vars) + 1)

# Cardinality clauses
## Scores for adjacent pairs of dice
cardinality_clauses = []
cnf = PBEnc.equals(
    lits=[vars[v] for v in AB_vars], bound=score, vpool=vpool, encoding=0
)
cardinality_clauses += cnf.clauses
cnf = PBEnc.equals(
    lits=[vars[v] for v in BC_vars], bound=score, vpool=vpool, encoding=0
)
cardinality_clauses += cnf.clauses
cnf = PBEnc.equals(
    lits=[vars[v] for v in CD_vars], bound=score, vpool=vpool, encoding=0
)
cardinality_clauses += cnf.clauses
cnf = PBEnc.equals(
    lits=[vars[v] for v in DA_vars], bound=score, vpool=vpool, encoding=0
)
cardinality_clauses += cnf.clauses

## Scores for non-adjacent pairs of dice
cnf = PBEnc.equals(
    lits=[vars[v] for v in CA_vars], bound=cross_score, vpool=vpool, encoding=0
)
cardinality_clauses += cnf.clauses
cnf = PBEnc.equals(
    lits=[vars[v] for v in DB_vars], bound=cross_score, vpool=vpool, encoding=0
)
cardinality_clauses += cnf.clauses
cnf = PBEnc.equals(
    lits=[vars[v] for v in AC_vars], bound=cross_score, vpool=vpool, encoding=0
)
cardinality_clauses += cnf.clauses
cnf = PBEnc.equals(
    lits=[vars[v] for v in BD_vars], bound=cross_score, vpool=vpool, encoding=0
)
cardinality_clauses += cnf.clauses

# Sorting clauses
sorting_clauses = []
## Horizonal implications
## These clauses caputure the implication that if (A3 > B2) then (A3 > B1)
for x, y in (
    ("A", "B"),
    ("B", "C"),
    ("C", "D"),
    ("D", "A"),
    ("C", "A"),
    ("D", "B"),
    ("A", "C"),
    ("B", "D"),
):
    for i in range(1, (d + 1)):
        for j in range(2, (d + 1)):
            v1 = vars[(x + ("%i" % i), y + ("%i" % j))]
            v2 = vars[(x + ("%i" % i), y + ("%i" % (j - 1)))]
            sorting_clauses.append([-v1, v2])

## Vertical implications
## These clauses capture the implication that if (A2 > B3) then (A3 > B3)
for x, y in (
    ("A", "B"),
    ("B", "C"),
    ("C", "D"),
    ("D", "A"),
    ("C", "A"),
    ("D", "B"),
    ("A", "C"),
    ("B", "D"),
):
    for i in range(1, d):
        for j in range(1, (d + 1)):
            v1 = vars[(x + ("%i" % i), y + ("%i" % j))]
            v2 = vars[(x + ("%i" % (i + 1)), y + ("%i" % j))]
            sorting_clauses.append([-v1, v2])

# Transitivity Clauses
## These clauses caputure the implication that if (A1 > B1) and (B1 > C1) then (A1 > C1)
transitivity_clauses = []
for x, y, z in (("A", "B", "C"), ("B", "C", "D"), ("C", "D", "A"), ("D", "A", "B")):
    for i in range(1, (d + 1)):
        for j in range(1, (d + 1)):
            for k in range(1, (d + 1)):
                v1 = vars[(x + ("%i" % i), y + ("%i" % j))]
                v2 = vars[(y + ("%i" % j), z + ("%i" % k))]
                v3 = vars[(z + ("%i" % k), x + ("%i" % i))]
                transitivity_clauses.append([v1, v2, v3])
                transitivity_clauses.append([-v1, -v2, -v3])

# Converse Clauses
## These clauses capture the implication that if (A1 > C1), then ~(C1 > A1)
converse_clauses = []
for i in range(1, (d + 1)):
    for j in range(1, (d + 1)):
        cnf = PBEnc.equals(
            lits=[vars[("A%i" % i, "C%i" % j)], vars[("C%i" % j, "A%i" % i)]],
            bound=1,
            vpool=vpool,
            encoding=0,
        )
        converse_clauses += cnf.clauses
        cnf = PBEnc.equals(
            lits=[vars[("B%i" % i, "D%i" % j)], vars[("D%i" % j, "B%i" % i)]],
            bound=1,
            vpool=vpool,
            encoding=0,
        )
        converse_clauses += cnf.clauses

m = Minisat22()
for clause in cardinality_clauses:
    m.add_clause(clause)

for clause in sorting_clauses:
    m.add_clause(clause)

for clause in transitivity_clauses:
    m.add_clause(clause)

for clause in converse_clauses:
    m.add_clause(clause)


counter = 0
is_solvable = m.solve()
print(is_solvable)

while is_solvable:
    counter += 1
    print(counter)

    res = m.get_model()
    print(res[: (8 * d ** 2)])

    elim = [-1 * r for r in res[: (8 * d ** 2)]]
    m.add_clause(elim)

    is_solvable = m.solve()

# =============================================================================
# Five dice sets
# =============================================================================
# d = 3
# adj_score = 5
# acr_score = 5

d = 6
adj_score = 24
acr_score = 16

A_vars = ["A%i" % i for i in range(1, d + 1)]
B_vars = ["B%i" % i for i in range(1, d + 1)]
C_vars = ["C%i" % i for i in range(1, d + 1)]
D_vars = ["D%i" % i for i in range(1, d + 1)]
E_vars = ["E%i" % i for i in range(1, d + 1)]

AB_vars = list(product(A_vars, B_vars))
BC_vars = list(product(B_vars, C_vars))
CD_vars = list(product(C_vars, D_vars))
DE_vars = list(product(D_vars, E_vars))
EA_vars = list(product(E_vars, A_vars))

CA_vars = list(product(C_vars, A_vars))
AD_vars = list(product(A_vars, D_vars))
DB_vars = list(product(D_vars, B_vars))
BE_vars = list(product(B_vars, E_vars))
EC_vars = list(product(E_vars, C_vars))

var_list = []
var_list += AB_vars + BC_vars + CD_vars + DE_vars + EA_vars
var_list += CA_vars + AD_vars + DB_vars + BE_vars + EC_vars
vars = dict((v, k) for k, v in enumerate(var_list, 1))
vpool = pysat.formula.IDPool(start_from=len(vars) + 1)

# Cardinality clauses
## Scores for adjacent pairs of dice
cardinality_clauses = []
var_lists = []
var_lists += [AB_vars, BC_vars, CD_vars, DE_vars, EA_vars]
for var_list in var_lists:
    cnf = PBEnc.equals(
        lits=[vars[v] for v in var_list], bound=adj_score, vpool=vpool, encoding=0,
    )
    cardinality_clauses += cnf.clauses

var_lists = []
var_lists += [CA_vars, AD_vars, DB_vars, BE_vars, EC_vars]
for var_list in var_lists:
    cnf = PBEnc.equals(
        lits=[vars[v] for v in var_list], bound=acr_score, vpool=vpool, encoding=0,
    )
    cardinality_clauses += cnf.clauses


# Sorting clauses
sorting_clauses = []
## Horizonal implications
## These clauses caputure the implication that if (A3 > B2) then (A3 > B1)
var_pairs = []
var_pairs += [("A", "B"), ("B", "C"), ("C", "D"), ("D", "E"), ("E", "A")]
var_pairs += [("C", "A"), ("A", "D"), ("D", "B"), ("B", "E"), ("E", "C")]
for x, y in var_pairs:
    for i in range(1, (d + 1)):
        for j in range(2, (d + 1)):
            v1 = vars[(x + ("%i" % i), y + ("%i" % j))]
            v2 = vars[(x + ("%i" % i), y + ("%i" % (j - 1)))]
            sorting_clauses.append([-v1, v2])

## Vertical implications
## These clauses capture the implication that if (A2 > B3) then (A3 > B3)
for x, y in var_pairs:
    for i in range(1, d):
        for j in range(1, (d + 1)):
            v1 = vars[(x + ("%i" % i), y + ("%i" % j))]
            v2 = vars[(x + ("%i" % (i + 1)), y + ("%i" % j))]
            sorting_clauses.append([-v1, v2])

# Transitivity Clauses
## These clauses caputure the implication that if (A1 > B1) and (B1 > C1) then (A1 > C1)
var_triplets = []
var_triplets += [
    ("A", "B", "C"),
    ("B", "C", "D"),
    ("C", "D", "E"),
    ("D", "E", "A"),
    ("E", "A", "B"),
]
transitivity_clauses = []
for x, y, z in var_triplets:
    for i in range(1, (d + 1)):
        for j in range(1, (d + 1)):
            for k in range(1, (d + 1)):
                v1 = vars[(x + ("%i" % i), y + ("%i" % j))]
                v2 = vars[(y + ("%i" % j), z + ("%i" % k))]
                v3 = vars[(z + ("%i" % k), x + ("%i" % i))]
                transitivity_clauses.append([v1, v2, v3])
                transitivity_clauses.append([-v1, -v2, -v3])

# ----------------------------------------------------------------------------

m = Minisat22()
for clause in cardinality_clauses:
    m.add_clause(clause)

for clause in sorting_clauses:
    m.add_clause(clause)

for clause in transitivity_clauses:
    m.add_clause(clause)

counter = 0
is_solvable = m.solve()
print(is_solvable)
res = np.array(m.get_model())
signs = res[: (10 * d ** 2)] > 0

AB_constraints = signs[: (d ** 2)].reshape(d, d)
BC_constraints = signs[(d ** 2) : 2 * (d ** 2)].reshape(d, d)
CD_constraints = signs[2 * (d ** 2) : 3 * (d ** 2)].reshape(d, d)
DE_constraints = signs[3 * (d ** 2) : 4 * (d ** 2)].reshape(d, d)
EA_constraints = signs[4 * (d ** 2) : 5 * (d ** 2)].reshape(d, d)

CA_constraints = signs[5 * (d ** 2) : 6 * (d ** 2)].reshape(d, d)
AD_constraints = signs[6 * (d ** 2) : 7 * (d ** 2)].reshape(d, d)
DB_constraints = signs[7 * (d ** 2) : 8 * (d ** 2)].reshape(d, d)
BE_constraints = signs[8 * (d ** 2) : 9 * (d ** 2)].reshape(d, d)
EC_constraints = signs[9 * (d ** 2) : 10 * (d ** 2)].reshape(d, d)

# ----------------------------------------------------------------------------


def update_bounds(lower_bounds, upper_bounds, constraints, faces):
    for i in range(d):
        if i > 0:
            lower_bounds[i] = max(lower_bounds[i], lower_bounds[i - 1])
        for j in range(d):
            if constraints[j, i]:
                upper_bounds[i] = min(faces[j], upper_bounds[i])
            else:
                lower_bounds[i] = max(faces[j], lower_bounds[i])
    return (lower_bounds, upper_bounds)


def find_faces(lower_bounds, upper_bounds):
    faces = []
    faces.append((lower_bounds[0] + upper_bounds[0]) / 2)
    for i in range(1, d):
        lower_bounds[i] = max(lower_bounds[i], faces[-1])
        faces.append((lower_bounds[i] + upper_bounds[i]) / 2)
    return faces


A_raw = list(np.arange(1, d + 1))

lower_bounds = np.zeros(d)
upper_bounds = (d + 1) * np.ones(d)
lower_bounds, upper_bounds = update_bounds(
    lower_bounds, upper_bounds, AB_constraints, A_raw
)

B_raw = find_faces(lower_bounds, upper_bounds)

lower_bounds = np.zeros(d)
upper_bounds = (d + 1) * np.ones(d)
lower_bounds, upper_bounds = update_bounds(
    lower_bounds, upper_bounds, BC_constraints, B_raw
)
lower_bounds, upper_bounds = update_bounds(
    lower_bounds, upper_bounds, (CA_constraints ^ True).transpose(), A_raw
)

C_raw = find_faces(lower_bounds, upper_bounds)

lower_bounds = np.zeros(d)
upper_bounds = (d + 1) * np.ones(d)
lower_bounds, upper_bounds = update_bounds(
    lower_bounds, upper_bounds, CD_constraints, C_raw
)
lower_bounds, upper_bounds = update_bounds(
    lower_bounds, upper_bounds, (DB_constraints ^ True).transpose(), B_raw
)
lower_bounds, upper_bounds = update_bounds(
    lower_bounds, upper_bounds, AD_constraints, A_raw
)

D_raw = find_faces(lower_bounds, upper_bounds)

lower_bounds = np.zeros(d)
upper_bounds = (d + 1) * np.ones(d)
lower_bounds, upper_bounds = update_bounds(
    lower_bounds, upper_bounds, DE_constraints, D_raw
)
lower_bounds, upper_bounds = update_bounds(
    lower_bounds, upper_bounds, (EC_constraints ^ True).transpose(), C_raw
)
lower_bounds, upper_bounds = update_bounds(
    lower_bounds, upper_bounds, BE_constraints, B_raw
)
lower_bounds, upper_bounds = update_bounds(
    lower_bounds, upper_bounds, (EA_constraints ^ True).transpose(), A_raw
)

E_raw = find_faces(lower_bounds, upper_bounds)

A_nat, B_nat, C_nat, D_nat, E_nat = natural_faces(d, A_raw, B_raw, C_raw, D_raw, E_raw)
A, B, C, D, E = collapse_values([A_nat, B_nat, C_nat, D_nat, E_nat])

print(compare_dice(A, B))
print(compare_dice(B, C))
print(compare_dice(C, D))
print(compare_dice(D, E))
print(compare_dice(E, A))
print()
print(compare_dice(C, A))
print(compare_dice(A, D))
print(compare_dice(D, B))
print(compare_dice(B, E))
print(compare_dice(E, C))
