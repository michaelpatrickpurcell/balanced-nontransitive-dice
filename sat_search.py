# SAT strategy
import numpy as np
import pysat
from pysat.solvers import Minisat22
from pysat.pb import PBEnc
from itertools import product

from utils import compare_dice, collapse_values, naturalize_values

# =============================================================================


def build_cardinality_clauses(d, var_dict, var_lists, scores, vpool):
    cardinality_clauses = []
    for var_list, score in zip(var_lists, scores):
        lits = [var_dict[v] for v in var_list]
        cnf = PBEnc.equals(lits=lits, bound=score, vpool=vpool, encoding=0)
        cardinality_clauses += cnf.clauses
    return cardinality_clauses


def build_horizontal_sorting_clauses(d, var_dict, var_pairs):
    # These clauses caputure the implications:
    # if (Xi > Yj) then (Xi > Yk) for k <= j
    horizontal_sorting_clauses = []
    for x, y in var_pairs:
        for i in range(1, (d + 1)):
            for j in range(2, (d + 1)):
                v1 = var_dict[(x + ("%i" % i), y + ("%i" % j))]
                v2 = var_dict[(x + ("%i" % i), y + ("%i" % (j - 1)))]
                horizontal_sorting_clauses.append([-v1, v2])
    return horizontal_sorting_clauses


def build_vertical_sorting_clauses(d, var_dict, var_pairs):
    # These clauses capture the implications
    # if (Xi > Yj) then (Xk > Yj) for k >= i
    vertical_sorting_clauses = []
    for x, y in var_pairs:
        for i in range(1, d):
            for j in range(1, (d + 1)):
                v1 = var_dict[(x + ("%i" % i), y + ("%i" % j))]
                v2 = var_dict[(x + ("%i" % (i + 1)), y + ("%i" % j))]
                vertical_sorting_clauses.append([-v1, v2])
    return vertical_sorting_clauses


def build_sorting_clauses(d, var_dict, var_pairs):
    # These clauses ensure that each constraint matrix is lower triangular
    sorting_clauses = []
    sorting_clauses += build_horizontal_sorting_clauses(d, var_dict, var_pairs)
    sorting_clauses += build_vertical_sorting_clauses(d, var_dict, var_pairs)
    return sorting_clauses


def build_transitivity_clauses(d, var_dict, var_names):
    # These clauses caputure the implications
    # if (Xi > Yj) and (Yj > Zk) then (Xi > Zk)
    # and
    # if (Xi < Yj) and (Yj < Zk) then (Xi < Zk)
    temp = var_names * 2
    var_triplets = [temp[i : (i + 3)] for i in range(len(var_names))]
    transitivity_clauses = []
    for x, y, z in var_triplets:
        for i in range(1, (d + 1)):
            for j in range(1, (d + 1)):
                for k in range(1, (d + 1)):
                    v1 = var_dict[(x + ("%i" % i), y + ("%i" % j))]
                    v2 = var_dict[(y + ("%i" % j), z + ("%i" % k))]
                    v3 = var_dict[(z + ("%i" % k), x + ("%i" % i))]
                    transitivity_clauses.append([v1, v2, v3])
                    transitivity_clauses.append([-v1, -v2, -v3])
    return transitivity_clauses


def build_converse_clauses(d, var_dict, var_pairs, vpool):
    # Converse Clauses
    ## These clauses capture the implication that if (A1 > C1), then ~(C1 > A1)
    converse_clauses = []
    for x, y in var_pairs:
        for i in range(1, (d + 1)):
            for j in range(1, (d + 1)):
                v1 = var_dict[(x + ("%i" % i), y + ("%i" % j))]
                v2 = vars[(y + ("%i" % j), x + ("%i" % i))]
                cnf = PBEnc.equals(lits=[v1, v2], bound=1, vpool=vpool, encoding=0)
                converse_clauses += cnf.clauses
    return converse_clauses


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


# =============================================================================
# Three dice sets
# =============================================================================
d = 3
score = 5

scores = [score] * 3
var_names = ["A", "B", "C"]
face_names = {x: ["%s%i" % (x, i) for i in range(1, d + 1)] for x in var_names}

# Outer cycle
temp = var_names * 2
var_pairs = [tuple(temp[i : (i + 2)]) for i in range(len(var_names))]

var_lists = [list(product(face_names[x], face_names[y])) for x, y in var_pairs]
var_dict = dict((v, k) for k, v in enumerate(sum(var_lists, []), 1))

vpool = pysat.formula.IDPool(start_from=len(var_dict) + 1)

# ----------------------------------------------------------------------------

clauses = []
clauses += build_cardinality_clauses(d, var_dict, var_lists, scores, vpool)
clauses += build_sorting_clauses(d, var_dict, var_pairs)
clauses += build_transitivity_clauses(d, var_dict, var_names)

m = Minisat22()
for clause in clauses:
    m.add_clause(clause)

counter = 0
is_solvable = m.solve()
print(is_solvable)
while is_solvable:
    counter += 1
    res = m.get_model()
    print(counter, res[: (3 * d ** 2)])
    elim = [-1 * r for r in res[: (3 * d ** 2)]]
    m.add_clause(elim)
    is_solvable = m.solve()


# =============================================================================
# Four dice sets
# =============================================================================
d = 4
adj_score = 10
acr_score = 8

scores = [adj_score] * 4 + [acr_score] * 4

var_names = ["A", "B", "C", "D"]
face_names = {x: ["%s%i" % (x, i) for i in range(1, d + 1)] for x in var_names}

# Outer cycle
temp = var_names * 2
var_pairs = [tuple(temp[i : (i + 2)]) for i in range(len(var_names))]
# Cross pairs
var_pairs += [("C", "A"), ("D", "B"), ("A", "C"), ("B", "D")]

var_lists = [list(product(face_names[x], face_names[y])) for x, y in var_pairs]
var_dict = dict((v, k) for k, v in enumerate(sum(var_lists, []), 1))

vpool = pysat.formula.IDPool(start_from=len(var_dict) + 1)

# ----------------------------------------------------------------------------

clauses = []
clauses += build_cardinality_clauses(d, var_dict, var_lists, scores, vpool)
clauses += build_sorting_clauses(d, var_dict, var_pairs)
clauses += build_transitivity_clauses(d, var_dict, var_names)
clauses += build_converse_clauses(d, var_dict, var_pairs[-2:], vpool)

m = Minisat22()
for clause in clauses:
    m.add_clause(clause)

counter = 0
is_solvable = m.solve()
print(is_solvable)
while is_solvable:
    counter += 1
    res = m.get_model()
    print(counter, res[: (8 * d ** 2)])
    elim = [-1 * r for r in res[: (8 * d ** 2)]]
    m.add_clause(elim)
    is_solvable = m.solve()

# =============================================================================
# Five dice sets
# =============================================================================
# d = 3
# adj_score = 5
# acr_score = 4
d = 6
adj_score = 24
acr_score = 16

scores = [adj_score] * 5 + [acr_score] * 5

var_names = ["A", "B", "C", "D", "E"]
face_names = {x: ["%s%i" % (x, i) for i in range(1, d + 1)] for x in var_names}

# Outer cycle
temp = var_names * 2
var_pairs = [tuple(temp[i : (i + 2)]) for i in range(len(var_names))]
# Inner cycle
temp = ["C", "A", "D", "B", "E"] * 2
var_pairs += [tuple(temp[i : (i + 2)]) for i in range(len(var_names))]

var_lists = [list(product(face_names[x], face_names[y])) for x, y in var_pairs]
var_dict = dict((v, k) for k, v in enumerate(sum(var_lists, []), 1))

vpool = pysat.formula.IDPool(start_from=len(var_dict) + 1)

# ----------------------------------------------------------------------------

clauses = []
clauses += build_cardinality_clauses(d, var_dict, var_lists, scores, vpool)
clauses += build_sorting_clauses(d, var_dict, var_pairs)
clauses += build_transitivity_clauses(d, var_dict, var_names)

m = Minisat22()
for clause in clauses:
    m.add_clause(clause)

is_solvable = m.solve()
print(is_solvable)
if is_solvable:
    res = np.array(m.get_model())
    signs = (res[: (10 * d ** 2)] > 0).reshape((10, d, d))
    constraints = {var_pair: sign for var_pair, sign in zip(var_pairs, signs)}

    # ------------------------------------------------------------------------

    A_raw = list(np.arange(1, d + 1))

    lower_bounds = np.zeros(d)
    upper_bounds = (d + 1) * np.ones(d)
    lower_bounds, upper_bounds = update_bounds(
        lower_bounds, upper_bounds, constraints[("A", "B")], A_raw
    )

    B_raw = find_faces(lower_bounds, upper_bounds)

    lower_bounds = np.zeros(d)
    upper_bounds = (d + 1) * np.ones(d)
    lower_bounds, upper_bounds = update_bounds(
        lower_bounds, upper_bounds, constraints[("B", "C")], B_raw
    )
    lower_bounds, upper_bounds = update_bounds(
        lower_bounds, upper_bounds, (constraints[("C", "A")] ^ True).transpose(), A_raw
    )

    C_raw = find_faces(lower_bounds, upper_bounds)

    lower_bounds = np.zeros(d)
    upper_bounds = (d + 1) * np.ones(d)
    lower_bounds, upper_bounds = update_bounds(
        lower_bounds, upper_bounds, constraints[("C", "D")], C_raw
    )
    lower_bounds, upper_bounds = update_bounds(
        lower_bounds, upper_bounds, (constraints[("D", "B")] ^ True).transpose(), B_raw
    )
    lower_bounds, upper_bounds = update_bounds(
        lower_bounds, upper_bounds, constraints[("A", "D")], A_raw
    )

    D_raw = find_faces(lower_bounds, upper_bounds)

    lower_bounds = np.zeros(d)
    upper_bounds = (d + 1) * np.ones(d)
    lower_bounds, upper_bounds = update_bounds(
        lower_bounds, upper_bounds, constraints[("D", "E")], D_raw
    )
    lower_bounds, upper_bounds = update_bounds(
        lower_bounds, upper_bounds, (constraints[("E", "C")] ^ True).transpose(), C_raw
    )
    lower_bounds, upper_bounds = update_bounds(
        lower_bounds, upper_bounds, constraints[("B", "E")], B_raw
    )
    lower_bounds, upper_bounds = update_bounds(
        lower_bounds, upper_bounds, (constraints[("E", "A")] ^ True).transpose(), A_raw
    )

    E_raw = find_faces(lower_bounds, upper_bounds)

    raw_faces = A_raw, B_raw, C_raw, D_raw, E_raw
    natural_faces = naturalize_values(*raw_faces)
    collapsed_faces = collapse_values(*natural_faces)
    A, B, C, D, E = collapsed_faces

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
