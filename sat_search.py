# SAT strategy
import numpy as np
import pysat
from pysat.solvers import Minisat22
from pysat.pb import PBEnc
from itertools import product, permutations

from utils import compare_dice, collapse_values, naturalize_values

# =============================================================================


def build_clauses(d, dice_names, scores):
    """
    Build the clauses that describe the SAT problem.
    """
    dice_pairs = list(permutations(dice_names, 2))
    n = len(dice_pairs)
    f = {x: ["%s%i" % (x, i) for i in range(1, d + 1)] for x in dice_names}
    var_lists = {(x, y): list(product(f[x], f[y])) for (x, y) in dice_pairs}

    variables = sum(var_lists.values(), [])
    var_dict = dict((v, k) for k, v in enumerate(variables, 1))

    vpool = pysat.formula.IDPool(start_from=n * d ** 2 + 1)

    clauses = []
    clauses += build_cardinality_clauses(d, var_dict, var_lists, scores, vpool)
    clauses += build_converse_clauses(d, var_dict, dice_names, vpool)
    clauses += build_sorting_clauses(d, var_dict, dice_names)
    clauses += build_transitivity_clauses(d, var_dict, dice_names)
    clauses += build_symmetry_clauses(d, var_dict, dice_names)
    return clauses


def build_cardinality_clauses(d, var_dict, var_lists, scores, vpool):
    """
    These clauses ensure that each pair of dice have the specified relationship.
    """
    dice_pairs = var_lists.keys()
    cardinality_clauses = []
    for dice_pair in dice_pairs:
        var_list = var_lists[dice_pair]
        score = scores[dice_pair]
        lits = [var_dict[v] for v in var_list]
        cnf = PBEnc.equals(lits=lits, bound=score, vpool=vpool, encoding=0)
        cardinality_clauses += cnf.clauses
    return cardinality_clauses


def build_horizontal_sorting_clauses(d, var_dict, dice_names):
    """
    These clauses caputure the implications:
        if (Xi > Yj) then (Xi > Yk) for k <= j
    """
    horizontal_sorting_clauses = []
    dice_pairs = list(permutations(dice_names, 2))
    for x, y in dice_pairs:
        for i in range(1, (d + 1)):
            for j in range(2, (d + 1)):
                v1 = var_dict[(x + ("%i" % i), y + ("%i" % j))]
                v2 = var_dict[(x + ("%i" % i), y + ("%i" % (j - 1)))]
                horizontal_sorting_clauses.append([-v1, v2])
    return horizontal_sorting_clauses


def build_vertical_sorting_clauses(d, var_dict, dice_names):
    """
    These clauses capture the implications:
        if (Xi > Yj) then (Xk > Yj) for k >= i
    """
    vertical_sorting_clauses = []
    dice_pairs = list(permutations(dice_names, 2))
    for x, y in dice_pairs:
        for i in range(1, d):
            for j in range(1, (d + 1)):
                v1 = var_dict[(x + ("%i" % i), y + ("%i" % j))]
                v2 = var_dict[(x + ("%i" % (i + 1)), y + ("%i" % j))]
                vertical_sorting_clauses.append([-v1, v2])
    return vertical_sorting_clauses


def build_sorting_clauses(d, var_dict, dice_names):
    """
    These clauses ensure that each constraint matrix is lower triangular.
    """
    sorting_clauses = []
    sorting_clauses += build_horizontal_sorting_clauses(d, var_dict, dice_names)
    sorting_clauses += build_vertical_sorting_clauses(d, var_dict, dice_names)
    return sorting_clauses


def build_transitivity_clauses(d, var_dict, dice_names):
    """
    These clauses caputure the implications
        if (Xi > Yj) and (Yj > Zk) then (Xi > Zk)
    and
        if (Xi < Yj) and (Yj < Zk) then (Xi < Zk)
    """
    transitivity_clauses = []
    dice_triplets = list(permutations(dice_names, 3))
    for x, y, z in dice_triplets:
        for i in range(1, (d + 1)):
            for j in range(1, (d + 1)):
                for k in range(1, (d + 1)):
                    v1 = var_dict[(x + ("%i" % i), y + ("%i" % j))]
                    v2 = var_dict[(y + ("%i" % j), z + ("%i" % k))]
                    v3 = var_dict[(z + ("%i" % k), x + ("%i" % i))]
                    transitivity_clauses.append([v1, v2, v3])
                    transitivity_clauses.append([-v1, -v2, -v3])
    return transitivity_clauses


def build_converse_clauses(d, var_dict, dice_names, vpool):
    """
    These clauses capture the implications:
        if (A1 > C1), then ~(C1 > A1)
    """
    converse_clauses = []
    dice_pairs = list(permutations(dice_names, 2))
    for x, y in dice_pairs:
        for i in range(1, (d + 1)):
            for j in range(1, (d + 1)):
                v1 = var_dict[(x + ("%i" % i), y + ("%i" % j))]
                v2 = var_dict[(y + ("%i" % j), x + ("%i" % i))]
                cnf = PBEnc.equals(lits=[v1, v2], bound=1, vpool=vpool, encoding=0)
                converse_clauses += cnf.clauses
    return converse_clauses


def build_symmetry_clauses(d, var_dict, dice_names):
    """
    These clauses ensure that A1 is the smallest face.
    """
    symmetry_clauses = []
    v0 = dice_names[0]
    for v in dice_names[1:]:
        for i in range(1, d + 1):
            symmetry_clauses.append([-var_dict[(v0 + "1", v + ("%i" % i))]])
            symmetry_clauses.append([var_dict[(v + ("%i" % i), v0 + "1")]])
    return symmetry_clauses


# ----------------------------------------------------------------------------


def sat_to_dice(d, dice_names, sat_solution, compress=True):
    dice_pairs = list(permutations(dice_names, 2))
    n = len(dice_pairs)

    signs_array = (sat_solution[: (n * d ** 2)] > 0).reshape((n, d, d))
    constraints = {v: s for v, s in zip(dice_pairs, signs_array)}

    raw_faces = recover_raw_values(d, dice_names, constraints)
    natural_faces = naturalize_values(*raw_faces)
    if compress:
        dice_faces = collapse_values(*natural_faces)
        dice_dict = {k: v for k, v in zip(dice_names, dice_faces)}
    else:
        dice_dict = {k: v for k, v in zip(dice_names, natural_faces)}
    return dice_dict


def recover_raw_values(d, var_names, constraints):
    """
    Find real-valued faces that satisfy the given constraints
    """
    raw_faces = dict()
    raw_faces[var_names[0]] = list(np.arange(1, d + 1))

    for var_name in var_names[1:]:
        lower_bounds = np.zeros(d)
        upper_bounds = (d + 1) * np.ones(d)
        for k in raw_faces:
            lower_bounds, upper_bounds = update_bounds(
                lower_bounds, upper_bounds, constraints[(k, var_name)], raw_faces[k]
            )
        raw_faces[var_name] = find_faces(lower_bounds, upper_bounds)
    return [raw_faces[v] for v in var_names]


def update_bounds(lower_bounds, upper_bounds, constraints, faces):
    """
    Derive bounds on possible values from the specified constraints.
    """
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
    """
    Find values for the faces that conform to the constraints
    imposed by the other dice and that are monotonically increasing.
    """
    faces = []
    faces.append((lower_bounds[0] + upper_bounds[0]) / 2)
    for i in range(1, d):
        # Update lower bound to ensure that the value of the current
        # face witll be larger than the value of the previous face.
        lower_bounds[i] = max(lower_bounds[i], faces[-1])
        faces.append((lower_bounds[i] + upper_bounds[i]) / 2)
    return faces


def verify_solution(scores, dice_solution):
    for x, y in scores:
        check = compare_dice(dice_solution[x], dice_solution[y])
        print((x, y), check, scores[(x, y)])


# =============================================================================
# Three-dice sets
# =============================================================================
dice_names = ["A", "B", "C"]
m = len(dice_names)

dice_pairs = list(permutations(dice_names, 2))
n = len(dice_pairs)

d = 18  # 34  # 31  # 13  # 3
score = 198  # 714  # 589  # 104  # 5

temp = [score, d ** 2 - score]
S = [[temp[(j - i) % (m - 1)] for j in range(m - 1)] for i in range(m)]
scores = {p: s for p, s in zip(dice_pairs, sum(S, []))}

# ----------------------------------------------------------------------------

clauses = build_clauses(d, dice_names, scores)

sat = Minisat22()
for clause in clauses:
    sat.add_clause(clause)

is_solvable = sat.solve()
print(is_solvable)
if is_solvable:
    sat_solution = np.array(sat.get_model())
    dice_solution = sat_to_dice(d, dice_names, sat_solution)
    print(dice_solution)

# =============================================================================
# Four dice sets
# =============================================================================
dice_names = ["A", "B", "C", "D"]
m = len(dice_names)

dice_pairs = list(permutations(dice_names, 2))
n = len(dice_pairs)

d = 4  # 6
adj_score = 10  # 22
acr_score = 8  # 18

temp = [adj_score, acr_score, d ** 2 - adj_score]
S = [[temp[(j - i) % (m - 1)] for j in range(m - 1)] for i in range(m)]
scores = {p: s for p, s in zip(dice_pairs, sum(S, []))}

# ----------------------------------------------------------------------------

clauses = build_clauses(d, dice_names, scores)

sat = Minisat22()
for clause in clauses:
    sat.add_clause(clause)

is_solvable = sat.solve()
print(is_solvable)
if is_solvable:
    sat_solution = np.array(sat.get_model())
    dice_solution = sat_to_dice(d, dice_names, sat_solution)
    print(dice_solution)

# =============================================================================
# Five dice sets
# =============================================================================
dice_names = ["A", "B", "C", "D", "E"]
m = len(dice_names)

dice_pairs = list(permutations(dice_names, 2))
n = len(dice_pairs)

# d = 3
# adj_score = 5
# acr_score = 4
d = 6
adj_score = 24
acr_score = 16

temp = [adj_score, d ** 2 - acr_score, acr_score, d ** 2 - adj_score]
S = [[temp[(j - i) % (m - 1)] for j in range(m - 1)] for i in range(m)]
scores = {p: s for p, s in zip(dice_pairs, sum(S, []))}

# ----------------------------------------------------------------------------

clauses = build_clauses(d, dice_names, scores)

sat = Minisat22()
for clause in clauses:
    sat.add_clause(clause)

is_solvable = sat.solve()
print(is_solvable)
if is_solvable:
    sat_solution = np.array(sat.get_model())
    dice_solution = sat_to_dice(d, dice_names, sat_solution)
    print(dice_solution)

# ============================================================================
# Code to find all solutions via SAT
# ============================================================================

# counter = 0
# is_solvable = m.solve()
# print(is_solvable)
# while is_solvable:
#     counter += 1
#     res = m.get_model()
#     print(counter, res[: (8 * d ** 2)])
#     elim = [-1 * r for r in res[: (8 * d ** 2)]]
#     m.add_clause(elim)
#     is_solvable = m.solve()
