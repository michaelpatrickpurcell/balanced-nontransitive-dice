# SAT strategy
import numpy as np
import pysat
from pysat.solvers import Minisat22
from pysat.pb import PBEnc
from itertools import product, permutations

from utils import compare_dice, collapse_values, recover_values

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

    natural_faces = recover_values(d, dice_names, constraints)
    if compress:
        dice_faces = collapse_values(*natural_faces)
        dice_dict = {k: v for k, v in zip(dice_names, dice_faces)}
    else:
        dice_dict = {k: v for k, v in zip(dice_names, natural_faces)}
    return dice_dict


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
n1_score = 10  # 22
n2_score = d ** 2 // 2  # 18

temp = [n1_score, n2_score, d ** 2 - n1_score]
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

d = 6
n1_score = 24
n2_score = 20

temp = [n1_score, n2_score, d ** 2 - n2_score, d ** 2 - n1_score]
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
    natural_solution = sat_to_dice(d, dice_names, sat_solution, compress=False)
    print(natural_solution)
    dice_solution = sat_to_dice(d, dice_names, sat_solution)
    print(dice_solution)


# =============================================================================
# Seven dice sets
# =============================================================================
dice_names = ["A", "B", "C", "D", "E", "F", "G"]
m = len(dice_names)

dice_pairs = list(permutations(dice_names, 2))
n = len(dice_pairs)

d = 6  # 17
n1_score = 20  # 187
n2_score = 20  # 170
n3_score = 16  # 153
temp = [
    n1_score,
    n2_score,
    n3_score,
    d ** 2 - n3_score,
    d ** 2 - n2_score,
    d ** 2 - n1_score,
]
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
# Four player Oskar dice variant
# =============================================================================
m = 19
letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
dice_names = [letters[i] for i in range(m)]

dice_pairs = list(permutations(dice_names, 2))
n = len(dice_pairs)

d = 5  # 7 # 5
score = 13  # 25  # 13
mask_index = [1, 4, 5, 6, 7, 9, 11, 16, 17]
mask = [1 if (i + 1) in mask_index else 0 for i in range(m - 1)]
temp = [score if mask[i] else d ** 2 - score for i in range(m - 1)]
S = [[temp[(j - i) % (m - 1)] for j in range(m - 1)] for i in range(m)]
scores = {p: s for p, s in zip(dice_pairs, sum(S, []))}

# ----------------------------------------------------------------------------

clauses = build_clauses(d, dice_names, scores)

sat = Minisat22()
for clause in clauses:
    sat.add_clause(clause)

is_solvable = sat.solve()
print(is_solvable, mask_index)
if is_solvable:
    sat_solution = np.array(sat.get_model())
    dice_solution = sat_to_dice(d, dice_names, sat_solution, compress=False)
    # print(dice_solution)

# Here's a solution for 19 five-sided dice with bias = 13/25.
# mask_index = [1, 4, 5, 6, 7, 9, 11, 16, 17]
#
# {'A': [0, 29, 37, 71, 90],
#  'B': [13, 31, 54, 59, 69],
#  'C': [6, 6, 46, 84, 86],
#  'D': [21, 34, 39, 58, 74],
#  'E': [4, 9, 56, 70, 89],
#  'F': [22, 22, 48, 55, 79],
#  'G': [10, 28, 45, 67, 76],
#  'H': [18, 27, 43, 57, 81],
#  'I': [25, 33, 38, 52, 77],
#  'J': [3, 20, 35, 82, 88],
#  'K': [1, 19, 53, 64, 91],
#  'L': [17, 36, 42, 63, 68],
#  'M': [5, 16, 49, 66, 92],
#  'N': [2, 24, 41, 73, 87],
#  'O': [8, 14, 50, 72, 83],
#  'P': [7, 32, 51, 61, 75],
#  'Q': [15, 26, 47, 60, 78],
#  'R': [12, 23, 44, 62, 85],
#  'S': [11, 30, 40, 65, 80]}


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
