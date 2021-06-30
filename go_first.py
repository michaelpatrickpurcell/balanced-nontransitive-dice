import numpy as np
from scipy.special import factorial

from itertools import permutations, product

import pysat
from pysat.pb import PBEnc
from pysat.solvers import Minisat22

from clauses import build_converse_clauses, build_sorting_clauses
from clauses import build_transitivity_clauses, build_symmetry_clauses
from clauses import build_cardinality_clauses

from utils import sat_to_dice


def build_permutation_clauses(d, var_dict_2, var_dict_m, dice_names):
    m = len(dice_names)
    faces = {x: ["%s%i" % (x, i) for i in range(1, d + 1)] for x in dice_names}
    permutation_clauses = []
    for xs in permutations(dice_names):
        for iis in product(range(d), repeat=m):
            vs = [var_dict_2[(faces[xs[j]][iis[j]], faces[xs[j+1]][iis[j+1]])] for j in range(m-1)]
            w = var_dict_m[tuple([faces[xs[j]][iis[j]] for j in range(m)])]
            permutation_clauses.append([-v for v in vs] + [w])
            permutation_clauses.extend([[-w, v] for v in vs])
    return permutation_clauses


# def build_winner_clauses(d, var_dict_2, var_dict_m, dice_names):
#     m = len(dice_names)
#     faces = {x: ["%s%i" % (x, i) for i in range(1, d + 1)] for x in dice_names}
#     winner_clauses = []
#     for xs in permutations(dice_names):
#         for iis in product(range(d), repeat=m):
#             vs = [var_dict_2[(faces[xs[0]][iis[0]], faces[xs[j]][iis[j]])] for j in range(1,m)]
#             w = var_dict_m[tuple([faces[xs[j]][iis[j]] for j in range(m)])]
#             permutation_clauses.append([-v for v in vs] + [w])
#             permutation_clauses.extend([[-w, v] for v in vs])
#     return permutation_clauses



def verify_go_first(dice_solution):
    counts = {x: 0 for x in permutations(range(len(dice_solution)))}
    for outcome in product(*dice_solution.values()):
        counts[tuple(np.argsort(outcome))] += 1
    return counts


# ============================================================================
m = 3
letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
dice_names = [letters[i] for i in range(m)]

d = 6
faces = {x: ["%s%i" % (x, i) for i in range(1, d + 1)] for x in dice_names}

pairwise_scores = {x: d**2 // 2 for x in permutations(dice_names, 2)}

score = d**m // factorial(m, exact=True)
scores = {x:score  for x in permutations(dice_names)}

# ============================================================================

dice_pairs = list(permutations(dice_names, 2))
n2 = len(dice_pairs)
start_enum = 1

# ----------------------------------------------------------------------------

var_lists_2 = {
    (x, y): list(product(faces[x], faces[y])) for (x, y) in dice_pairs
}
variables_2 = sum(var_lists_2.values(), [])

var_dict_2 = dict((v, k) for k, v in enumerate(variables_2, start_enum))
start_enum += len(variables_2)

# ----------------------------------------------------------------------------

dice_perms = list(permutations(dice_names, m))
nm = len(dice_perms)

var_lists_m = {xs: list(product(*[faces[x] for x in xs])) for xs in dice_perms}
variables_m = sum(var_lists_m.values(), [])

var_dict_m = dict((v, k) for k, v in enumerate(variables_m, start_enum))
start_enum += len(variables_m)

# ----------------------------------------------------------------------------
# Set up a variable poll that will be used for all cardinality or
# threshold constraint clauses
# if vpool == None:
vpool = pysat.formula.IDPool(start_from=start_enum)

# ----------------------------------------------------------------------------
print("Building Clauses")
# Build clauses for one-die comparisons
clauses = []
clauses += build_converse_clauses(d, var_dict_2, dice_names)
clauses += build_sorting_clauses(d, var_dict_2, faces)
clauses += build_transitivity_clauses(d, var_dict_2, faces)
clauses += build_symmetry_clauses(d, var_dict_2, dice_names)
clauses += build_cardinality_clauses(d, var_dict_2, var_lists_2, pairwise_scores, vpool)

# ----------------------------------------------------------------------------

clauses += build_permutation_clauses(d, var_dict_2, var_dict_m, dice_names)

clauses += build_cardinality_clauses(d, var_dict_m, var_lists_m, scores, vpool)

# ----------------------------------------------------------------------------

print("Starting SAT solver.")
sat = Minisat22()
for clause in clauses:
    sat.add_clause(clause)

%time is_solvable = sat.solve()
if is_solvable:
    model = np.array(sat.get_model())
    sat_solution = np.array(sat.get_model())
    dice_solution = sat_to_dice(d, dice_names, sat_solution, compress=False)
else:
    dice_solution = None

print(dice_solution)
if is_solvable:
    counts = verify_go_first(dice_solution)
    print(counts)
    print(np.all(np.array(list(counts.values())) == score))

# ============================================================================

# score = 12**4 // 24
# go_first_4x12 = {
#     "A": [1, 8, 11, 14, 19, 22, 27, 30, 35, 38, 41, 48],
#     "B": [2, 7, 10, 15, 18, 23, 26, 31, 34, 39, 42, 47],
#     "C": [3, 6, 12, 13, 17, 24, 25, 32, 36, 37, 43, 46],
#     "D": [4, 5, 9,  16, 20, 21, 28, 29, 33, 40, 44, 45]
# }
#
# counts = verify_go_first(go_first_4x12)
# print(counts)
# print(np.all(np.array(list(counts.values())) == score))
