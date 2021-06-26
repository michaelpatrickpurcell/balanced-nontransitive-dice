# SAT strategy
import numpy as np
from pysat.solvers import Minisat22
from itertools import permutations

from utils import compare_dice, compress_values, recover_values
from utils import sat_to_dice, verify_solution
from clauses import build_clauses

# ============================================================================


def sat_search(d, dice_names, scores):
    clauses = build_clauses(d, dice_names, scores)

    sat = Minisat22()
    for clause in clauses:
        sat.add_clause(clause)

    is_solvable = sat.solve()
    print(is_solvable)
    if is_solvable:
        sat_solution = np.array(sat.get_model())
        dice_solution = sat_to_dice(d, dice_names, sat_solution)
    else:
        dice_solution = None

    return dice_solution


# =============================================================================
# Three-dice sets
# =============================================================================
dice_names = ["A", "B", "C"]
m = len(dice_names)

dice_pairs = list(permutations(dice_names, 2))
n = len(dice_pairs)

d = 13  # 34

n1_score = 104  # 714
temp = [n1_score, d ** 2 - n1_score]
S = [[temp[(j - i) % (m - 1)] for j in range(m - 1)] for i in range(m)]
scores = {p: s for p, s in zip(dice_pairs, sum(S, []))}

dice_solution = sat_search(d, dice_names, scores)
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

dice_solution = sat_search(d, dice_names, scores)
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

dice_solution = sat_search(d, dice_names, scores)
print(dice_solution)

# =============================================================================
# Six dice sets
# =============================================================================
dice_names = ["A", "B", "C", "D", "E", "F"]
m = len(dice_names)

dice_pairs = list(permutations(dice_names, 2))
n = len(dice_pairs)

d = 6

n1_score = 21
n2_score = 15
n3_score = 18
temp = [n1_score, n2_score, n3_score, d ** 2 - n2_score, d ** 2 - n1_score]
S = [[temp[(j - i) % (m - 1)] for j in range(m - 1)] for i in range(m)]
scores = {p: s for p, s in zip(dice_pairs, sum(S, []))}

dice_solution = sat_search(d, dice_names, scores)
print(dice_solution)

# =============================================================================
# Seven dice sets
# =============================================================================
dice_names = ["A", "B", "C", "D", "E", "F", "G"]
m = len(dice_names)

dice_pairs = list(permutations(dice_names, 2))
n = len(dice_pairs)

d = 6

score = 20
mask_index = [1, 2, 4]
mask = [1 if (i + 1) in mask_index else 0 for i in range(m - 1)]
temp = [score if mask[i] else d ** 2 - score for i in range(m - 1)]
S = [[temp[(j - i) % (m - 1)] for j in range(m - 1)] for i in range(m)]
scores = {p: s for p, s in zip(dice_pairs, sum(S, []))}

dice_solution = sat_search(d, dice_names, scores)
print(dice_solution)

# =============================================================================
# Four player Oskar dice variant
# =============================================================================
m = 19
letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
dice_names = [letters[i] for i in range(m)]

dice_pairs = list(permutations(dice_names, 2))
n = len(dice_pairs)

d = 5  # 7

score = 13  # 25
mask_index = [1, 4, 5, 6, 7, 9, 11, 16, 17]
mask = [1 if (i + 1) in mask_index else 0 for i in range(m - 1)]
temp = [score if mask[i] else d ** 2 - score for i in range(m - 1)]
S = [[temp[(j - i) % (m - 1)] for j in range(m - 1)] for i in range(m)]
scores = {p: s for p, s in zip(dice_pairs, sum(S, []))}

dice_solution = sat_search(d, dice_names, scores)
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
