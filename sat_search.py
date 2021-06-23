# SAT strategy
import numpy as np
import pysat
from pysat.solvers import Minisat22
from pysat.pb import PBEnc
from itertools import product, permutations

from utils import compare_dice, collapse_values, recover_values
from utils import sat_to_dice, verify_solution
from clauses import build_clauses

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
