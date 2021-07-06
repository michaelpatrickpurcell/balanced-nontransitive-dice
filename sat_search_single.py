# SAT strategy
import numpy as np
from itertools import permutations

from utils import verify_solution, sat_search

# =============================================================================
# Three-dice sets
# =============================================================================
dice_names = ["A", "B", "C"]
m = len(dice_names)

dice_pairs = list(permutations(dice_names, 2))
n = len(dice_pairs)

d = 34  # 13

# ----------------------------------------------------------------------------

n1_score = 714  # 104
temp = [n1_score, d ** 2 - n1_score]
S = [[temp[(j - i) % (m - 1)] for j in range(m - 1)] for i in range(m)]
scores = {p: s for p, s in zip(dice_pairs, sum(S, []))}

# ----------------------------------------------------------------------------

dice_solution = sat_search(d, dice_names, scores)
print(dice_solution)
if dice_solution is not None:
    verify_solution(scores, dice_solution)

# =============================================================================
# Four dice sets
# =============================================================================
dice_names = ["A", "B", "C", "D"]
m = len(dice_names)

dice_pairs = list(permutations(dice_names, 2))
n = len(dice_pairs)

d = 4  # 6

# ----------------------------------------------------------------------------

n1_score = 10  # 22
n2_score = d ** 2 // 2  # 18
temp = [n1_score, n2_score, d ** 2 - n1_score]
S = [[temp[(j - i) % (m - 1)] for j in range(m - 1)] for i in range(m)]
scores = {p: s for p, s in zip(dice_pairs, sum(S, []))}

# ----------------------------------------------------------------------------

dice_solution = sat_search(d, dice_names, scores)
print(dice_solution)
if dice_solution is not None:
    verify_solution(scores, dice_solution)

# =============================================================================
# Five dice sets
# =============================================================================
dice_names = ["A", "B", "C", "D", "E"]
m = len(dice_names)

dice_pairs = list(permutations(dice_names, 2))
n = len(dice_pairs)

d = 6

# ----------------------------------------------------------------------------

n1_score = 24
n2_score = 20
temp = [n1_score, n2_score, d ** 2 - n2_score, d ** 2 - n1_score]
S = [[temp[(j - i) % (m - 1)] for j in range(m - 1)] for i in range(m)]
scores = {p: s for p, s in zip(dice_pairs, sum(S, []))}

# ----------------------------------------------------------------------------

dice_solution = sat_search(d, dice_names, scores)
print(dice_solution)
if dice_solution is not None:
    verify_solution(scores, dice_solution)

# =============================================================================
# Six dice sets
# =============================================================================
dice_names = ["A", "B", "C", "D", "E", "F"]
m = len(dice_names)

dice_pairs = list(permutations(dice_names, 2))
n = len(dice_pairs)

d = 6

# ----------------------------------------------------------------------------

n1_score = 21
n2_score = 15
n3_score = 18
temp = [n1_score, n2_score, n3_score, d ** 2 - n2_score, d ** 2 - n1_score]
S = [[temp[(j - i) % (m - 1)] for j in range(m - 1)] for i in range(m)]
scores = {p: s for p, s in zip(dice_pairs, sum(S, []))}

# ----------------------------------------------------------------------------

dice_solution = sat_search(d, dice_names, scores)
print(dice_solution)
if dice_solution is not None:
    verify_solution(scores, dice_solution)

# =============================================================================
# Seven dice sets
# =============================================================================
dice_names = ["A", "B", "C", "D", "E", "F", "G"]
m = len(dice_names)

dice_pairs = list(permutations(dice_names, 2))
n = len(dice_pairs)

d = 6

# ----------------------------------------------------------------------------

score = 20
mask_index = [1, 2, 4]
mask = [1 if (i + 1) in mask_index else 0 for i in range(m - 1)]
temp = [score if mask[i] else d ** 2 - score for i in range(m - 1)]
S = [[temp[(j - i) % (m - 1)] for j in range(m - 1)] for i in range(m)]
scores = {p: s for p, s in zip(dice_pairs, sum(S, []))}

# ----------------------------------------------------------------------------

dice_solution = sat_search(d, dice_names, scores)
print(dice_solution)
if dice_solution is not None:
    verify_solution(scores, dice_solution)

# =============================================================================
# Four player Oskar dice variant
# =============================================================================
m = 19
letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
dice_names = [letters[i] for i in range(m)]

dice_pairs = list(permutations(dice_names, 2))
n = len(dice_pairs)

d = 5

# ----------------------------------------------------------------------------

score = 13
mask_index = [1, 4, 5, 6, 7, 9, 11, 16, 17]
mask = [1 if (i + 1) in mask_index else 0 for i in range(m - 1)]
temp = [score if mask[i] else d ** 2 - score for i in range(m - 1)]
S = [[temp[(j - i) % (m - 1)] for j in range(m - 1)] for i in range(m)]
scores = {p: s for p, s in zip(dice_pairs, sum(S, []))}

# ----------------------------------------------------------------------------

dice_solution = sat_search(d, dice_names, scores)
print(dice_solution)
if dice_solution is not None:
    verify_solution(scores, dice_solution)

# =============================================================================
# Five player Oskar dice variant
# =============================================================================
m = 67  # 67
dice_names = ["D%i" % i for i in range(m)]

dice_pairs = list(permutations(dice_names, 2))
n = len(dice_pairs)

d = 5

# ----------------------------------------------------------------------------

score = d ** 2 // 2 + 1
mask_index = sorted([x for x in set(np.arange(1, m) ** 2 % m)])
mask = [1 if (i + 1) in mask_index else 0 for i in range(m - 1)]
temp = [score if mask[i] else d ** 2 - score for i in range(m - 1)]
S = [[temp[(j - i) % (m - 1)] for j in range(m - 1)] for i in range(m)]
scores = {p: s for p, s in zip(dice_pairs, sum(S, []))}

# ----------------------------------------------------------------------------

dice_solution = sat_search(d, dice_names, scores)
print(dice_solution)
if dice_solution is not None:
    verify_solution(scores, dice_solution)
