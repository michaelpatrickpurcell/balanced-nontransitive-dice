import numpy as np
from itertools import permutations, product

from utils import verify_doubling_solution, sat_search_max_min

# ============================================================================
# Three-dice sets with max-pool/min-pool reversing
# ============================================================================
dice_names = ["A", "B", "C"]
m = len(dice_names)

dice_pairs = list(permutations(dice_names, 2))
n = len(dice_pairs)

d = 6

# ----------------------------------------------------------------------------

score_pairs = [("A", "B"), ("B", "C"), ("C", "A")]
scores = {sp: d ** 2 // 2 for sp in score_pairs}
max_scores = {sp: d ** 4 // 2 + 1 for sp in score_pairs}
min_scores = {sp: d ** 4 // 2 - 1 for sp in score_pairs}

# ----------------------------------------------------------------------------

dice_solution = sat_search_max_min(d, dice_names, scores, max_scores, min_scores)
print(dice_solution)
if dice_solution is not None:
    verify_doubling_solution(scores, max_scores, min_scores, dice_solution)

# Here's one solution that works for six-sided dice
# dice_solution = {
#     "A": [0, 6, 7, 8, 14, 16],
#     "B": [3, 4, 5, 10, 12, 17],
#     "C": [1, 2, 9, 11, 13, 15],
# }
