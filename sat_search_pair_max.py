import numpy as np
from itertools import permutations, product
import pysat
from pysat.solvers import Minisat22

from utils import sat_to_dice
from utils import verify_doubling_solution

from clauses import build_max_min_clauses

# ============================================================================


def sat_search_max_min(d, dice_names, scores, max_scores, min_scores):
    clauses = build_max_min_clauses(d, dice_names, scores, max_scores, min_scores)

    sat = Minisat22()
    for clause in clauses:
        sat.add_clause(clause)

    is_solvable = sat.solve()
    if is_solvable:
        model = np.array(sat.get_model())
        sat_solution = np.array(sat.get_model())
        dice_solution = sat_to_dice(d, dice_names, sat_solution, compress=False)
    else:
        dice_solution = None

    return dice_solution


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
