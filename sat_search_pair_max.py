import numpy as np
from itertools import permutations, product
import pysat
from pysat.solvers import Minisat22
from pysat.pb import PBEnc

from utils import compare_dice, compare_doubled_dice
from utils import compress_values, recover_values
from utils import sat_to_constraints, sat_to_dice
from utils import verify_solution, verify_doubling_solution

from clauses import build_cardinality_clauses, build_converse_clauses
from clauses import build_lower_bound_clauses, build_upper_bound_clauses
from clauses import build_sorting_clauses, build_symmetry_clauses
from clauses import build_transitivity_clauses
from clauses import build_max_doubling_clauses, build_min_doubling_clauses
from clauses import build_clauses

# ============================================================================
# Three-dice sets with max-pool/min-pool reversing
# ============================================================================
dice_names = ["A", "B", "C"]
m = len(dice_names)

dice_pairs = list(permutations(dice_names, 2))
n = len(dice_pairs)

d = 6
start_enum = 1

# ----------------------------------------------------------------------------

score_pairs = [("A", "B"), ("B", "C"), ("C", "A")]
scores = {sp: d ** 2 // 2 for sp in score_pairs}
max_scores = {sp: d ** 4 // 2 + 1 for sp in score_pairs}
min_scores = {sp: d ** 4 // 2 - 1 for sp in score_pairs}

# ----------------------------------------------------------------------------

faces_1v1 = {x: ["%s%i" % (x, i) for i in range(1, d + 1)] for x in dice_names}
var_lists_1v1 = {
    (x, y): list(product(faces_1v1[x], faces_1v1[y])) for (x, y) in dice_pairs
}
variables_1v1 = sum(var_lists_1v1.values(), [])

var_dict_1v1 = dict((v, k) for k, v in enumerate(variables_1v1, start_enum))
start_enum += len(variables_1v1)

# ----------------------------------------------------------------------------

faces_2v2 = {x: list(product(faces_1v1[x], repeat=2)) for x in dice_names}
var_lists_2v2 = {
    (x, y): list(product(faces_2v2[x], faces_2v2[y])) for (x, y) in dice_pairs
}
variables_2v2 = sum(var_lists_2v2.values(), [])

var_dict_2v2_max = dict((v, k) for k, v in enumerate(variables_2v2, start_enum))
start_enum += len(variables_2v2)

var_dict_2v2_min = dict((v, k) for k, v in enumerate(variables_2v2, start_enum))
start_enum += len(variables_2v2)

# ----------------------------------------------------------------------------
# Set up a variable poll that will be used for all cardinality or
# threshold constraint clauses
vpool = pysat.formula.IDPool(start_from=start_enum)

# ----------------------------------------------------------------------------
# Build clauses for one-die comparisons
clauses = []
clauses += build_clauses(d, dice_names, scores, vpool)

# ----------------------------------------------------------------------------
# Build clauses for two-dice comparisons with max-pooling
clauses += build_max_doubling_clauses(d, var_dict_1v1, var_dict_2v2_max, dice_names)
clauses += build_lower_bound_clauses(
    d ** 2, var_dict_2v2_max, var_lists_2v2, max_scores, vpool
)

# ----------------------------------------------------------------------------
# Build clauses for two-dice comparisons with min-pooling
clauses += build_min_doubling_clauses(d, var_dict_1v1, var_dict_2v2_min, dice_names)
clauses += build_upper_bound_clauses(
    d ** 2, var_dict_2v2_min, var_lists_2v2, min_scores, vpool
)

# ----------------------------------------------------------------------------

sat = Minisat22()
for clause in clauses:
    sat.add_clause(clause)

is_solvable = sat.solve()
print(is_solvable)

if is_solvable:
    model = np.array(sat.get_model())

    temp = sum([list(product(faces_1v1[x], faces_1v1[y])) for (x, y) in dice_pairs], [])
    var_indices = [model[var_dict_1v1[t] - 1] for t in temp]
    sat_solution = np.array(var_indices)
    dice_solution = sat_to_dice(d, dice_names, sat_solution)
    print(dice_solution)

    verify_doubling_solution(scores, max_scores, min_scores, dice_solution)


# Here's one solution that works for six-sided dice
# dice_solution = {
#     "A": [0, 6, 7, 8, 14, 16],
#     "B": [3, 4, 5, 10, 12, 17],
#     "C": [1, 2, 9, 11, 13, 15],
# }
