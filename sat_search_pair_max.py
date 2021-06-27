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
from clauses import build_max_doubling_clauses

# ============================================================================
# Three-dice sets with max-pool reversing
# ============================================================================
dice_names = ["A", "B", "C"]
m = len(dice_names)

dice_pairs = list(permutations(dice_names, 2))
n = len(dice_pairs)

d = 6

singled_score = d ** 2 // 2 + 1  # 5

scores = {
    ("A", "B"): singled_score,
    ("B", "C"): singled_score,
    ("C", "A"): singled_score,
}

doubled_score = d ** 4 // 2
doubled_scores = {
    ("A", "B"): doubled_score,
    ("C", "A"): doubled_score,
    ("B", "C"): doubled_score,
}
# doubled_scores = {
#     ("A", "B"): 10160,
#     ("B", "C"): 10224,
#     ("C", "A"): 9840,
# }

dice_pairs = list(permutations(dice_names, 2))
n = len(dice_pairs)
f = {x: ["%s%i" % (x, i) for i in range(1, d + 1)] for x in dice_names}
df = {x: list(product(f[x], repeat=2)) for x in dice_names}
var_lists = {(x, y): list(product(df[x], df[y])) for (x, y) in dice_pairs}

variables = sum(var_lists.values(), [])
var_dict = dict((v, k) for k, v in enumerate(variables, 1))

singleton_var_lists = {}
for x, y in permutations(dice_names, 2):
    temp = []
    for i in range(d):
        for j in range(d):
            temp.append((f[x][i], f[y][j]))
    singleton_var_lists[(x, y)] = temp

singleton_var_dict = {}
for x, y in permutations(dice_names, 2):
    for i in range(d):
        for j in range(d):
            key = (f[x][i], f[y][j])
            doubled_key = ((f[x][i], f[x][i]), (f[y][j], f[y][j]))
            singleton_var_dict[key] = var_dict[doubled_key]


vpool = pysat.formula.IDPool(start_from=n * (d ** 4) + 1)

clauses = []
# clauses += build_cardinality_clauses(d ** 2, var_dict, var_lists, doubled_scores, vpool)
clauses += build_upper_bound_clauses(d ** 2, var_dict, var_lists, doubled_scores, vpool)
clauses += build_max_doubling_clauses(d, var_dict, dice_names)

# clauses += build_cardinality_clauses(
#     d, singleton_var_dict, singleton_var_lists, scores, vpool
# )
clauses += build_lower_bound_clauses(
    d, singleton_var_dict, singleton_var_lists, scores, vpool
)

clauses += build_converse_clauses(d, singleton_var_dict, dice_names)
clauses += build_sorting_clauses(d, singleton_var_dict, f)
clauses += build_transitivity_clauses(d, singleton_var_dict, f)

clauses += build_symmetry_clauses(d, singleton_var_dict, dice_names)

sat = Minisat22()
for clause in clauses:
    sat.add_clause(clause)

is_solvable = sat.solve()
print(is_solvable)

if is_solvable:
    model = np.array(sat.get_model())

    temp = sum([list(product(f[x], f[y])) for (x, y) in dice_pairs], [])
    var_indices = [model[singleton_var_dict[t] - 1] for t in temp]
    sat_solution = np.array(var_indices)
    dice_solution = sat_to_dice(d, dice_names, sat_solution)
    print(dice_solution)

    verify_doubling_solution(scores, doubled_scores, dice_solution)


# Here's one solution that works for six-sided dice
# dice_solution = {
#     "A": [0, 6, 7, 8, 14, 16],
#     "B": [3, 4, 5, 10, 12, 17],
#     "C": [1, 2, 9, 11, 13, 15],
# }
