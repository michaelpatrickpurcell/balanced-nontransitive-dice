import numpy as np
from scipy.special import factorial

from itertools import permutations, product

from tqdm import tqdm

import pysat
from pysat.pb import PBEnc
from pysat.solvers import Minisat22, Glucose4, Gluecard4, Minicard
from clauses import build_converse_clauses, build_sorting_clauses
from clauses import build_transitivity_clauses, build_symmetry_clauses
from clauses import build_cardinality_clauses, build_cardinality_lits
from clauses import build_clauses

from clauses import build_permutation_clauses, build_winner_clauses
from clauses import build_exclusivity_clauses, build_exclusivity_lits

from utils import sat_to_dice, verify_go_first

# ============================================================================

m = 5
letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
dice_names = [letters[i] for i in range(m)]

d = 6
faces = {x: ["%s%i" % (x, i) for i in range(1, d + 1)] for x in dice_names}

scores_2 = {x: d ** 2 // 2 for x in permutations(dice_names, 2)}

dice_perms1 = [
    ("A", "B", "C"),
    ("B", "C", "D"),
    ("C", "D", "E"),
    ("D", "E", "A"),
    ("E", "A", "B"),
]
dice_perms1 += [
    ("A", "B", "D"),
    ("B", "C", "E"),
    ("C", "D", "A"),
    ("D", "E", "B"),
    ("E", "A", "C"),
]
dice_perms2 = sum([list(permutations(dp))[2::2] for dp in dice_perms1], [])

dice_perms = dice_perms1 + dice_perms2

score1 = d ** 3 // 3 + 6
scores1 = dict()
for x in dice_perms1:
    scores1[x] = score1

score2 = d ** 3 // 3
scores2 = dict()
for x in dice_perms2:
    scores2[x] = score2

# ============================================================================

dice_pairs = list(permutations(dice_names, 2))
n2 = len(dice_pairs)
start_enum = 1

# ----------------------------------------------------------------------------

var_lists_2 = {(x, y): list(product(faces[x], faces[y])) for (x, y) in dice_pairs}
variables_2 = sum(var_lists_2.values(), [])

var_dict_2 = dict((v, k) for k, v in enumerate(variables_2, start_enum))
start_enum += len(variables_2)

# ----------------------------------------------------------------------------

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
# clauses += build_cardinality_clauses(d, var_dict_2, var_lists_2, scores_2, vpool)

# ----------------------------------------------------------------------------

clauses += build_winner_clauses(d, var_dict_2, var_dict_m, dice_names, dice_perms)
clauses += build_cardinality_clauses(
    d, var_dict_m, var_lists_m, scores1, vpool, pb=PBEnc.atleast
)
clauses += build_cardinality_clauses(
    d, var_dict_m, var_lists_m, scores2, vpool, pb=PBEnc.atmost
)

# ----------------------------------------------------------------------------

print("Starting SAT solver.")
sat = Glucose4()

for clause in clauses:
    sat.add_clause(clause)

is_solvable = sat.solve()
if is_solvable:
    model = np.array(sat.get_model())
    sat_solution = np.array(sat.get_model())
    dice_solution = sat_to_dice(d, dice_names, sat_solution, compress=False)
else:
    dice_solution = None


counts = dict()
for xs in dice_perms1:
    for ys in permutations(xs):
        counts[ys] = 0
        dice_triple = [dice_solution[y] for y in ys]
        for face_triple in product(*dice_triple):
            if face_triple[0] == max(face_triple):
                counts[ys] += 1
        print("%s: %s" % (ys, counts[ys]))
    print()


# {'A': array([ 0,  9, 10, 21, 24, 25]),
#  'B': array([ 5, 11, 12, 20, 22, 23]),
#  'C': array([14, 15, 16, 17, 18, 19]),
#  'D': array([ 1,  2,  3,  4, 28, 29]),
#  'E': array([ 6,  7,  8, 13, 26, 27])}
