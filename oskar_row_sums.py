import numpy as np

from itertools import permutations, product, combinations
from pysat.pb import PBEnc

from utils import verify_solution, sat_search, build_sat, sat_to_dice


m = 19
dice_names = ["D%i" % i for i in range(m)]

d = 5
cardinality_clauses = False

dice_pairs = list(permutations(dice_names, 2))
n = len(dice_pairs)
faces = {x: ["%s%i" % (x, i) for i in range(1, d + 1)] for x in dice_names}
var_lists = {(x, y): list(product(faces[x], faces[y])) for (x, y) in dice_pairs}
variables = sum(var_lists.values(), [])
var_dict = dict((v, k) for k, v in enumerate(variables, 1))

assumptions = []

# ----------------------------------------------------------------------------

score = d ** 2 // 2 + 1
mask_index = sorted([x for x in set(np.arange(1, m) ** 2 % m)])
mask = [1 if (i + 1) in mask_index else 0 for i in range(m - 1)]
temp = [score if mask[i] else d ** 2 - score for i in range(m - 1)]
S = [[temp[(j - i) % (m - 1)] for j in range(m - 1)] for i in range(m)]
scores = {p: s for p, s in zip(dice_pairs, sum(S, [])) if s == score}

# ----------------------------------------------------------------------------

sat = build_sat(d,
    dice_names,
    scores,
    cardinality_clauses=cardinality_clauses,
    symmetry_clauses=True,
)

row_sum_target = ((m*d * (m*d - 1) // 2) // m) - ((d * (d-1)) // 2)
for x in dice_names:
    row_vars = sum([var_lists[x,y] for y in dice_names if x != y], [])
    row_lits = [var_dict[v] for v in row_vars]
    sat.add_atmost(row_lits, row_sum_target)
    conv_lits = [-l for l in row_lits]
    sat.add_atmost(conv_lits, (m - 1) * d ** 2 - row_sum_target)

m0 = 3
row_sum_target = ((m0*d * (m0*d - 1) // 2) // m0) - ((d * (d-1)) // 2)
triangles = []
for xs in combinations(dice_names, 3):
    if ((xs[0],xs[1]) in scores) and ((xs[1],xs[2]) in scores) and ((xs[2],xs[0]) in scores):
        for y in xs:
            row_vars = sum([var_lists[y,z] for z in xs if y != z], [])
            row_lits = [var_dict[v] for v in row_vars]
            sat.add_atmost(row_lits, row_sum_target)
            conv_lits = [-l for l in row_lits]
            sat.add_atmost(conv_lits, (m0 - 1) * d ** 2 - row_sum_target)

# ----------------------------------------------------------------------------

%time is_solvable = sat.solve()
if is_solvable:
    sat_solution = np.array(sat.get_model())
    dice_solution = sat_to_dice(d, dice_names, sat_solution, compress=False)
else:
    sat_solution = None
    dice_solution = None
