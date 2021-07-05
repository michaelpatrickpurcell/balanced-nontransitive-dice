import numpy as np
from scipy.special import factorial

from itertools import combinations, permutations, product

import pysat
from pysat.pb import PBEnc
from pysat.solvers import Minisat22, Glucose4, Minicard
from pysat.solvers import *
from clauses import build_converse_clauses, build_sorting_clauses
from clauses import build_transitivity_clauses, build_symmetry_clauses
from clauses import build_cardinality_clauses, build_cardinality_lits

from utils import sat_to_dice

def verify_max_doubling_solution(scores_2v2, dice_solution):
    counts = dict()
    for key in scores_2v2.keys():
        ((kw, kx), (ky, kz)) = key
        w = dice_solution[kw]
        x = dice_solution[kx]
        y = dice_solution[ky]
        z = dice_solution[kz]
        counts[key] = 0
        for i,j,k,l in product(w,x,y,z):
            if max(i,j) > max(k,l):
                counts[key] += 1
        print(key, counts[key], "(%i)" % scores_2v2[key])
    return counts


# ============================================================================

def build_max_doubling_clauses(var_dict_1v1, var_dict_2v2):
    max_doubling_clauses = []
    for kv,v in var_dict_2v2.items():
        kw,kx,ky,kz = kv[0][0], kv[0][1], kv[1][0], kv[1][1]
        wy = var_dict_1v1[(kw, ky)]
        wz = var_dict_1v1[(kw, kz)]
        xy = var_dict_1v1[(kx, ky)]
        xz = var_dict_1v1[(kx, kz)]
        # Need to encode (if ((wy and wz) or (xy and xz)) then v)
        # Equivalent to ((if (wy and wz) then v) and (if (xy and xz) then v))
        max_doubling_clauses.extend([[-wy, -wz, v], [-xy, -xz, v]])
        # Need to encode (if v then ((wy and wz) or (xy and xz)))
        # Equivalent to ((if v then (wy and wz)) or (if v then (xy and xz)))
        max_doubling_clauses.extend([[-v, wy, xy], [-v, wy, xz], [-v, wz, xy], [-v, wz, xz]])
    return max_doubling_clauses

# ============================================================================

m = 5
letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
dice_names = [letters[i] for i in range(m)]

d = 6
faces_1 = {x: ["%s%i" % (x, i) for i in range(1, d + 1)] for x in dice_names}
faces_2 = {(x,y): list(product(faces_1[x], faces_1[y])) for x,y in combinations(dice_names, 2)}

dice_perms_1v1 = list(permutations(dice_names, 2))
scores_1v1 = {x: d ** 2 // 2 for x in dice_perms_1v1}

dice_perms_2v2 = []
for x in combinations(dice_names, 2):
    remaining_dice_names = [z for z in dice_names if z not in x]
    dice_perms_2v2.extend([(x,y) for y in combinations(remaining_dice_names, 2)])

offset = 19
scores_2v2 = {
 (('A', 'B'), ('C', 'D')): d ** 4 // 2 + offset,
 (('A', 'C'), ('B', 'D')): d ** 4 // 2 + offset,
 (('A', 'C'), ('D', 'E')): d ** 4 // 2 + offset,
 (('A', 'D'), ('B', 'C')): d ** 4 // 2 + offset,
 (('A', 'D'), ('B', 'E')): d ** 4 // 2 + offset,
 (('A', 'E'), ('B', 'C')): d ** 4 // 2 + offset,
 (('B', 'C'), ('D', 'E')): d ** 4 // 2 + offset,
 (('B', 'D'), ('A', 'E')): d ** 4 // 2 + offset,
 (('B', 'D'), ('C', 'E')): d ** 4 // 2 + offset,
 (('B', 'E'), ('A', 'C')): d ** 4 // 2 + offset,
 (('B', 'E'), ('C', 'D')): d ** 4 // 2 + offset,
 (('C', 'D'), ('A', 'E')): d ** 4 // 2 + offset,
 (('C', 'E'), ('A', 'B')): d ** 4 // 2 + offset,
 (('C', 'E'), ('A', 'D')): d ** 4 // 2 + offset,
 (('D', 'E'), ('A', 'B')): d ** 4 // 2 + offset,
}

# ============================================================================

start_enum = 1

# ----------------------------------------------------------------------------

var_lists_1v1 = {(x, y): list(product(faces_1[x], faces_1[y])) for (x, y) in dice_perms_1v1}
variables_1v1 = sum(var_lists_1v1.values(), [])

var_dict_1v1 = dict((v, k) for k, v in enumerate(variables_1v1, start_enum))
start_enum += len(variables_1v1)

# ----------------------------------------------------------------------------

var_lists_2v2 = {(x, y): list(product(faces_2[x], faces_2[y])) for (x, y) in dice_perms_2v2}
variables_2v2 = sum(var_lists_2v2.values(), [])
var_dict_2v2 = dict((v, k) for k, v in enumerate(variables_2v2, start_enum))
start_enum += len(variables_2v2)

# ----------------------------------------------------------------------------
# Set up a variable poll that will be used for all cardinality or
# threshold constraint clauses

vpool = pysat.formula.IDPool(start_from=start_enum)

# ----------------------------------------------------------------------------

print("Building Clauses")
# Build clauses for one-die comparisons
clauses = []
clauses += build_converse_clauses(d, var_dict_1v1, dice_names)
clauses += build_sorting_clauses(d, var_dict_1v1, faces_1)
clauses += build_transitivity_clauses(d, var_dict_1v1, faces_1)
clauses += build_symmetry_clauses(d, var_dict_1v1, dice_names)
# clauses += build_cardinality_clauses(d, var_dict_1v1, var_lists_1v1, scores_1v1, vpool)

# ----------------------------------------------------------------------------

clauses += build_max_doubling_clauses(var_dict_1v1, var_dict_2v2)
# clauses += build_cardinality_clauses(d**2, var_dict_2v2, var_lists_2v2, scores_2v2, vpool, pb=PBEnc.atleast)

# temp_dict = {x: var_dict_2v2[x] for x in scores_2v2}
temp_lists = {x: var_lists_2v2[x] for x in scores_2v2}
cardinality_lits = build_cardinality_lits(d**2, var_dict_2v2, temp_lists)

# ----------------------------------------------------------------------------

print("Starting SAT solver.")
sat = Minicard()

for clause in clauses:
    sat.add_clause(clause)

for x, lits in cardinality_lits.items():
    conv_lits = [-l for l in lits]
    sat.add_atmost(conv_lits, d ** 4 - scores_2v2[x])

%time is_solvable = sat.solve()
if is_solvable:
    model = np.array(sat.get_model())
    sat_solution = np.array(sat.get_model())
    dice_solution = sat_to_dice(d, dice_names, sat_solution, compress=False)
else:
    dice_solution = None

print(dice_solution)
