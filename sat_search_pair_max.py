import numpy as np
from itertools import permutations, product
import pysat
from pysat.solvers import Minisat22
from pysat.pb import PBEnc
from utils import compare_dice, collapse_values, recover_values
from utils import sat_to_dice, verify_solution

from clauses import build_cardinality_clauses, build_converse_clauses
from clauses import build_sorting_clauses, build_transitivity_clauses, build_symmetry_clauses

def compare_doubled_dice(first, second):
    hits = 0
    for i,x in enumerate(first):
        for j,y in enumerate(second):
            if y < x:
                hits += (2*i + 1)*(2*j + 1)
    return hits


# ============================================================================
# Three-dice sets with max-pool reversing
# ============================================================================
dice_names = ["A", "B", "C"]
doubled_dice_names = [die_name * 2 for die_name in dice_names]
m = len(dice_names)

dice_pairs = list(permutations(dice_names, 2))
doubled_dice_pairs = list(permutations(doubled_dice_names, 2))
n = len(dice_pairs)

d = 6

singled_score = 21
# doubled_score = 16

temp = [singled_score, d ** 2 - singled_score]
S = [[temp[(j - i) % (m - 1)] for j in range(m - 1)] for i in range(m)]
scores = {p: s for p, s in zip(dice_pairs, sum(S, []))}

# temp = {
#     ('AA', 'BB'): 49,
#     ('AA', 'CC'): 81 - 33,
#     ('BB', 'AA'): 81 - 49,
#     ('BB', 'CC'): 57,
#     ('CC', 'AA'): 33,
#     ('CC', 'BB'): 81 - 57,
# }
# scores.update(temp)

# temp = [doubled_score, d ** 2 - doubled_score]
# S = [[temp[(j - i) % (m - 1)] for j in range(m - 1)] for i in range(m)]
# scores = {p: s for p, s in zip(doubled_dice_pairs, sum(S, []))}

dice_pairs = list(permutations(dice_names, 2))
n = len(dice_pairs)
f = {x: ["%s%i" % (x, i) for i in range(1, d + 1)] for x in dice_names}
df = {x: list(product(f[x], repeat=2)) for x in dice_names}
var_lists = {(x, y): list(product(df[x], df[y])) for (x, y) in dice_pairs}

variables = sum(var_lists.values(), [])
var_dict = dict((v, k) for k, v in enumerate(variables, 1))

singleton_var_lists = {}
for x,y in permutations(dice_names, 2):
    temp = []
    for i in range(d):
        for j in range(d):
            temp.append((f[x][i], f[y][j]))
    singleton_var_lists[(x,y)] = temp

singleton_var_dict = {}
for x,y in permutations(dice_names, 2):
    for i in range(d):
        for j in range(d):
            key = (f[x][i], f[y][j])
            doubled_key = ((f[x][i],f[x][i]), (f[y][j],f[y][j]))
            singleton_var_dict[key] = var_dict[doubled_key]


vpool = pysat.formula.IDPool(start_from=n * (d ** 4) + 1)

clauses = []
clauses += build_cardinality_clauses(d, singleton_var_dict, singleton_var_lists, scores, vpool)
clauses += build_converse_clauses(d, singleton_var_dict, dice_names, vpool)
clauses += build_sorting_clauses(d, singleton_var_dict, dice_names)
clauses += build_transitivity_clauses(d, singleton_var_dict, dice_names)
clauses += build_symmetry_clauses(d, singleton_var_dict, dice_names)

# Need to write the function that builds doubling clauses!








sat = Minisat22()
for clause in clauses:
    sat.add_clause(clause)

is_solvable = sat.solve()
print(is_solvable)

sat_solution = np.array(sat.get_model())


temp = sum([list(product(f[x], f[y])) for (x, y) in dice_pairs], [])
temp2 = [sat_solution[singleton_var_dict[t]-1] for t in temp]
singled_sat_solution = np.array([np.sign(t) * i for i,t in enumerate(temp2, 1)])

dice_solution = sat_to_dice(d, dice_names, singled_sat_solution)
