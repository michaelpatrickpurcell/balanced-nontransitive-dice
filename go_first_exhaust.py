import numpy as np
from scipy.special import factorial

from itertools import combinations, permutations, product

from tqdm import tqdm

import pysat
from pysat.pb import PBEnc
from pysat.solvers import Minisat22, Glucose4, Gluecard4
from pysat.solvers import *
from clauses import build_converse_clauses, build_sorting_clauses
from clauses import build_transitivity_clauses, build_symmetry_clauses
from clauses import build_cardinality_clauses, build_cardinality_lits

from utils import sat_to_dice, sat_to_constraints, recover_values
from utils import verify_go_first

# ============================================================================
# Find all possible pair compairson patterns
m = 2
letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
dice_names = [letters[i] for i in range(m)]

d = 6
faces = {x: ["%s%i" % (x, i) for i in range(1, d + 1)] for x in dice_names}

scores = {x: d ** 2 // 2 for x in permutations(dice_names, 2)}

# ============================================================================

dice_pairs = list(permutations(dice_names, 2))
n2 = len(dice_pairs)
start_enum = 1

var_lists_2 = {(x, y): list(product(faces[x], faces[y])) for (x, y) in dice_pairs}
variables_2 = sum(var_lists_2.values(), [])

var_dict_2 = dict((v, k) for k, v in enumerate(variables_2, start_enum))
start_enum += len(variables_2)

print("Building Clauses")
# Build clauses for one-die comparisons
clauses = []
clauses += build_converse_clauses(d, var_dict_2, dice_names)
clauses += build_sorting_clauses(d, var_dict_2, faces)
clauses += build_transitivity_clauses(d, var_dict_2, faces)
# clauses += build_symmetry_clauses(d, var_dict_2, dice_names)
# clauses += build_structure_clauses(d, var_dict_2, var_lists_2, pairwise_scores)
# clauses += build_cardinality_clauses(d, var_dict_2, var_lists_2, scores, vpool)

# ----------------------------------------------------------------------------

cardinality_lits_2 = build_cardinality_lits(d, var_dict_2, var_lists_2)

# ----------------------------------------------------------------------------

print("Starting SAT solver.")
sat = Minicard()
# sat = Glucose4()

for clause in clauses:
    sat.add_clause(clause)

for x, lits in cardinality_lits_2.items():
    sat.add_atmost(lits, scores[x])
    conv_lits = [-l for l in lits]
    sat.add_atmost(conv_lits, d ** 2 - scores[x])

sat_solutions = []
TRIALS = 2**16
is_solvable = True
for i in tqdm(range(TRIALS)):
    is_solvable = sat.solve()  # assumptions=assumptions)
    if is_solvable:
        sat_solution = np.array(sat.get_model())
        sat_solutions.append(sat_solution)
        dice_solution = sat_to_dice(d, dice_names, sat_solution, compress=False)
        elim = [-1 * int(r) for r in sat_solution[: (m * (m - 1) * d ** 2)]]
        sat.add_clause(elim)
    else:
        break

print("Recovering constraint arrays")
constraints = []
for sat_solution in tqdm(sat_solutions):
    temp = sat_to_constraints(d, dice_names, sat_solution)
    constraints.append(np.matrix(temp[("A", "B")], dtype=np.int))

# ============================================================================

print("Creating master_dict")
master_dict = {i:x for i,x in enumerate(constraints)}

# with open('go_first_d12_master_dict.pickle', 'wb') as handle:
#     pickle.dump(master_dict, handle)

comp_dict = {}
for i,x in tqdm(master_dict.items()):
    comp_dict[i] = []
    for j,y in master_dict.items():
        temp = np.sum(x @ y)
        if temp == d**3 // 6:
            comp_dict[i].append(j)

# with open('go_first_d12_comp_dict.pickle', 'wb') as handle:
#     pickle.dump(comp_dict, handle)


# ============================================================================
# with open('go_first_d12_master_dict.pickle', 'rb') as handle:
#     master_dict = pickle.load(handle)
#
# with open('go_first_d12_comp_dict.pickle', 'rb') as handle:
#     comp_dict = pickle.load(handle)
#
print("Finding Survivors")
survivors = []
for i in tqdm(comp_dict):
    for j in comp_dict[i]:
        for k in comp_dict[j]:
            if i in comp_dict[k]:
                survivors.append((i,j,k))

print("Deduping survivors")
deduped_survivors = []
for i,j,k in tqdm(survivors):
    if np.all(master_dict[i][0,:] == 0) and np.all(master_dict[k][:,0] == 1):
        deduped_survivors.append((i,j,k))

print("Verifying results")
dice_solutions = []
for survivor in tqdm(deduped_survivors):
    temp = {}
    temp[("A", "B")] = np.array(master_dict[survivor[0]]).astype(np.bool)
    temp[("B", "A")] = np.transpose(temp[("A", "B")]) ^ True
    temp[("B", "C")] = np.array(master_dict[survivor[1]]).astype(np.bool)
    temp[("C", "B")] = np.transpose(temp[("B", "C")]) ^ True
    temp[("C", "A")] = np.array(master_dict[survivor[2]]).astype(np.bool)
    temp[("A", "C")] = np.transpose(temp[("C", "A")]) ^ True


    natural_faces = recover_values(d, ["A", "B", "C"], temp)
    dice_dict = {k: v for k, v in zip(["A", "B", "C"], natural_faces)}
    dice_solutions.append(dice_dict)
    verify_go_first(dice_dict)
    print()
    # if np.any(np.array(list(test.values())) != d**3 // 6):
    #     print("Error! Error! Error!")
    # else:
    #     print("Good hit")
    #     print(dice_dict)
