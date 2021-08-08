import numpy as np
from itertools import permutations, product
from pysat.pb import PBEnc

from utils import verify_solution, sat_search, build_sat
from utils import sat_to_dice

m = 19
dice_names = ["D%i" % i for i in range(m)]

d = 9
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

mask = [1 if (i + 1) in mask_index[:8] else 0 for i in range(m - 1)]
temp = [score if mask[i] else d ** 2 - score for i in range(m - 1)]
S = [[temp[(j - i) % (m - 1)] for j in range(m - 1)] for i in range(m)]
scores_1 = {p: s for p, s in zip(dice_pairs, sum(S, [])) if s == score}
sat_1 = build_sat(
    d,
    dice_names,
    scores_1,
    cardinality_clauses=cardinality_clauses,
    symmetry_clauses=False,
    pb=PBEnc.equals,
)

mask = [1 if (i + 1) in mask_index[:8] else 0 for i in range(m - 1)]
temp = [score if mask[i] else d ** 2 - score for i in range(m - 1)]
S = [[temp[(j - i) % (m - 1)] for j in range(m - 1)] for i in range(m)]
scores_2 = {p: s for p, s in zip(dice_pairs, sum(S, [])) if s == score}
sat_2 = build_sat(
    d,
    dice_names,
    scores_2,
    cardinality_clauses=cardinality_clauses,
    symmetry_clauses=False,
    pb=PBEnc.equals,
)

mask = [1 if (i + 1) in mask_index[:9] else 0 for i in range(m - 1)]
temp = [score if mask[i] else d ** 2 - score for i in range(m - 1)]
S = [[temp[(j - i) % (m - 1)] for j in range(m - 1)] for i in range(m)]
scores_3 = {p: s for p, s in zip(dice_pairs, sum(S, [])) if s == score}
sat_3 = build_sat(
    d,
    dice_names,
    scores_3,
    cardinality_clauses=cardinality_clauses,
    symmetry_clauses=False,
    pb=PBEnc.equals,
)

flag = False
dice_solution = None
for i, partial_solution_1 in enumerate(sat_1.enum_models()):
    print("i", i)
    assumptions_1 = []
    for x in scores_1:
        for y in var_lists[x]:
            assumptions_1.append(int(partial_solution_1[var_dict[y] - 1]))
    for j, partial_solution_2 in enumerate(sat_2.enum_models(assumptions_1)):
        print("j", j)
        assumptions_2 = []
        for x in scores_2:
            for y in var_lists[x]:
                assumptions_2.append(int(partial_solution_2[var_dict[y] - 1]))
        is_solvable = sat_3.solve(assumptions_2)
        if is_solvable:
            print("hit")
            sat_solution = np.array(sat_3.get_model())
            dice_solution = sat_to_dice(d, dice_names, sat_solution, compress=False)
            flag = True
            break
    if flag:
        break


print(dice_solution)
if dice_solution is not None:
    verify_solution(scores_3, dice_solution)


#
# is_solvable = sat.solve(assumptions=assumptions)
# if is_solvable:
#     sat_solution = np.array(sat.get_model())
#     dice_solution = sat_to_dice(d, dice_names, sat_solution, compress=False)
# else:
#     sat_solution = None
#     dice_solution = None
#
# assumptions = []
# for x in scores:
#     for y in var_lists[x]:
#         assumptions.append(int(sat_solution[var_dict[y] - 1]))

# ----------------------------------------------------------------------------

# ============================================================================

# m = 19
# dice_names = ["D%i" % i for i in range(m)]
#
# d = 5
# cardinality_clauses = False
#
# dice_pairs = list(permutations(dice_names, 2))
# n = len(dice_pairs)
# faces = {x: ["%s%i" % (x, i) for i in range(1, d + 1)] for x in dice_names}
# var_lists = {(x, y): list(product(faces[x], faces[y])) for (x, y) in dice_pairs}
# variables = sum(var_lists.values(), [])
# var_dict = dict((v, k) for k, v in enumerate(variables, 1))
#
# assumptions = []
#
# # ----------------------------------------------------------------------------
#
# score = d ** 2 // 2 + 1
# mask_index = sorted([x for x in set(np.arange(1, m) ** 2 % m)])
# # mask_index = mask_index[:-1]
# mask = [1 if (i + 1) in mask_index else 0 for i in range(m - 1)]
# temp = [score if mask[i] else d ** 2 - score for i in range(m - 1)]
# S = [[temp[(j - i) % (m - 1)] for j in range(m - 1)] for i in range(m)]
# # scores = {p: s for p, s in zip(dice_pairs, sum(S, []))}
# scores = {p: s for p, s in zip(dice_pairs, sum(S, [])) if s == score}
#
# # ----------------------------------------------------------------------------
#
# sat = build_sat(
#     d,
#     dice_names,
#     scores,
#     cardinality_clauses=cardinality_clauses,
#     symmetry_clauses=True,
# )
#
# full_sat_solutions = sat.enum_models(assumptions=assumptions)
# partial_sat_solutions = set()
# for i, sol in enumerate(full_sat_solutions):
#     if i % 1000 == 0:
#         print(i)
#     partial_sat_solutions.add(tuple(sol))
#
#     # temp = []
#     # for x in scores:
#     #     for y in var_lists[x]:
#     #         temp.append(int(sol[var_dict[y] - 1]))
#     # partial_sat_solutions.add(tuple(temp))
#
#
# # ----------------------------------------------------------------------------
#
# score = d ** 2 // 2 + 1
# mask_index = sorted([x for x in set(np.arange(1, m) ** 2 % m)])
# mask = [1 if (i + 1) in mask_index else 0 for i in range(m - 1)]
# temp = [score if mask[i] else d ** 2 - score for i in range(m - 1)]
# S = [[temp[(j - i) % (m - 1)] for j in range(m - 1)] for i in range(m)]
# # scores = {p: s for p, s in zip(dice_pairs, sum(S, []))}
# scores = {p: s for p, s in zip(dice_pairs, sum(S, [])) if s == score}
#
# # ----------------------------------------------------------------------------
#
# sat = build_sat(
#     d,
#     dice_names,
#     scores,
#     cardinality_clauses=cardinality_clauses,
#     symmetry_clauses=True,
# )
#
# dice_solutions = []
# for assumptions in tqdm(partial_sat_solutions):
#     is_solvable = list(sat.enum_models(assumptions=assumptions))
#     if len(is_solvable) > 0:
#         print(len(is_solvable))
#     #     sat_solution = np.array(sat.get_model())
#     #     dice_solution = sat_to_dice(d, dice_names, sat_solution, compress=False)
#     # dice_solutions.append(dice_solution)
#     # else:
#     #     sat_solution = None
#     #     dice_solution = None
