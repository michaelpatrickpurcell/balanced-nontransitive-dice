import numpy as np
from scipy.special import factorial, comb
from copy import copy
from collections import Counter

import random

from itertools import combinations, permutations, product, combinations_with_replacement

from tqdm import tqdm

import pysat
from pysat.pb import PBEnc
from pysat.solvers import Minisat22, Glucose4, Gluecard4
from pysat.solvers import *
from clauses import build_converse_clauses, build_sorting_clauses
from clauses import build_transitivity_clauses, build_symmetry_clauses
from clauses import build_cardinality_clauses, build_cardinality_lits

from utils import sat_to_dice, sat_to_constraints, dice_to_constraints, recover_values
from utils import verify_go_first, word_to_dice, dice_to_word, permute_letters


def score_perm3s(word, verbose=False):
    dice = word_to_dice(word)
    dice_names = list(dice.keys())
    constraints = dice_to_constraints(dice, dtype=np.float)
    scores = dict()
    for x, y, z in permutations(dice_names, 3):
        score = np.sum(constraints[(x, y)] @ constraints[(y, z)])
        scores[(x, y, z)] = score
        if verbose:
            print(score)
    return scores


def score_perm4s(word, verbose=False):
    dice = word_to_dice(word)
    dice_names = list(dice.keys())
    constraints = dice_to_constraints(dice, dtype=np.float)
    scores = dict()
    for w, x, y, z in permutations(dice_names, 4):
        score = np.sum(constraints[(w, x)] @ constraints[(x, y)] @ constraints[(y, z)])
        scores[(w, x, y, z)] = score
        if verbose:
            print(score)
    return scores


def score_perm5s(word, verbose=False):
    dice = word_to_dice(word)
    dice_names = list(dice.keys())
    constraints = dice_to_constraints(dice, dtype=np.float)
    scores = dict()
    for v, w, x, y, z in permutations(dice_names, 5):
        score = np.sum(
            constraints[(v, w)]
            @ constraints[(w, x)]
            @ constraints[(x, y)]
            @ constraints[(y, z)]
        )
        scores[(v, w, x, y, z)] = score
        if verbose:
            print(score)
    return scores


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
TRIALS = 2 ** 16
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
master_dict = {i: x for i, x in enumerate(constraints)}

# with open('go_first_d12_master_dict.pickle', 'wb') as handle:
#     pickle.dump(master_dict, handle)

comp_dict = {}
for i, x in tqdm(master_dict.items()):
    comp_dict[i] = []
    for j, y in master_dict.items():
        temp = np.sum(x @ y)
        if temp == d ** 3 // 6:
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
                survivors.append((i, j, k))

print("Deduping survivors")
deduped_survivors = []
for i, j, k in tqdm(survivors):
    if np.all(master_dict[i][0, :] == 0) and np.all(master_dict[k][:, 0] == 1):
        deduped_survivors.append((i, j, k))

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


words, segmented_words = zip(*[dice_to_word(x) for x in dice_solutions])

atoms = [["ABC", "ACB"], ["BAC", "BCA"], ["CAB", "CBA"]]
valid_solutions = []
for x, y, z in product(*atoms):
    word = "".join([x, x[::-1], y, y[::-1], z, z[::-1]])
    dice_solution = word_to_dice(word)
    print(dice_solution)
    is_valid = verify_go_first(dice_solution)
    if is_valid:
        valid_solutions.append(dice_solution)


valid_solutions = []
for x, y, z in product(sum(atoms, []), repeat=3):
    word = "".join([x, x[::-1], y, y[::-1], z, z[::-1]])
    dice_solution = word_to_dice(word)
    is_valid = verify_go_first(dice_solution)
    if is_valid:
        valid_solutions.append(dice_solution)


m = 4
n = (m * (m - 1)) // 2
d = 12
temp = list(permutations("ABCD"))
atoms = ["".join(t) for t in temp]
valid_solutions = []
for x, y, z in product(atoms[:6], atoms[18:24], atoms[12:18]):
    word = "".join(
        [x, x[::-1], y, y[::-1], z, z[::-1], z, z[::-1], y, y[::-1], x, x[::-1]]
    )
    dice_solution = word_to_dice(word)
    print(dice_solution)
    is_valid = verify_go_first(dice_solution)
    if is_valid:
        valid_solutions.append(dice_solution)


valid_words = [dice_to_word(x)[0] for x in valid_solutions]
for vw in valid_words:
    counts = {}
    for i in range(len(vw) // 2):
        s = vw[2 * i : 2 * (i + 1)]
        if s in counts:
            counts[s] += 1
        else:
            counts[s] = 1
    print(counts)
    print()


atoms = list(permutations("abcd", 2))
atoms = ["".join(x) for x in atoms]

word = ""
while atoms:
    for x, y in combinations(atoms, 2):
        if set(x + y) == set("abcd"):
            word += x + y + y[::-1] + x[::-1]
            atoms.pop(atoms.index(x))
            atoms.pop(atoms.index(x[::-1]))
            atoms.pop(atoms.index(y))
            atoms.pop(atoms.index(y[::-1]))
            break


def rotate_left(chunk):
    return chunk[1:] + chunk[:1]


def rotate_right(chunk):
    return chunk[-1:] + chunk[:-1]


n = 8
rot1_word = ""
for i in range(len(word) // n):
    rot1_word += rotate_left(word[n * i : n * i + n // 2])
    rot1_word += rotate_right(word[n * i + n // 2 : n * i + n])

rot2_word = ""
for i in range(len(rot1_word) // n):
    rot2_word += rotate_left(rot1_word[n * i : n * i + n // 2])
    rot2_word += rotate_right(rot1_word[n * i + n // 2 : n * i + n])

rot3_word = ""
for i in range(len(rot2_word) // n):
    rot3_word += rotate_left(rot2_word[n * i : n * i + n // 2])
    rot3_word += rotate_right(rot2_word[n * i + n // 2 : n * i + n])


word_segments = [word[i * n : (i + 1) * n] for i in range(len(word) // n)]
rot1_word_segments = [
    rot1_word[i * n : (i + 1) * n] for i in range(len(rot1_word) // n)
]
rot2_word_segments = [
    rot2_word[i * n : (i + 1) * n] for i in range(len(rot2_word) // n)
]
rot3_word_segments = [
    rot3_word[i * n : (i + 1) * n] for i in range(len(rot3_word) // n)
]
interleaved_segments = list(zip(word_segments))  # , rot1_word_segments,
# rot2_word_segments, rot3_word_segments))

big_word = "".join(sum(interleaved_segments, tuple()))

dice = word_to_dice(big_word + big_word[::-1])


atoms = list(permutations("abcdef", 3))
atoms = ["".join(x) for x in atoms]

word = ""
while atoms:
    for x, y in combinations(atoms, 2):
        if set(x + y) == set("abcdef"):
            word += x + y + y[::-1] + x[::-1]
            atoms.pop(atoms.index(x))
            atoms.pop(atoms.index(x[::-1]))
            atoms.pop(atoms.index(y))
            atoms.pop(atoms.index(y[::-1]))
            break


# ============================================================================
# This finds a set of:
#   5d2 that satisfy 2/5 permutation fairness
#   5d30 that satisfy 3/5 permutation fairness
#   5d60 that satisfy 4/5 permutation fairness
dice_names = "abcde"
dice_names = "cbead"
dice_names = "eadcb"
m = len(dice_names)
word = dice_names + dice_names[::-1]
coverage_dict = {perm: set() for perm in permutations(range(len(dice_names)))}
for perm in coverage_dict.keys():
    current_word = permute_letters(word, np.array(perm))
    counts = score_perm5s(current_word)
    nonzero_counts = set([x for x in counts if counts[x] > 0])
    coverage_dict[perm] = nonzero_counts

coverage_counter = {k: 0 for k in permutations(dice_names)}
all_perms = set(permutations(dice_names))
perms_list = []
flag = False
ctr = 0
while not flag:
    ctr += 1
    costs = dict()
    for perm in coverage_dict.keys():
        costs[perm] = sum([coverage_counter[x] for x in coverage_dict[perm]])
    winning_perms = [perm for perm in costs if costs[perm] == min(costs.values())]
    winning_perm = winning_perms[0]
    perms_list.append(winning_perm)
    for x in coverage_dict[winning_perm]:
        coverage_counter[x] += 1
    coverage_set = set(coverage_counter.values())
    print(ctr, coverage_set)
    if len(coverage_set) == 1:
        flag = True


# ============================================================================
# This finds a set of:
#   4d6 that satisfy 3/4 permutation fairness
#   4d12 that satisfy 4/4 permutation fairness
dice_names = "abcd"
m = len(dice_names)
word = dice_names + dice_names[::-1]
coverage_dict = {perm: set() for perm in permutations(range(len(dice_names)))}
for perm in coverage_dict.keys():
    current_word = permute_letters(word, np.array(perm))
    counts = score_perm4s(current_word)
    nonzero_counts = set([x for x in counts if counts[x] > 0])
    coverage_dict[perm] = nonzero_counts


coverage_counter = {k: 0 for k in permutations(dice_names, 4)}
all_perms = set(permutations(dice_names))
perms_list = []
flag = False
while not flag:
    costs = dict()
    for perm in coverage_dict.keys():
        cost = sum([coverage_counter[x] for x in coverage_dict[perm]])
        costs[perm] = cost
    winning_perms = [perm for perm in costs if costs[perm] == min(costs.values())]
    winning_perm = winning_perms[0]
    perms_list.append(winning_perm)
    for x in coverage_dict[winning_perm]:
        coverage_counter[x] += 1
    coverage_set = set(coverage_counter.values())
    print(coverage_set)
    if len(coverage_set) == 1:
        flag = True


# ============================================================================
# This finds a set of:
#   5d2 that satisfy 2/5 permutation fairness
#   5d12 that satisfy 3/5 permutation fairness
#   5d24 that satisfy 4/5 permutation fairness
#   5d2880 that satisfy 5/5 permutation fairness
dice_names = "abcde"
# dice_names = "becad"
m = len(dice_names)
word = dice_names + dice_names[::-1]
coverage_dict = {perm: set() for perm in permutations(range(len(dice_names)))}
for perm in coverage_dict.keys():
    current_word = permute_letters(word, np.array(perm))
    counts = score_perm3s(current_word)
    nonzero_counts = set([x for x in counts if counts[x] > 0])
    coverage_dict[perm] = nonzero_counts


coverage_counter = {k: 0 for k in permutations(dice_names, 3)}
all_perms = set(permutations(dice_names))
perms_list = []
flag = False
while not flag:
    costs = dict()
    for perm in coverage_dict.keys():
        cost = sum([coverage_counter[x] for x in coverage_dict[perm]])
        costs[perm] = cost
    winning_perms = [perm for perm in costs if costs[perm] == min(costs.values())]
    winning_perm = winning_perms[0]
    perms_list.append(winning_perm)
    for x in coverage_dict[winning_perm]:
        coverage_counter[x] += 1
    coverage_set = set(coverage_counter.values())
    if len(coverage_set) == 1:
        flag = True

# ============================================================================

big_word = "".join([permute_letters(word, perm) for perm in perms_list])
big_worddrow = big_word + big_word[::-1]
dice_word = big_worddrow
dice = word_to_dice(dice_word)

# with open('four_of_five_go_first.pickle', 'wb') as handle:
#     pickle.dump(dice, handle, protocol=pickle.HIGHEST_PROTOCOL)

constraints = dice_to_constraints(dice, dtype=np.int)

for x, y in permutations(dice_names, 2):
    score = np.sum(constraints[(x, y)])
    print((x, y), score)
print()

scores = score_perm3s(dice_word)
for k, s in scores.items():
    print(k, s)

scores = score_perm4s(dice_word)
for k, s in scores.items():
    print(k, s)

scores = score_perm5s(dice_word)
for k, s in scores.items():
    print(k, s)

# ----------------------------------------------------------------------------
# This finds a set of:
# 5d2880 that are 5/5 permutation fair

all_perms = list(permutations(range(m)))

m = len(dice)
permuted_words = [
    permute_letters(dice_word, np.array(p)) for p in permutations(range(m))
]
huge_word = "".join(permuted_words)

huge_dice = word_to_dice(huge_word)

constraints = dice_to_constraints(huge_dice, dtype=np.float)

for x, y in permutations(dice_names, 2):
    score = np.sum(constraints[(x, y)])
    print((x, y), score)
print()

scores = score_perm3s(huge_word)
for k, s in scores.items():
    print(k, s)

scores = score_perm4s(huge_word)
for k, s in scores.items():
    print(k, s)

scores = score_perm5s(huge_word, verbose=True)
for k, s in scores.items():
    print(k, s)


# ----------------------------------------------------------------------------
# This finds a set of:
#
dice_names = "abcde"
m = len(dice_names)

all_perms = list(permutations(range(m)))
permuted_scores = [score_perm5s(pw) for pw in permuted_words]

# ============================================================================
word = dice_names + dice_names[::-1]
coverage_dict = {perm: set() for perm in permutations(range(len(dice_names)))}
for perm in coverage_dict.keys():
    current_word = permute_letters(word, np.array(perm))
    counts = score_perm2s(current_word)
    nonzero_counts = set([x for x in counts if counts[x] > 0])
    coverage_dict[perm] = nonzero_counts


# template = Counter([2] * 56 + [1] * 2 + [3] * 2)
# template = Counter([2] * 52 + [1] * 4 + [3] * 4)
# template = Counter([2] * 48 + [1] * 6 + [3] * 6)
template = Counter([2] * 20 + [1] * 40)
targets = list(permutations(dice_names, 2))
perm_triplets = list(combinations(permutations(range(5)), 2))
coverage_counters = {}
for ps in tqdm(perm_triplets):
    coverage_counter = {x: 0 for x in targets}
    for p in ps:
        for x in coverage_dict[p]:
            coverage_counter[x] += 1

    if Counter(list(coverage_counter.values())) == template:
        coverage_counters[ps] = coverage_counter

survivors = {}
for x in tqdm(coverage_counters):
    for y in coverage_counters:
        candidate = copy(coverage_counters[x])
        for t in targets:
            candidate[t] += coverage_counters[y][t]
        coverage_set = set(candidate.values())
        if coverage_set == {4}:
            survivors[x + y] = candidate


big_words = [
    "".join([permute_letters(word, perm) for perm in survivor])
    for survivor in survivors
]
big_worddrows = [big_word + big_word[::-1] for big_word in big_words]
dice_words = big_worddrows
dice = [word_to_dice(dice_word) for dice_word in dice_words]


scores = [score_perm5s(dice_word) for dice_word in tqdm(dice_words)]

counter = Counter(scores[0].values())
# Notice that there are really only 60 equivalent sets of vals_2_hits
unique_scores = []
indices = []
for i, score in enumerate(scores):
    if score not in unique_scores:
        unique_scores.append(score)
        indices.append(i)


dice_words = [dice_words[i] for i in indices]

dice_word = "".join(dice_words)

scores = score_perm4s(dice_word, verbose=True)
for k, s in scores.items():
    print(k, s)

scores = score_perm5s(dice_word)
for k, s in scores.items():
    print(k, s)


for us in combinations_with_replacement(unique_scores, 5):
    counts = {k: sum([u[k] for u in us]) for k in us[0]}
    count_set = set(list(counts.values()))
    if len(count_set) == 1:
        print("Hit!")
        break


all_values = set(scores[0].values())
target = 5 * (24 ** 5 / 120)
value_sets = []
for vs in combinations_with_replacement(all_values, 5):
    if sum(vs) == target:
        value_sets.append(set(vs))


valid_pairs = set()
for i in tqdm(range(len(dice_words))):
    for j in tqdm(range(len(dice_words))):
        for x in scores[j]:
            vp = set([scores[i][x], scores[j][x]])
            is_valid = False
            for vs in value_sets:
                if vp.issubset(vs):
                    is_valid = True
                    break
            if not is_valid:
                break
        if is_valid:
            valid_pairs.add((i, j))

valid_triplets = set()
for i in tqdm(valid_pairs):
    for j in range(len(dice_words)):
        for x in scores[j]:
            vp = set([scores[i[0]][x], scores[i[1]][x], scores[j][x]])
            is_valid = False
            for vs in value_sets:
                if vp.issubset(vs):
                    is_valid = True
                    break
            if not is_valid:
                break
        if is_valid:
            valid_triplets.add((i[0], i[1], j))

pruned_valid_triplets = set()
for x in tqdm(valid_triplets):
    if (x[0], x[2], x[1]) not in pruned_valid_triplets:
        pruned_valid_triplets.add(x)


valid_quads = set()
for i in tqdm(pruned_valid_triplets):
    for j in range(len(scores)):
        for x in scores[j]:
            vp = set([scores[i[0]][x], scores[i[1]][x], scores[i[2]][x], scores[j][x]])
            is_valid = False
            for vs in value_sets:
                if vp.issubset(vs):
                    is_valid = True
                    break
            if not is_valid:
                break
        if is_valid:
            valid_quads.add((i[0], i[1], i[2], j))


valid_quints = set()
for i in tqdm(valid_quads):
    for j in range(len(scores)):
        for x in scores[j]:
            vp = set(
                [
                    scores[i[0]][x],
                    scores[i[1]][x],
                    scores[i[2]][x],
                    scores[i[3]][x],
                    scores[j][x],
                ]
            )
            is_valid = False
            for vs in value_sets:
                if vp.issubset(vs):
                    is_valid = True
                    break
            if not is_valid:
                break
        if is_valid:
            valid_quints.add((i[0], i[1], i[2], i[3], j))


for qs in valid_quints:
    check = [sum([scores[q][x] for q in qs]) for x in scores[0]]
    if len(set(check)) < 30:
        print(len(set(check)))
