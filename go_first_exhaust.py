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