import numpy as np
from scipy.special import factorial

from itertools import permutations, product

import pysat
from pysat.pb import PBEnc
from pysat.solvers import Minisat22, Glucose4, Lingeling, Mergesat3
from pysat.solvers import *
from clauses import build_converse_clauses, build_sorting_clauses
from clauses import build_transitivity_clauses, build_symmetry_clauses
from clauses import build_cardinality_clauses

from utils import sat_to_dice


def build_permutation_clauses(d, var_dict_2, var_dict_m, dice_names):
    m = len(dice_names)
    faces = {x: ["%s%i" % (x, i) for i in range(1, d + 1)] for x in dice_names}
    permutation_clauses = []
    for xs in permutations(dice_names):
        for iis in product(range(d), repeat=m):
            z = list(zip(xs, iis))
            vs = [var_dict_2[(faces[x][i], faces[y][j])] for ((x,i), (y,j)) in zip(z, z[1:])]
            w = var_dict_m[tuple([faces[y][j] for y,j in z])]
            permutation_clauses.append([-v for v in vs] + [w])
            permutation_clauses.extend([[-w, v] for v in vs])
    return permutation_clauses


def build_winner_clauses(d, var_dict_2, var_dict_m, dice_names, dice_perms):
    m = len(dice_names)
    faces = {x: ["%s%i" % (x, i) for i in range(1, d + 1)] for x in dice_names}
    permutation_clauses = []
    for xs in dice_perms:
        for iis in product(range(d), repeat=m):
            x,i = xs[0], iis[0]
            z = list(zip(xs[1:], iis[1:]))
            vs = [var_dict_2[(faces[x][i], faces[y][j])] for y,j in z]
            w = var_dict_m[tuple([faces[x][i]] + [faces[y][j] for y,j in z])]
            permutation_clauses.append([-v for v in vs] + [w])
            permutation_clauses.extend([[-w, v] for v in vs])
    return permutation_clauses


def build_exclusivity_clauses(d, var_dict_m, dice_names, vpool):
    m = len(dice_names)
    faces = {x: ["%s%i" % (x, i) for i in range(1, d + 1)] for x in dice_names}
    exclusivity_clauses = []
    for x in product(range(d), repeat=m):
        column = [faces[dice_names[i]][x[i]] for i in range(m)]
        lits = [var_dict_m[tuple(key)] for key in permutations(column)]
        cnf = PBEnc.equals(lits=lits, bound=1, vpool=vpool, encoding=0)
        exclusivity_clauses += cnf.clauses
    return exclusivity_clauses


def verify_go_first(dice_solution):
    counts = {x: 0 for x in permutations(range(len(dice_solution)))}
    for outcome in product(*dice_solution.values()):
        counts[tuple(np.argsort(outcome))] += 1
    return counts


# ============================================================================
m = 4
letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
dice_names = [letters[i] for i in range(m)]

d = 6
faces = {x: ["%s%i" % (x, i) for i in range(1, d + 1)] for x in dice_names}

pairwise_scores = {x: d ** 2 // 2 for x in permutations(dice_names, 2)}

score = d ** m // factorial(m, exact=True)
scores = {x: score for x in permutations(dice_names)}

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

dice_perms = list(permutations(dice_names, m))
nm = len(dice_perms)

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
clauses += build_cardinality_clauses(d, var_dict_2, var_lists_2, pairwise_scores, vpool)

# ----------------------------------------------------------------------------

clauses += build_permutation_clauses(d, var_dict_2, var_dict_m, dice_names)
clauses += build_exclusivity_clauses(d, var_dict_m, dice_names, vpool)
clauses += build_cardinality_clauses(d, var_dict_m, var_lists_m, scores, vpool)

# ----------------------------------------------------------------------------

print("Starting SAT solver.")
# sat = Minisat22() # Wall time: 8.63 s
# sat = Glucose3() # Wall time: 174 ms
sat = Glucose4() # Wall time: 207 ms
# sat = Lingeling() # Wall time: 2.35 s
# sat = Mergesat3() # Wall time: 21.8 s
# sat = Cadical() # Wall time: 1min 16s
# sat = Gluecard3() # Wall time: 185 ms
# sat = Gluecard4() # Wall time: 187 ms
# sat = MapleChrono() # Wall time: 4min 42s
# sat = MapleCM() # Wall time: 11.7 s
# sat = Maplesat() # Wall time: 5.64 s
# sat = Minicard() # Wall time: 9.07 s
# sat = MinisatGH() # Wall time: 8.44 s


for clause in clauses:
    sat.add_clause(clause)

%time is_solvable = sat.solve()
if is_solvable:
    model = np.array(sat.get_model())
    sat_solution = np.array(sat.get_model())
    dice_solution = sat_to_dice(d, dice_names, sat_solution, compress=False)
else:
    dice_solution = None

print(dice_solution)
if is_solvable:
    counts = verify_go_first(dice_solution)
    print(counts)
    print(np.all(np.array(list(counts.values())) == score))

# ============================================================================

# score = 12**4 // 24
# go_first_4x12 = {
#     "A": [1, 8, 11, 14, 19, 22, 27, 30, 35, 38, 41, 48],
#     "B": [2, 7, 10, 15, 18, 23, 26, 31, 34, 39, 42, 47],
#     "C": [3, 6, 12, 13, 17, 24, 25, 32, 36, 37, 43, 46],
#     "D": [4, 5, 9,  16, 20, 21, 28, 29, 33, 40, 44, 45]
# }
#
# counts = verify_go_first(go_first_4x12)
# print(counts)
# print(np.all(np.array(list(counts.values())) == score))


# ============================================================================
m = 5
letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
dice_names = [letters[i] for i in range(m)]

d = 6
faces = {x: ["%s%i" % (x, i) for i in range(1, d + 1)] for x in dice_names}

pairwise_scores = {x: d ** 2 // 2 for x in permutations(dice_names, 2)}

dice_perms1 = [("A","B","C"), ("B","C","D"), ("C","D","E"), ("D", "E", "A"), ("E", "A", "B")]
dice_perms1 += [("A","B","D"), ("B","C","E"), ("C","D","A"), ("D", "E", "B"), ("E", "A", "C")]
dice_perms2 = sum([list(permutations(dp))[2::2] for dp in dice_perms1], [])

dice_perms = dice_perms1 + dice_perms2

score1 = d ** 3 // 3
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
# clauses += build_cardinality_clauses(d, var_dict_2, var_lists_2, pairwise_scores, vpool)

# ----------------------------------------------------------------------------

clauses += build_winner_clauses(d, var_dict_2, var_dict_m, dice_names, dice_perms)
# clauses += build_exclusivity_clauses(d, var_dict_m, dice_names, vpool)
clauses += build_cardinality_clauses(d, var_dict_m, var_lists_m, scores1, vpool, pb=PBEnc.atleast)
clauses += build_cardinality_clauses(d, var_dict_m, var_lists_m, scores2, vpool, pb=PBEnc.atmost)

# ----------------------------------------------------------------------------

print("Starting SAT solver.")
# sat = Minisat22() # Wall time: 8.63 s
# sat = Glucose3() # Wall time: 174 ms
sat = Glucose4() # Wall time: 207 ms
# sat = Lingeling() # Wall time: 2.35 s
# sat = Mergesat3() # Wall time: 21.8 s
# sat = Cadical() # Wall time: 1min 16s
# sat = Gluecard3() # Wall time: 185 ms
# sat = Gluecard4() # Wall time: 187 ms
# sat = MapleChrono() # Wall time: 4min 42s
# sat = MapleCM() # Wall time: 11.7 s
# sat = Maplesat() # Wall time: 5.64 s
# sat = Minicard() # Wall time: 9.07 s
# sat = MinisatGH() # Wall time: 8.44 s


for clause in clauses:
    sat.add_clause(clause)

%time is_solvable = sat.solve()
if is_solvable:
    model = np.array(sat.get_model())
    sat_solution = np.array(sat.get_model())
    dice_solution = sat_to_dice(d, dice_names, sat_solution, compress=False)
else:
    dice_solution = None

print(dice_solution)
if is_solvable:
    counts = verify_go_first(dice_solution)
    print(counts)
    print(np.all(np.array(list(counts.values())) == score))


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
