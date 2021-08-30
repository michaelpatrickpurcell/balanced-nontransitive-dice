# SAT strategy
import numpy as np
from itertools import permutations
from pysat.pb import PBEnc

from utils import sat_search, sat_exhaust
from utils import verify_solution, sat_to_dice

# =============================================================================
# Introduction
# =============================================================================
# Condorcet dice

dice_names = "ABC"
dice_pairs = list(permutations(dice_names, 2))

n = len(dice_names)
d = 3

# ----------------------------------------------------------------------------

w1 = 5
temp = [w1, d ** 2 - w1]
S = [[temp[(j - i) % (n - 1)] for j in range(n - 1)] for i in range(n)]
scores = {p: s for p, s in zip(dice_pairs, sum(S, []))}

# ----------------------------------------------------------------------------

sat_solution = sat_search(d, dice_names, scores)
dice_solution = sat_to_dice(d, dice_names, sat_solution)
for x in dice_solution:
    print("%s: " % x, dice_solution[x])

if dice_solution is not None:
    print(verify_solution(scores, dice_solution))
print()

# =============================================================================
# Efron's dice

dice_names = "ABCD"
dice_pairs = list(permutations(dice_names, 2))

n = len(dice_names)
d = 6

# ----------------------------------------------------------------------------

w1 = 24
scores = {p: w1 for p in [("A", "B"), ("B", "C"), ("C", "D"), ("D", "A")]}

# ----------------------------------------------------------------------------

sat_solution = sat_search(d, dice_names, scores)
dice_solution = sat_to_dice(d, dice_names, sat_solution)
for x in dice_solution:
    print("%s: " % x, dice_solution[x])

if dice_solution is not None:
    print(verify_solution(scores, dice_solution))
print()

# =============================================================================
# Simple Cyclic Dice
# =============================================================================
# A minimal simple cyclic set of four dice

dice_names = "ABCD"
dice_pairs = list(permutations(dice_names, 2))

n = len(dice_names)
d = 6

# ----------------------------------------------------------------------------

w1 = 22
w2 = d ** 2 // 2
temp = [w1, w2, d ** 2 - w1]
S = [[temp[(j - i) % (n - 1)] for j in range(n - 1)] for i in range(n)]
scores = {p: s for p, s in zip(dice_pairs, sum(S, []))}

# ----------------------------------------------------------------------------

sat_solution = sat_search(d, dice_names, scores)
dice_solution = sat_to_dice(d, dice_names, sat_solution)
for x in dice_solution:
    print("%s: " % x, dice_solution[x])

if dice_solution is not None:
    print(verify_solution(scores, dice_solution))
print()

# =============================================================================
# Rotationally Symmetric Dice
# =============================================================================
# Pentagram dice (a set of five six-sided rotationally symmetric dice)

dice_names = "ABCDE"
dice_pairs = list(permutations(dice_names, 2))
n = len(dice_names)
d = 6

# ----------------------------------------------------------------------------

w1 = 24
w2 = 20
temp = [w1, w2, d ** 2 - w2, d ** 2 - w1]
S = [[temp[(j - i) % (n - 1)] for j in range(n - 1)] for i in range(n)]
scores = {p: s for p, s in zip(dice_pairs, sum(S, []))}

# ----------------------------------------------------------------------------

sat_solution = sat_search(d, dice_names, scores)
dice_solution = sat_to_dice(d, dice_names, sat_solution)
for x in dice_solution:
    print("%s: " % x, dice_solution[x])

if dice_solution is not None:
    print(verify_solution(scores, dice_solution))
print()

# =============================================================================
# Hexagram dice (a set of six six-sided rotationally symmetric dice)

dice_names = "ABCDEF"
dice_pairs = list(permutations(dice_names, 2))

n = len(dice_names)
d = 6

# ----------------------------------------------------------------------------

w1 = 21
w2 = 15
w3 = 18
temp = [w1, w2, w3, d ** 2 - w2, d ** 2 - w1]
S = [[temp[(j - i) % (n - 1)] for j in range(n - 1)] for i in range(n)]
scores = {p: s for p, s in zip(dice_pairs, sum(S, []))}

# ----------------------------------------------------------------------------

sat_solution = sat_search(d, dice_names, scores)
dice_solution = sat_to_dice(d, dice_names, sat_solution)
for x in dice_solution:
    print("%s: " % x, dice_solution[x])

if dice_solution is not None:
    print(verify_solution(scores, dice_solution))
print()

# =============================================================================
# Paley Dice
# =============================================================================
# Oskar dice (a set of seven three-sided dice that realize P_7)

dice_names = "ABCDEFG"
dice_pairs = list(permutations(dice_names, 2))

n = len(dice_names)
d = 3

# ----------------------------------------------------------------------------

w = 5
mask_index = [1, 2, 4]
mask = [1 if (i + 1) in mask_index else 0 for i in range(n - 1)]
temp = [w if mask[i] else d ** 2 - w for i in range(n - 1)]
S = [[temp[(j - i) % (n - 1)] for j in range(n - 1)] for i in range(n)]
scores = {p: s for p, s in zip(dice_pairs, sum(S, []))}

# ----------------------------------------------------------------------------

sat_solution = sat_search(d, dice_names, scores)
dice_solution = sat_to_dice(d, dice_names, sat_solution)
for x in dice_solution:
    print("%s: " % x, dice_solution[x])

if dice_solution is not None:
    print(verify_solution(scores, dice_solution))
print()

# =============================================================================
# A set of nineteen five-sided dice that realize P_19

dice_names = "ABCDEFGHIJKLMNOPQRS"
dice_pairs = list(permutations(dice_names, 2))

n = 19
d = 5

# ----------------------------------------------------------------------------

w = 13
mask_index = [1, 4, 5, 6, 7, 9, 11, 16, 17]
mask = [1 if (i + 1) in mask_index else 0 for i in range(n - 1)]
temp = [w if mask[i] else d ** 2 - w for i in range(n - 1)]
S = [[temp[(j - i) % (n - 1)] for j in range(n - 1)] for i in range(n)]
scores = {p: s for p, s in zip(dice_pairs, sum(S, []))}

# ----------------------------------------------------------------------------

sat_solution = sat_search(d, dice_names, scores)
dice_solution = sat_to_dice(d, dice_names, sat_solution)
for x in dice_solution:
    print("%s: " % x, dice_solution[x])

if dice_solution is not None:
    print(verify_solution(scores, dice_solution))
print()

# =============================================================================
# Here we show that no set of nineteen three-sided dice realizes P_19

dice_names = "ABCDEFGHIJKLMNOPQRS"
dice_pairs = list(permutations(dice_names, 2))

n = 19
d = 3

# ----------------------------------------------------------------------------

w = 5
mask_index = [1, 4, 5, 6, 7, 9, 11, 16, 17]
mask = [1 if (i + 1) in mask_index else 0 for i in range(n - 1)]
temp = [w if mask[i] else d ** 2 - w for i in range(n - 1)]
S = [[temp[(j - i) % (n - 1)] for j in range(n - 1)] for i in range(n)]
scores = {p: s for p, s in zip(dice_pairs, sum(S, [])) if s == w}

# ----------------------------------------------------------------------------

sat_solution = sat_search(d, dice_names, scores, pb_type=PBEnc.atleast)
print(sat_solution)

# =============================================================================
# Here we show that no set of nineteen four-sided dice realizes P_19

dice_names = "ABCDEFGHIJKLMNOPQRS"
dice_pairs = list(permutations(dice_names, 2))

n = 19
d = 4

# ----------------------------------------------------------------------------

w = 9
mask_index = [1, 4, 5, 6, 7, 9, 11, 16, 17]
mask = [1 if (i + 1) in mask_index else 0 for i in range(n - 1)]
temp = [w if mask[i] else d ** 2 - w for i in range(n - 1)]
S = [[temp[(j - i) % (n - 1)] for j in range(n - 1)] for i in range(n)]
scores = {p: s for p, s in zip(dice_pairs, sum(S, [])) if s == w}

# ----------------------------------------------------------------------------

sat_solution = sat_search(d, dice_names, scores, pb_type=PBEnc.atleast)
print(sat_solution)
