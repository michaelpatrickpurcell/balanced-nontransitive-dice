import numpy as np

from itertools import permutations
from pysat.pb import PBEnc

from utils import verify_solution, sat_search

from argparse import ArgumentParser

parser = ArgumentParser(description="Find a set of three-player Oskar dice.")
parser.add_argument(
    "number_of_sides", type=int, help="The number of sides for each die"
)
parser.add_argument(
    "--cardinality_clauses",
    action="store_true",
    help="Use pseudo-Boolean clauses to enforce cardinality constraints",
)
args = parser.parse_args()

# =============================================================================
# Three player Oskar dice variant
# =============================================================================
m = 7
dice_names = ["D%i" % i for i in range(m)]

dice_pairs = list(permutations(dice_names, 2))
n = len(dice_pairs)

d = args.number_of_sides
cardinality_clauses = args.cardinality_clauses

# ----------------------------------------------------------------------------

score = d ** 2 // 2 + 1
mask_index = sorted([x for x in set(np.arange(1, m) ** 2 % m)])
mask = [1 if (i + 1) in mask_index else 0 for i in range(m - 1)]
temp = [score if mask[i] else d ** 2 - score for i in range(m - 1)]
S = [[temp[(j - i) % (m - 1)] for j in range(m - 1)] for i in range(m)]
# scores = {p: s for p, s in zip(dice_pairs, sum(S, []))}
scores = {p: s for p, s in zip(dice_pairs, sum(S, [])) if s == score}

# ----------------------------------------------------------------------------

dice_solution = sat_search(
    d,
    dice_names,
    scores,
    cardinality_clauses=cardinality_clauses,
    symmetry_clauses=False,
    pb=PBEnc.atleast,
)
print(dice_solution)
if dice_solution is not None:
    verify_solution(scores, dice_solution)
