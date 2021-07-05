import numpy as np
from scipy.special import factorial
from itertools import permutations, product
from tqdm import tqdm
from utils import sat_search_go_first, verify_go_first

# ============================================================================

m = 3
letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
dice_names = [letters[i] for i in range(m)]

d = 6
scores_2 = {x: d ** 2 // 2 for x in permutations(dice_names, 2)}
scores_m = {x: d ** m // factorial(m, exact=True) for x in permutations(dice_names)}

# ----------------------------------------------------------------------------

dice_solution = sat_search_go_first(d, dice_names, scores_2, scores_m)
print(dice_solution)
if dice_solution is not None:
    verify_go_first(dice_solution)

# ============================================================================

m = 3
letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
dice_names = [letters[i] for i in range(m)]

d = 12
scores_2 = {x: d ** 2 // 2 for x in permutations(dice_names, 2)}
scores_m = {x: d ** m // factorial(m, exact=True) for x in permutations(dice_names)}

# ----------------------------------------------------------------------------

dice_solution = sat_search_go_first(d, dice_names, scores_2, scores_m)
print(dice_solution)
if dice_solution is not None:
    verify_go_first(dice_solution)

# ============================================================================

m = 3
letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
dice_names = [letters[i] for i in range(m)]

d = 30
scores_2 = {x: d ** 2 // 2 for x in permutations(dice_names, 2)}
scores_m = {x: d ** m // factorial(m, exact=True) for x in permutations(dice_names)}

# ----------------------------------------------------------------------------

dice_solution = sat_search_go_first(d, dice_names, scores_2, scores_m)
print(dice_solution)
if dice_solution is not None:
    verify_go_first(dice_solution)
