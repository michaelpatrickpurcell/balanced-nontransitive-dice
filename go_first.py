import numpy as np
from scipy.special import factorial
from itertools import permutations, product
from tqdm import tqdm
from utils import sat_search_go_first, verify_go_first
from utils import dice_to_word, word_to_dice, dice_to_constraints

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

n = 3
m = n
letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
dice_names = [letters[i] for i in range(n)]

d = 12
scores_2 = {x: d ** 2 // 2 for x in permutations(dice_names, 2)}
scores_m = {x: d ** m // factorial(m, exact=True) for x in permutations(dice_names)}

# ----------------------------------------------------------------------------

dice_solution = sat_search_go_first(d, dice_names, scores_2, scores_m)
print(dice_solution)
if dice_solution is not None:
    verify_go_first(dice_solution)

# ============================================================================

n = 3
m = n
letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
dice_names = [letters[i] for i in range(n)]

d = 30
scores_2 = {x: d ** 2 // 2 for x in permutations(dice_names, 2)}
scores_m = {x: d ** m // factorial(m, exact=True) for x in permutations(dice_names)}

# ----------------------------------------------------------------------------

dice_solution = sat_search_go_first(d, dice_names, scores_2, scores_m)
print(dice_solution)
if dice_solution is not None:
    verify_go_first(dice_solution)


# ============================================================================
# Look for sets of dice that are 3/m permutation-fair
n = 6
m = 3
letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
dice_names = [letters[i] for i in range(n)]

d = 12
scores_2 = {x: d ** 2 // 2 for x in permutations(dice_names, 2)}
scores_m = {x: d ** m // factorial(m, exact=True) for x in permutations(dice_names, m)}

# ----------------------------------------------------------------------------

dice_solution = sat_search_go_first(d, dice_names, scores_2, scores_m, m=m)
print(dice_solution)
if dice_solution is not None:
    verify_go_first(dice_solution)
    temp = dice_to_word(dice_solution)[0]
    palindrome_dice_solution = word_to_dice(temp + temp[::-1])
    verify_go_first(palindrome_dice_solution)

# score_perm3s(dice_to_word(dice_solution)[0])
