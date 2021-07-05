import numpy as np
from scipy.special import factorial
from itertools import permutations, product

from pysat.solvers import Minisat22, Minicard
from clauses import build_clauses, build_max_min_clauses
from clauses import build_permutation_clauses
from clauses import build_cardinality_lits, build_exclusivity_lits


def compare_dice(first, second):
    hits = 0
    for x in first:
        for y in second:
            if y < x:
                hits += 1
    return hits


def compare_doubled_dice(first, second, comp="max"):
    d = len(first)
    hits = 0
    if comp == "max":
        indices = range(1, 2 * d, 2)
    if comp == "min":
        indices = range(2 * d - 1, 0, -2)
    for i, x in zip(indices, first):
        for j, y in zip(indices, second):
            if y < x:
                hits += i * j
    return hits


def recover_values(d, dice_names, constraints):
    natural_faces = []
    for die in dice_names:
        faces = np.arange(d)
        for die2 in dice_names:
            if die != die2:
                faces += constraints[(die, die2)].sum(1)
        natural_faces.append(faces)
    return natural_faces


def compress_values(*args):
    T = {}
    for i, die in enumerate(args):
        T.update({k: i for k in die})
    n = len(T.keys())
    T_list = [T[i] for i in range(n)]
    current_value = 0
    current_die = T_list[0]
    compressed_dice = [[] for _ in args]
    compressed_dice[current_die].append(current_value)
    for i in range(1, n):
        previous_die = current_die
        current_die = T_list[i]
        if current_die != previous_die:
            current_value += 1
        compressed_dice[current_die].append(current_value)
    return compressed_dice


def sat_to_constraints(d, dice_names, sat_solution, compress=True):
    dice_pairs = list(permutations(dice_names, 2))
    n = len(dice_pairs)

    signs_array = (sat_solution[: (n * d ** 2)] > 0).reshape((n, d, d))
    constraints = {v: s for v, s in zip(dice_pairs, signs_array)}

    return constraints


def sat_to_dice(d, dice_names, sat_solution, compress=True):
    constraints = sat_to_constraints(d, dice_names, sat_solution)

    natural_faces = recover_values(d, dice_names, constraints)
    if compress:
        dice_faces = compress_values(*natural_faces)
        dice_dict = {k: v for k, v in zip(dice_names, dice_faces)}
    else:
        dice_dict = {k: v for k, v in zip(dice_names, natural_faces)}
    return dice_dict


# ----------------------------------------------------------------------------


def verify_solution(scores, dice_solution):
    for x, y in scores:
        check = compare_dice(dice_solution[x], dice_solution[y])
        print((x, y), check, scores[(x, y)])


def verify_doubling_solution(
    scores, doubled_scores_max, doubled_scores_min, dice_solution
):
    verify_solution(scores, dice_solution)
    print()
    for x, y in doubled_scores_max:
        check = compare_doubled_dice(dice_solution[x], dice_solution[y], "max")
        print((x, y), check, doubled_scores_max[(x, y)])
    print()
    for x, y in doubled_scores_min:
        check = compare_doubled_dice(dice_solution[x], dice_solution[y], "min")
        print((x, y), check, doubled_scores_min[(x, y)])


def verify_go_first(dice_solution):
    m = len(dice_solution)
    keys = list(dice_solution.keys())
    d = len(dice_solution[keys[0]])
    check = d ** m // factorial(m, exact=True)
    counts = {x: 0 for x in permutations(range(len(dice_solution)))}
    for outcome in product(*dice_solution.values()):
        key = tuple(np.argsort(outcome))
        counts[key] += 1
    for k in counts:
        print(k, check, counts[k])


# ============================================================================


def sat_search(d, dice_names, scores, card_clauses=False):
    clauses, cardinality_lits = build_clauses(
        d, dice_names, scores, card_clauses=card_clauses
    )

    sat = Minicard()
    for clause in clauses:
        sat.add_clause(clause)

    if not card_clauses:
        for x, lits in cardinality_lits.items():
            sat.add_atmost(lits, scores[x])
            conv_lits = [-l for l in lits]
            sat.add_atmost(conv_lits, d ** 2 - scores[x])

    is_solvable = sat.solve()
    if is_solvable:
        sat_solution = np.array(sat.get_model())
        dice_solution = sat_to_dice(d, dice_names, sat_solution)
    else:
        dice_solution = None

    return dice_solution


# ----------------------------------------------------------------------------


def sat_search_max_min(d, dice_names, scores, max_scores, min_scores):
    clauses = build_max_min_clauses(d, dice_names, scores, max_scores, min_scores)

    sat = Minisat22()
    for clause in clauses:
        sat.add_clause(clause)

    is_solvable = sat.solve()
    if is_solvable:
        model = np.array(sat.get_model())
        sat_solution = np.array(sat.get_model())
        dice_solution = sat_to_dice(d, dice_names, sat_solution, compress=False)
    else:
        dice_solution = None

    return dice_solution


# ----------------------------------------------------------------------------


def sat_search_go_first(d, dice_names, scores_2, scores_m):
    m = len(dice_names)
    start_enum = 1
    dice_pairs = list(permutations(dice_names, 2))
    faces = {x: ["%s%i" % (x, i) for i in range(1, d + 1)] for x in dice_names}

    # ------------------------------------------------------------------------

    var_lists_2 = {(x, y): list(product(faces[x], faces[y])) for (x, y) in dice_pairs}
    variables_2 = sum(var_lists_2.values(), [])

    var_dict_2 = dict((v, k) for k, v in enumerate(variables_2, start_enum))
    start_enum += len(variables_2)

    # ------------------------------------------------------------------------

    dice_perms = list(permutations(dice_names))
    var_lists_m = {xs: list(product(*[faces[x] for x in xs])) for xs in dice_perms}
    variables_m = sum(var_lists_m.values(), [])
    var_dict_m = dict((v, k) for k, v in enumerate(variables_m, start_enum))
    start_enum += len(variables_m)

    # ------------------------------------------------------------------------

    clauses_2, cardinality_lits_2 = build_clauses(d, dice_names, scores_2)

    # ------------------------------------------------------------------------

    clauses_m = build_permutation_clauses(d, var_dict_2, var_dict_m, dice_names)
    cardinality_lits_m = build_cardinality_lits(d, var_dict_m, var_lists_m)
    exclusivity_lits = build_exclusivity_lits(d, var_dict_m, dice_names)

    # ------------------------------------------------------------------------

    clauses = clauses_2 + clauses_m

    sat = Minicard()

    for clause in clauses:
        sat.add_clause(clause)

    for x, lits in cardinality_lits_2.items():
        sat.add_atmost(lits, scores_2[x])
        conv_lits = [-l for l in lits]
        sat.add_atmost(conv_lits, d ** 2 - scores_2[x])

    for x, lits in cardinality_lits_m.items():
        sat.add_atmost(lits, scores_m[x])
        conv_lits = [-l for l in lits]
        sat.add_atmost(conv_lits, d ** m - scores_m[x])

    for x, lits in exclusivity_lits.items():
        sat.add_atmost(lits, 1)
        conv_lits = [-l for l in lits]
        sat.add_atmost(conv_lits, len(lits) - 1)

    is_solvable = sat.solve()
    if is_solvable:
        sat_solution = np.array(sat.get_model())
        dice_solution = sat_to_dice(d, dice_names, sat_solution, compress=False)
    else:
        dice_solution = None

    return dice_solution
