import numpy as np
from itertools import permutations
from pysat.solvers import Minisat22
from clauses import build_clauses


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


# ============================================================================


def sat_search(d, dice_names, scores):
    clauses = build_clauses(d, dice_names, scores)

    sat = Minisat22()
    for clause in clauses:
        sat.add_clause(clause)

    is_solvable = sat.solve()
    if is_solvable:
        sat_solution = np.array(sat.get_model())
        dice_solution = sat_to_dice(d, dice_names, sat_solution)
    else:
        dice_solution = None

    return dice_solution
