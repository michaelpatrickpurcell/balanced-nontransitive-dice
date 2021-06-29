import numpy as np
from itertools import permutations


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
        indices = range(1, 2*d, 2)
    if comp == "min":
        indices = range(2*d-1, 0, -2)
    for i,x in zip(indices, first):
        for j,y in zip(indices, second):
            if y < x:
                hits += i*j
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


def verify_doubling_solution(scores, doubled_scores_max, doubled_scores_min, dice_solution):
    verify_solution(scores, dice_solution)
    print()
    for x, y in doubled_scores_max:
        check = compare_doubled_dice(dice_solution[x], dice_solution[y], "max")
        print((x, y), check, doubled_scores_max[(x, y)])
    print()
    for x, y in doubled_scores_min:
        check = compare_doubled_dice(dice_solution[x], dice_solution[y], "min")
        print((x, y), check, doubled_scores_min[(x, y)])



# =================================================================
# DEPRECATED: These have been replaced by recover_values()
# =================================================================


# def recover_raw_values(d, dice_names, constraints):
#     """
#     Find real-valued faces that satisfy the given constraints
#     """
#     raw_faces = dict()
#     raw_faces[dice_names[0]] = list(np.arange(1, d + 1))
#
#     for die in dice_names[1:]:
#         lower_bounds = np.zeros(d)
#         upper_bounds = (d + 1) * np.ones(d)
#         for k in raw_faces:
#             lower_bounds, upper_bounds = update_bounds(
#                 lower_bounds, upper_bounds, constraints[(k, die)], raw_faces[k]
#             )
#         raw_faces[die] = find_faces(lower_bounds, upper_bounds)
#     return [raw_faces[v] for v in dice_names]
#
#
# def naturalize_values(*args):
#     temp = sum(args, [])
#     m = len(temp)
#     n = len(args)
#     d = m // n
#     all_faces = dict(zip(sorted(temp), range(m)))
#     ret = [[all_faces[args[i][j]] for j in range(d)] for i in range(n)]
#     return ret
#
#
# def update_bounds(lower_bounds, upper_bounds, constraints, faces):
#     """
#     Derive bounds on possible values from the specified constraints.
#     """
#     for i in range(d):
#         if i > 0:
#             lower_bounds[i] = max(lower_bounds[i], lower_bounds[i - 1])
#         for j in range(d):
#             if constraints[j, i]:
#                 upper_bounds[i] = min(faces[j], upper_bounds[i])
#             else:
#                 lower_bounds[i] = max(faces[j], lower_bounds[i])
#     return (lower_bounds, upper_bounds)
#
#
# def find_faces(lower_bounds, upper_bounds):
#     """
#     Find values for the faces that conform to the constraints
#     imposed by the other dice and that are monotonically increasing.
#     """
#     faces = []
#     faces.append((lower_bounds[0] + upper_bounds[0]) / 2)
#     for i in range(1, d):
#         # Update lower bound to ensure that the value of the current
#         # face witll be larger than the value of the previous face.
#         lower_bounds[i] = max(lower_bounds[i], faces[-1])
#         faces.append((lower_bounds[i] + upper_bounds[i]) / 2)
#     return faces
