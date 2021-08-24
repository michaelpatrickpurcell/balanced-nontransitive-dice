import numpy as np

from itertools import permutations, product
from pysat.solvers import Minicard

from constraints import build_constraints


# ============================================================================


def sat_search(d, dice_names, scores):
    """
    Find a solution to a nonstandard dice problem
    """
    sat = build_sat(d=d, dice_names=dice_names, scores=scores)

    is_solvable = sat.solve()
    if is_solvable:
        sat_solution = np.array(sat.get_model())
    else:
        sat_solution = None

    return sat_solution


def build_sat(d, dice_names, scores, symmetry_clauses=True):
    """
    Build a SAT solver to solve a nonstandard dice problem
    """
    clauses, lits = build_constraints(d, dice_names, scores)

    sat = Minicard()
    for clause in clauses:
        sat.add_clause(clause)

    for x, ls in lits:
        sat.add_atmost(ls, scores[x])
        conv_ls = [-l for l in ls]
        sat.add_atmost(conv_ls, d ** 2 - scores[x])

    return sat


# ----------------------------------------------------------------------------


def sat_exhaust(d, dice_names, scores):
    """
    Find all solutions to a nonstandard dice problem
    """
    sat = build_sat(d=d, dice_names=dice_names, scores=scores)
    solutions = sat.enum_models()
    return [np.array(s) for s in solutions]


# ============================================================================


def sat_to_dice(d, dice_names, sat_solution):
    """
    Convert a SAT solution into a set of nonstandard dice
    """
    constraints = sat_to_constraints(d, dice_names, sat_solution)
    natural_faces = constraints_to_dice(d, dice_names, constraints)
    dice_dict = {k: v for k, v in zip(dice_names, natural_faces)}
    return dice_dict


def sat_to_constraints(d, dice_names, sat_solution):
    """
    Convert a SAT solution into a set of constraint matrices
    """
    dice_pairs = list(product(dice_names, repeat=2))
    n = len(dice_pairs)

    signs_array = (sat_solution[: (n * d ** 2)] > 0).reshape((n, d, d))
    constraints = {v: s for v, s in zip(dice_pairs, signs_array)}

    return constraints


def constraints_to_dice(d, dice_names, constraints):
    """
    Convert a set of constraint matrices into a set of dice
    """
    natural_faces = []
    for die in dice_names:
        faces = np.zeros(d, dtype=np.int)
        for die2 in dice_names:
            faces += constraints[(die, die2)].sum(1)
        natural_faces.append(faces)
    return natural_faces


# ============================================================================


def verify_solution(scores, dice_solution, verbose=False):
    """
    Verify that a set of dice is a solution to a nonstandard dice problem
    """
    all_check = True
    for x, y in scores:
        check = compare_dice(dice_solution[x], dice_solution[y])
        if check != scores[(x, y)]:
            all_check = False
        if verbose:
            print((x, y), check, scores[(x, y)])
    return all_check


def compare_dice(first, second):
    """
    Compute the number of faces of first that exceed faces of second
    """
    hits = 0
    for x in first:
        for y in second:
            if x > y:
                hits += 1
    return hits
