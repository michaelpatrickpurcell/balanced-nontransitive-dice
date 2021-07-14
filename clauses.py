import pysat
from pysat.pb import PBEnc
from itertools import product, permutations

# ============================================================================
# Utilities for problems that involve 1v1 dice comparisons
# ============================================================================


def build_clauses(
    d, dice_names, scores, vpool=None, card_clauses=False, pb=PBEnc.equals
):
    """
    Build the clauses that describe the SAT problem.
    """
    dice_pairs = list(permutations(dice_names, 2))
    n = len(dice_pairs)
    faces = {x: ["%s%i" % (x, i) for i in range(1, d + 1)] for x in dice_names}
    var_lists = {(x, y): list(product(faces[x], faces[y])) for (x, y) in dice_pairs}

    variables = sum(var_lists.values(), [])
    var_dict = dict((v, k) for k, v in enumerate(variables, 1))

    clauses = []
    clauses += build_converse_clauses(d, var_dict, dice_names)
    clauses += build_sorting_clauses(d, var_dict, faces)
    clauses += build_transitivity_clauses(d, var_dict, faces)
    clauses += build_symmetry_clauses(d, var_dict, dice_names)
    clauses += build_structure_clauses(d, var_dict, var_lists, scores)

    if card_clauses:
        if vpool == None:
            vpool = pysat.formula.IDPool(start_from=n * d ** 2 + 1)
        clauses += build_cardinality_clauses(d, var_dict, var_lists, scores, vpool, pb)
        cardinality_lits = dict()
    else:
        cardinality_lits = build_cardinality_lits(d, var_dict, var_lists, scores)

    return clauses, cardinality_lits


# ============================================================================


def build_horizontal_sorting_clauses(d, var_dict, face_dict):
    """
    These clauses caputure the implications:
        if (Xi > Yj) then (Xi > Yk) for k <= j
    """
    horizontal_sorting_clauses = []
    for x, y in permutations(face_dict.keys(), 2):
        for i in range(d):
            for j in range(1, d):
                v1 = var_dict[(face_dict[x][i], face_dict[y][j])]
                v2 = var_dict[(face_dict[x][i], face_dict[y][j - 1])]
                horizontal_sorting_clauses.append([-v1, v2])
    return horizontal_sorting_clauses


def build_vertical_sorting_clauses(d, var_dict, face_dict):
    """
    These clauses capture the implications:
        if (Xi > Yj) then (Xk > Yj) for k >= i
    """
    vertical_sorting_clauses = []
    for x, y in permutations(face_dict.keys(), 2):
        for i in range(d - 1):
            for j in range(d):
                v1 = var_dict[(face_dict[x][i], face_dict[y][j])]
                v2 = var_dict[(face_dict[x][i + 1], face_dict[y][j])]
                vertical_sorting_clauses.append([-v1, v2])
    return vertical_sorting_clauses


def build_sorting_clauses(d, var_dict, face_dict):
    """
    These clauses ensure that each constraint matrix is lower triangular.
    """
    sorting_clauses = []
    sorting_clauses += build_horizontal_sorting_clauses(d, var_dict, face_dict)
    sorting_clauses += build_vertical_sorting_clauses(d, var_dict, face_dict)
    return sorting_clauses


def build_transitivity_clauses(d, var_dict, face_dict):
    """
    These clauses caputure the implications
        if (Xi > Yj) and (Yj > Zk) then (Xi > Zk)
    and
        if (Xi < Yj) and (Yj < Zk) then (Xi < Zk)
    """
    transitivity_clauses = []
    for x, y, z in permutations(face_dict.keys(), 3):
        for i in range(d):
            for j in range(d):
                for k in range(d):
                    v1 = var_dict[(face_dict[x][i], face_dict[y][j])]
                    v2 = var_dict[(face_dict[y][j], face_dict[z][k])]
                    v3 = var_dict[(face_dict[z][k], face_dict[x][i])]
                    transitivity_clauses.append([v1, v2, v3])
                    transitivity_clauses.append([-v1, -v2, -v3])
    return transitivity_clauses


def build_converse_clauses(d, var_dict, dice_names):
    """
    These clauses capture the implications:
        if (A1 > C1), then ~(C1 > A1)
    """
    converse_clauses = []
    for x, y in var_dict:
        v1 = var_dict[(x, y)]
        v2 = var_dict[(y, x)]
        converse_clauses.append([-v1, -v2])
        converse_clauses.append([v1, v2])
    return converse_clauses


def build_symmetry_clauses(d, var_dict, dice_names):
    """
    These clauses ensure that A1 is the smallest face.
    """
    symmetry_clauses = []
    v0 = dice_names[0]
    for v in dice_names[1:]:
        for i in range(1, d + 1):
            symmetry_clauses.append([-var_dict[(v0 + "1", v + ("%i" % i))]])
            symmetry_clauses.append([var_dict[(v + ("%i" % i), v0 + "1")]])
    return symmetry_clauses


def build_structure_clauses(d, var_dict, var_lists, scores):
    structure_clauses = []
    # for x, var_list in var_lists.items():
    for x in scores:
        var_list = var_lists[x]
        score = scores[x]
        for i, j in product(range(1, d + 1), repeat=2):
            v = var_dict[(x[0] + "%i" % i, x[1] + "%i" % j)]
            if ((d + 1 - i) * j) > score:
                structure_clauses.append([-v])
            elif (i * (d + 1 - j)) > d ** 2 - score:
                structure_clauses.append([v])
    return structure_clauses


# ----------------------------------------------------------------------------


def build_cardinality_clauses(d, var_dict, var_lists, scores, vpool, pb=PBEnc.equals):
    """
    These clauses ensure that each pair of dice have the specified relationship.
    """
    dice_pairs = var_lists.keys()
    cardinality_clauses = []
    for dice_pair, score in scores.items():
        var_list = var_lists[dice_pair]
        score = scores[dice_pair]
        lits = [var_dict[v] for v in var_list]
        cnf = pb(lits=lits, bound=score, vpool=vpool, encoding=0)
        cardinality_clauses += cnf.clauses
    return cardinality_clauses


def build_cardinality_lits(d, var_dict, var_lists, scores):
    cardinality_lits = dict()
    # for dice_pair, var_list in var_lists.items():
    for dice_pair in scores:
        var_list = var_lists[dice_pair]
        lits = [var_dict[v] for v in var_list]
        cardinality_lits[dice_pair] = lits
    return cardinality_lits


# ============================================================================
# Utilities for max/min dice-doubling problems
# ============================================================================


def build_max_min_clauses(d, dice_names, scores, max_scores, min_scores, vpool=None):
    dice_pairs = list(permutations(dice_names, 2))
    n = len(dice_pairs)
    start_enum = 1

    # ------------------------------------------------------------------------

    faces_1v1 = {x: ["%s%i" % (x, i) for i in range(1, d + 1)] for x in dice_names}
    var_lists_1v1 = {
        (x, y): list(product(faces_1v1[x], faces_1v1[y])) for (x, y) in dice_pairs
    }
    variables_1v1 = sum(var_lists_1v1.values(), [])

    var_dict_1v1 = dict((v, k) for k, v in enumerate(variables_1v1, start_enum))
    start_enum += len(variables_1v1)

    # ------------------------------------------------------------------------

    faces_2v2 = {x: list(product(faces_1v1[x], repeat=2)) for x in dice_names}
    var_lists_2v2 = {
        (x, y): list(product(faces_2v2[x], faces_2v2[y])) for (x, y) in dice_pairs
    }
    variables_2v2 = sum(var_lists_2v2.values(), [])

    var_dict_2v2_max = dict((v, k) for k, v in enumerate(variables_2v2, start_enum))
    start_enum += len(variables_2v2)

    var_dict_2v2_min = dict((v, k) for k, v in enumerate(variables_2v2, start_enum))
    start_enum += len(variables_2v2)

    # ------------------------------------------------------------------------
    # Set up a variable poll that will be used for all cardinality or
    # threshold constraint clauses
    if vpool == None:
        vpool = pysat.formula.IDPool(start_from=start_enum)

    # ------------------------------------------------------------------------

    # Build clauses for one-die comparisons
    clauses = []
    clauses += build_converse_clauses(d, var_dict_1v1, dice_names)
    clauses += build_sorting_clauses(d, var_dict_1v1, faces_1v1)
    clauses += build_transitivity_clauses(d, var_dict_1v1, faces_1v1)
    clauses += build_symmetry_clauses(d, var_dict_1v1, dice_names)
    clauses += build_cardinality_clauses(
        d, var_dict_1v1, var_lists_1v1, scores, vpool, PBEnc.equals
    )

    # ------------------------------------------------------------------------
    # Build clauses for two-dice comparisons with max-pooling
    clauses += build_doubling_clauses(
        d, var_dict_1v1, var_dict_2v2_max, dice_names, max
    )
    clauses += build_cardinality_clauses(
        d ** 2, var_dict_2v2_max, var_lists_2v2, max_scores, vpool, PBEnc.atleast
    )

    # ------------------------------------------------------------------------
    # Build clauses for two-dice comparisons with min-pooling
    clauses += build_doubling_clauses(
        d, var_dict_1v1, var_dict_2v2_min, dice_names, min
    )
    clauses += build_cardinality_clauses(
        d ** 2, var_dict_2v2_min, var_lists_2v2, min_scores, vpool, PBEnc.atmost
    )

    return clauses


# ----------------------------------------------------------------------------


def build_doubling_clauses(d, var_dict_1v1, var_dict_2v2, dice_names, pool_func):
    f = {x: ["%s%i" % (x, i) for i in range(1, d + 1)] for x in dice_names}
    doubling_clauses = []
    for x, y in permutations(dice_names, 2):
        for i, ii, j, jj in product(range(d), repeat=4):
            i_star = pool_func(i, ii)
            j_star = pool_func(j, jj)
            v1 = var_dict_1v1[(f[x][i_star], f[y][j_star])]
            key = ((f[x][i], f[x][ii]), (f[y][j], f[y][jj]))
            v2 = var_dict_2v2[key]
            doubling_clauses.append([-v1, v2])
            doubling_clauses.append([v1, -v2])
    return doubling_clauses


# ============================================================================
# Utilities for problems that involve m-way dice comparisons
# ============================================================================
def build_permutation_clauses(d, var_dict_2, var_dict_m, dice_names, m=None):
    if m == None:
        m = len(dice_names)
    faces = {x: ["%s%i" % (x, i) for i in range(1, d + 1)] for x in dice_names}
    permutation_clauses = []
    for xs in permutations(dice_names):
        for iis in product(range(d), repeat=m):
            z = list(zip(xs, iis))
            vs = [
                var_dict_2[(faces[x][i], faces[y][j])]
                for ((x, i), (y, j)) in zip(z, z[1:])
            ]
            w = var_dict_m[tuple([faces[y][j] for y, j in z])]
            permutation_clauses.append([-v for v in vs] + [w])
            permutation_clauses.extend([[-w, v] for v in vs])
    return permutation_clauses


def build_winner_clauses(d, var_dict_2, var_dict_m, dice_names, dice_perms, m=None):
    if m == None:
        m = len(dice_names)
    faces = {x: ["%s%i" % (x, i) for i in range(1, d + 1)] for x in dice_names}
    winner_clauses = []
    for xs in dice_perms:
        for iis in product(range(d), repeat=m):
            x, i = xs[0], iis[0]
            z = list(zip(xs[1:], iis[1:]))
            vs = [var_dict_2[(faces[x][i], faces[y][j])] for y, j in z]
            w = var_dict_m[tuple([faces[x][i]] + [faces[y][j] for y, j in z])]
            winner_clauses.append([-v for v in vs] + [w])
            winner_clauses.extend([[-w, v] for v in vs])
    return permutation_clauses


def build_exclusivity_clauses(d, var_dict_m, dice_names, vpool, m=None):
    if m == None:
        m = len(dice_names)
    faces = {x: ["%s%i" % (x, i) for i in range(1, d + 1)] for x in dice_names}
    exclusivity_clauses = []
    for x in product(range(d), repeat=m):
        column = [faces[dice_names[i]][x[i]] for i in range(m)]
        lits = [var_dict_m[tuple(key)] for key in permutations(column)]
        cnf = PBEnc.equals(lits=lits, bound=1, vpool=vpool, encoding=0)
        exclusivity_clauses += cnf.clauses
    return exclusivity_clauses


def build_exclusivity_lits(d, var_dict_m, dice_names, m=None):
    if m == None:
        m = len(dice_names)
    faces = {x: ["%s%i" % (x, i) for i in range(1, d + 1)] for x in dice_names}
    exclusivity_lits = dict()
    for x in product(range(d), repeat=m):
        column = [faces[dice_names[i]][x[i]] for i in range(m)]
        lits = [var_dict_m[tuple(key)] for key in permutations(column)]
        exclusivity_lits[x] = lits
    return exclusivity_lits
