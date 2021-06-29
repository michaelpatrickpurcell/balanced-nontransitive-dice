import pysat
from pysat.pb import PBEnc
from itertools import product, permutations


def build_clauses(d, dice_names, scores, vpool=None):
    """
    Build the clauses that describe the SAT problem.
    """
    dice_pairs = list(permutations(dice_names, 2))
    n = len(dice_pairs)
    f = {x: ["%s%i" % (x, i) for i in range(1, d + 1)] for x in dice_names}
    var_lists = {(x, y): list(product(f[x], f[y])) for (x, y) in dice_pairs}

    variables = sum(var_lists.values(), [])
    var_dict = dict((v, k) for k, v in enumerate(variables, 1))

    if vpool == None:
        vpool = pysat.formula.IDPool(start_from=n * d ** 2 + 1)

    clauses = []
    clauses += build_cardinality_clauses(d, var_dict, var_lists, scores, vpool)
    clauses += build_converse_clauses(d, var_dict, dice_names)  # , vpool)
    clauses += build_sorting_clauses(d, var_dict, f)  # dice_names)
    clauses += build_transitivity_clauses(d, var_dict, f)  # dice_names)
    clauses += build_symmetry_clauses(d, var_dict, dice_names)
    return clauses


def build_cardinality_clauses(d, var_dict, var_lists, scores, vpool):
    """
    These clauses ensure that each pair of dice have the specified relationship.
    """
    dice_pairs = var_lists.keys()
    cardinality_clauses = []
    for dice_pair, score in scores.items():
        var_list = var_lists[dice_pair]
        score = scores[dice_pair]
        lits = [var_dict[v] for v in var_list]
        cnf = PBEnc.equals(lits=lits, bound=score, vpool=vpool, encoding=0)
        cardinality_clauses += cnf.clauses
    return cardinality_clauses


def build_lower_bound_clauses(d, var_dict, var_lists, scores, vpool):
    """
    These clauses ensure that each pair of dice have the specified relationship.
    """
    dice_pairs = var_lists.keys()
    cardinality_clauses = []
    for dice_pair, score in scores.items():
        var_list = var_lists[dice_pair]
        lits = [var_dict[v] for v in var_list]
        cnf = PBEnc.geq(lits=lits, bound=score, vpool=vpool, encoding=0)
        cardinality_clauses += cnf.clauses
    return cardinality_clauses


def build_upper_bound_clauses(d, var_dict, var_lists, scores, vpool):
    """
    These clauses ensure that each pair of dice have the specified relationship.
    """
    dice_pairs = var_lists.keys()
    cardinality_clauses = []
    for dice_pair, score in scores.items():
        var_list = var_lists[dice_pair]
        lits = [var_dict[v] for v in var_list]
        cnf = PBEnc.atmost(lits=lits, bound=score, vpool=vpool, encoding=0)
        cardinality_clauses += cnf.clauses
    return cardinality_clauses


def build_horizontal_sorting_clauses(d, var_dict, face_dict):  # dice_names):
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


def build_transitivity_clauses(d, var_dict, face_dict):  # dice_names):
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


def build_max_doubling_clauses(d, var_dict_1v1, var_dict_2v2, dice_names):
    f = {x: ["%s%i" % (x, i) for i in range(1, d + 1)] for x in dice_names}
    doubling_clauses = []
    for x, y in permutations(dice_names, 2):
        for i, ii, j, jj in product(range(d), repeat=4):
            i_max = max(i, ii)
            j_max = max(j, jj)
            v1 = var_dict_1v1[(f[x][i_max], f[y][j_max])]
            key = ((f[x][i], f[x][ii]), (f[y][j], f[y][jj]))
            v2 = var_dict_2v2[key]
            doubling_clauses.append([-v1, v2])
            doubling_clauses.append([v1, -v2])
    return doubling_clauses


def build_min_doubling_clauses(d, var_dict_1v1, var_dict_2v2, dice_names):
    f = {x: ["%s%i" % (x, i) for i in range(1, d + 1)] for x in dice_names}
    doubling_clauses = []
    for x, y in permutations(dice_names, 2):
        for i, ii, j, jj in product(range(d), repeat=4):
            i_min = min(i, ii)
            j_min = min(j, jj)
            v1 = var_dict_1v1[(f[x][i_min], f[y][j_min])]
            key = ((f[x][i], f[x][ii]), (f[y][j], f[y][jj]))
            v2 = var_dict_2v2[key]
            doubling_clauses.append([-v1, v2])
            doubling_clauses.append([v1, -v2])
    return doubling_clauses
