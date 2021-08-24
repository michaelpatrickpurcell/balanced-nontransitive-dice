import pysat
from itertools import product, chain


# ============================================================================


def build_constraints(d, dice_names, scores):
    """
    Build the clauses that describe the SAT problem.
    """
    dice_pairs = list(product(dice_names, repeat=2))
    n = len(dice_pairs)
    faces = {x: ["%s%i" % (x, i) for i in range(1, d + 1)] for x in dice_names}
    var_lists = {(x, y): list(product(faces[x], faces[y])) for (x, y) in dice_pairs}

    variables = sum(var_lists.values(), [])
    var_dict = dict((v, k) for k, v in enumerate(variables, 1))

    reflexive = build_reflexive_clauses(d, var_dict, faces)
    sorting = build_sorting_clauses(d, var_dict, faces)
    converse = build_converse_clauses(d, var_dict, dice_names)
    transitivity = build_transitivity_clauses(d, var_dict, faces)
    symmetry = build_symmetry_clauses(d, var_dict, dice_names)
    clauses = chain(reflexive, sorting, converse, transitivity, symmetry)

    cardinality_lits = build_cardinality_lits(d, var_dict, var_lists, scores)

    return clauses, cardinality_lits


# ============================================================================


def build_reflexive_clauses(d, var_dict, face_dict):
    """
    These clauses capture the structure of the intradice face relationships.
    """
    for x in face_dict.keys():
        for i in range(d):
            for j in range(d):
                v1 = var_dict[(face_dict[x][i], face_dict[x][j])]
                if i >= j:
                    yield [v1]
                else:
                    yield [-v1]


# ----------------------------------------------------------------------------


def build_sorting_clauses(d, var_dict, face_dict):
    """
    These clauses ensure that each constraint matrix is lower triangular.
    """
    h_sort = build_horizontal_sorting_clauses(d, var_dict, face_dict)
    v_sort = build_vertical_sorting_clauses(d, var_dict, face_dict)
    return chain(h_sort, v_sort)


def build_horizontal_sorting_clauses(d, var_dict, face_dict):
    """
    These clauses caputure the implications:
        if (Xi > Yj) then (Xi > Yk) for k <= j
    """
    for x, y in product(face_dict.keys(), repeat=2):
        for i in range(d):
            for j in range(1, d):
                v1 = var_dict[(face_dict[x][i], face_dict[y][j])]
                v2 = var_dict[(face_dict[x][i], face_dict[y][j - 1])]
                yield [-v1, v2]


def build_vertical_sorting_clauses(d, var_dict, face_dict):
    """
    These clauses capture the implications:
        if (Xi > Yj) then (Xk > Yj) for k >= i
    """
    for x, y in product(face_dict.keys(), repeat=2):
        for i in range(d - 1):
            for j in range(d):
                v1 = var_dict[(face_dict[x][i], face_dict[y][j])]
                v2 = var_dict[(face_dict[x][i + 1], face_dict[y][j])]
                yield [-v1, v2]


# ----------------------------------------------------------------------------


def build_converse_clauses(d, var_dict, dice_names):
    """
    These clauses capture the implications:
        (A1 > C1) if and only if ~(C1 > A1)
    """
    for x, y in var_dict:
        if x != y:
            v1 = var_dict[(x, y)]
            v2 = var_dict[(y, x)]
            yield [-v1, -v2]


# ----------------------------------------------------------------------------


def build_transitivity_clauses(d, var_dict, face_dict):
    """
    These clauses caputure the implications
        if (Xi > Yj) and (Yj > Zk) then (Xi > Zk)
    and
        if (Xi < Yj) and (Yj < Zk) then (Xi < Zk)
    """
    for x, y, z in product(face_dict.keys(), repeat=3):
        for i in range(d):
            for j in range(d):
                for k in range(d):
                    v1 = var_dict[(face_dict[x][i], face_dict[y][j])]
                    v2 = var_dict[(face_dict[y][j], face_dict[z][k])]
                    v3 = var_dict[(face_dict[x][i], face_dict[z][k])]
                    yield [-v1, -v2, v3]


# ----------------------------------------------------------------------------


def build_symmetry_clauses(d, var_dict, dice_names):
    """
    These clauses ensure that A1 is the smallest face.
    """
    v0 = dice_names[0]
    for v in dice_names[1:]:
        for i in range(1, d + 1):
            yield [-var_dict[(v0 + "1", v + ("%i" % i))]]
            yield [var_dict[(v + ("%i" % i), v0 + "1")]]


# ----------------------------------------------------------------------------


def build_cardinality_lits(d, var_dict, var_lists, scores):
    """
    These literals ensure that P(R_X > R_Y) = scores[(X,Y)].
    """
    for x in scores:
        var_list = var_lists[x]
        ls = [var_dict[v] for v in var_list]
        yield (x, ls)
