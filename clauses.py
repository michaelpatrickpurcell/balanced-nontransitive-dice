import pysat
from pysat.pb import PBEnc
from itertools import product, permutations


def build_clauses(d, dice_names, scores):
    """
    Build the clauses that describe the SAT problem.
    """
    dice_pairs = list(permutations(dice_names, 2))
    n = len(dice_pairs)
    f = {x: ["%s%i" % (x, i) for i in range(1, d + 1)] for x in dice_names}
    var_lists = {(x, y): list(product(f[x], f[y])) for (x, y) in dice_pairs}

    variables = sum(var_lists.values(), [])
    var_dict = dict((v, k) for k, v in enumerate(variables, 1))

    vpool = pysat.formula.IDPool(start_from=n * d ** 2 + 1)

    clauses = []
    clauses += build_cardinality_clauses(d, var_dict, var_lists, scores, vpool)
    clauses += build_converse_clauses(d, var_dict, dice_names, vpool)
    clauses += build_sorting_clauses(d, var_dict, dice_names)
    clauses += build_transitivity_clauses(d, var_dict, dice_names)
    clauses += build_symmetry_clauses(d, var_dict, dice_names)
    return clauses


def build_cardinality_clauses(d, var_dict, var_lists, scores, vpool):
    """
    These clauses ensure that each pair of dice have the specified relationship.
    """
    dice_pairs = var_lists.keys()
    cardinality_clauses = []
    for dice_pair in dice_pairs:
        var_list = var_lists[dice_pair]
        score = scores[dice_pair]
        lits = [var_dict[v] for v in var_list]
        cnf = PBEnc.equals(lits=lits, bound=score, vpool=vpool, encoding=0)
        cardinality_clauses += cnf.clauses
    return cardinality_clauses


def build_horizontal_sorting_clauses(d, var_dict, dice_names):
    """
    These clauses caputure the implications:
        if (Xi > Yj) then (Xi > Yk) for k <= j
    """
    horizontal_sorting_clauses = []
    dice_pairs = list(permutations(dice_names, 2))
    for x, y in dice_pairs:
        for i in range(1, (d + 1)):
            for j in range(2, (d + 1)):
                v1 = var_dict[(x + ("%i" % i), y + ("%i" % j))]
                v2 = var_dict[(x + ("%i" % i), y + ("%i" % (j - 1)))]
                horizontal_sorting_clauses.append([-v1, v2])
    return horizontal_sorting_clauses


def build_vertical_sorting_clauses(d, var_dict, dice_names):
    """
    These clauses capture the implications:
        if (Xi > Yj) then (Xk > Yj) for k >= i
    """
    vertical_sorting_clauses = []
    dice_pairs = list(permutations(dice_names, 2))
    for x, y in dice_pairs:
        for i in range(1, d):
            for j in range(1, (d + 1)):
                v1 = var_dict[(x + ("%i" % i), y + ("%i" % j))]
                v2 = var_dict[(x + ("%i" % (i + 1)), y + ("%i" % j))]
                vertical_sorting_clauses.append([-v1, v2])
    return vertical_sorting_clauses


def build_sorting_clauses(d, var_dict, dice_names):
    """
    These clauses ensure that each constraint matrix is lower triangular.
    """
    sorting_clauses = []
    sorting_clauses += build_horizontal_sorting_clauses(d, var_dict, dice_names)
    sorting_clauses += build_vertical_sorting_clauses(d, var_dict, dice_names)
    return sorting_clauses


def build_transitivity_clauses(d, var_dict, dice_names):
    """
    These clauses caputure the implications
        if (Xi > Yj) and (Yj > Zk) then (Xi > Zk)
    and
        if (Xi < Yj) and (Yj < Zk) then (Xi < Zk)
    """
    transitivity_clauses = []
    dice_triplets = list(permutations(dice_names, 3))
    for x, y, z in dice_triplets:
        for i in range(1, (d + 1)):
            for j in range(1, (d + 1)):
                for k in range(1, (d + 1)):
                    v1 = var_dict[(x + ("%i" % i), y + ("%i" % j))]
                    v2 = var_dict[(y + ("%i" % j), z + ("%i" % k))]
                    v3 = var_dict[(z + ("%i" % k), x + ("%i" % i))]
                    transitivity_clauses.append([v1, v2, v3])
                    transitivity_clauses.append([-v1, -v2, -v3])
    return transitivity_clauses


def build_converse_clauses(d, var_dict, dice_names, vpool):
    """
    These clauses capture the implications:
        if (A1 > C1), then ~(C1 > A1)
    """
    converse_clauses = []
    dice_pairs = list(permutations(dice_names, 2))
    for x, y in dice_pairs:
        for i in range(1, (d + 1)):
            for j in range(1, (d + 1)):
                v1 = var_dict[(x + ("%i" % i), y + ("%i" % j))]
                v2 = var_dict[(y + ("%i" % j), x + ("%i" % i))]
                cnf = PBEnc.equals(lits=[v1, v2], bound=1, vpool=vpool, encoding=0)
                converse_clauses += cnf.clauses
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
