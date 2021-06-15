import numpy as np
from itertools import combinations
import numba
from numba.core import types
from numba.typed import Dict

int_array = types.int64[:]
int_2d_array = types.int64[:,:]

from scipy.special import binom

@numba.njit
def compare_dice(first, second):
    hits = 0
    for x in first:
        for y in second:
            if y < x:
                hits += 1
    return hits

@numba.njit
def compare_dice2(first, second):
    return np.sum(np.expand_dims(first, 0) > np.expand_dims(second, 1))
    #return np.sum(first > second)

# =============================================================================
# Three dice sets
# =============================================================================
d = 3 # Anything bigger than d=6 will take a long time!
threshold = d**2 // 2
faces = set(range(1,3*d))

def find_hits3(d):
    threshold = d**2 // 2
    faces = set(range(1,3*d))
    hits = {}
    for first_die_tuple in combinations(faces, d-1):
        first_die = set(first_die_tuple + (0,))
        #print(first_die)
        for second_die_tuple in combinations(faces.difference(first_die), d):
            second_die = set(second_die_tuple)
            third_die = faces.difference(first_die.union(second_die))
            comp12 = compare_dice(first_die, second_die)
            if comp12 > threshold:
                comp23 = compare_dice(second_die, third_die)
                comp31 = compare_dice(third_die, first_die)
                scores = (comp12, comp23, comp31)
                res = (first_die, second_die, third_die)
                if (scores[0] == scores[1] == scores[2]):
                    score = scores[0]
                    if scores[0] in hits:
                        hits[score].append(res)
                    else:
                        hits[score] = [res]
    return hits

@numba.njit
def binom(n, r):
    ''' Binomial coefficient, nCr, aka the "choose" function
        n! / (r! * (n - r)!)
    '''
    p = 1
    for i in range(1, min(r, n - r) + 1):
        p *= n
        p //= i
        n -= 1
    return p

@numba.njit
def increment_odometer(odometer, max_val):
    if odometer[-1] < max_val:
        odometer[-1] += 1
    else:
        increment_odometer(odometer[:-1], max_val-1)
        odometer[-1] = odometer[-2] + 1

@numba.njit
def generate_first_dice(d):
    n = binom(3*d-1, d-1)
    ret = np.zeros((n,d), dtype=np.int64)
    odometer = np.arange(d)
    for i in range(n):
        ret[i] = np.copy(odometer)
        if i < n-1:
            increment_odometer(odometer, 3*d-1)
    return ret

@numba.njit
def generate_second_indices(d):
    n = binom(2*d, d)
    ret = np.zeros((n,d), dtype=np.int64)
    odometer = np.arange(d)
    for i in range(n):
        ret[i] = np.copy(odometer)
        if i < n-1:
            increment_odometer(odometer, 2*d-1)
    return ret

@numba.njit
def find_hits3b(d):
    hits = Dict.empty(key_type=types.int64, value_type=int_2d_array)
    threshold = d**2 // 2
    faces = set(np.arange(3*d))
    first_dice = generate_first_dice(d)
    second_indices = generate_second_indices(d)
    for first_die in first_dice:
        #print(first_die)
        remaining_faces = faces.difference(set(first_die))
        remaining_faces_array = np.array(sorted(list(remaining_faces)))
        for i in range(len(second_indices)):
            second_die = remaining_faces_array[second_indices[i]]
            third_die_faces = remaining_faces.difference(set(second_die))
            third_die = np.array(sorted(list(third_die_faces)))
            # print(second_die)
            # print(third_die)
            comp12 = compare_dice(first_die, second_die)
            if comp12 > threshold:
                comp23 = compare_dice(second_die, third_die)
                comp31 = compare_dice(third_die, first_die)
                scores = (comp12, comp23, comp31)
                res = (first_die, second_die, third_die)
                if (scores[0] == scores[1] == scores[2]):
                    score = scores[0]
                    if score in hits:
                        temp = np.expand_dims(np.concatenate((first_die, second_die, third_die)), 0)
                        hits[score] = np.row_stack((hits[score], temp))
                    else:
                        hits[score] = np.expand_dims(np.concatenate((first_die, second_die, third_die)), 0)
    return hits

# SAT strategy
import pysat
from pysat.solvers import Minisat22
from pysat.pb import PBEnc
from itertools import combinations_with_replacement, product

d = 3
score = 5
A_vars = ['A%i' % i for i in range(1,d+1)]
B_vars = ['B%i' % i for i in range(1,d+1)]
C_vars = ['C%i' % i for i in range(1,d+1)]

AB_vars = list(product(A_vars, B_vars))
BC_vars = list(product(B_vars, C_vars))
CA_vars = list(product(C_vars, A_vars))

vars = dict((v,k) for k,v in enumerate(AB_vars + BC_vars + CA_vars, 1))
vpool = pysat.formula.IDPool(start_from = len(vars)+1)

# Cardinality clauses
cardinality_clauses = []
cnf = PBEnc.equals(lits=[vars[v] for v in AB_vars], bound=score, vpool=vpool, encoding=0)
cardinality_clauses += cnf.clauses
cnf = PBEnc.equals(lits=[vars[v] for v in BC_vars], bound=score, vpool=vpool, encoding=0)
cardinality_clauses += cnf.clauses
cnf = PBEnc.equals(lits=[vars[v] for v in CA_vars], bound=score, vpool=vpool, encoding=0)
cardinality_clauses += cnf.clauses

# Sorting clauses
sorting_clauses = []

## Horizonal implications
## These clauses caputure the implication that if (A3 > B2) then (A3 > B1)
for x,y in (('A','B'), ('B','C'), ('C', 'A')):
    for i in range(1,(d+1)):
        for j in range(2,(d+1)):
            v1 = vars[(x+('%i' %i), y+('%i' % j))]
            v2 = vars[(x+('%i' %i), y+('%i' % (j-1)))]
            sorting_clauses.append([-v1, v2])

## Vertical implications
## These clauses capture the implication that if (A2 > B3) (then A3 > B3)
for x,y in (('A','B'), ('B','C'), ('C', 'A')):
    for i in range(1,d):
        for j in range(1,(d+1)):
            v1 = vars[(x+('%i' % i), y+('%i' % j))]
            v2 = vars[(x+('%i' % (i+1)), y+('%i' % j))]
            sorting_clauses.append([-v1, v2])

# Transitivity Clauses
transitivity_clauses = []
for i in range(1,(d+1)):
    for j in range(1,(d+1)):
        for k in range(1,(d+1)):
            v1 = vars[('A%i' %i, 'B%i' % j)]
            v2 = vars[('B%i' %j, 'C%i' % k)]
            v3 = vars[('C%i' %k, 'A%i' % i)]
            transitivity_clauses.append([v1, v2, v3])
            transitivity_clauses.append([-v1, -v2, -v3])


m = Minisat22()
for clause in cardinality_clauses:
    m.add_clause(clause)

for clause in sorting_clauses:
    m.add_clause(clause)

for clause in transitivity_clauses:
    m.add_clause(clause)

print(m.solve())
res = m.get_model()
print(res)

elim = [-1*r for r in res[:27]]
m.add_clause(elim)

# =============================================================================
# Four dice sets
# =============================================================================
d = 5 # Anything bigger than d=5 will take a long time!
threshold = d**2 / 2
faces = set(range(1,4*d))

hits = {}
used = set()
for first_die_tuple in combinations(faces, d-1):
    first_die = set(first_die_tuple + (0,))
    used = first_die
    print(first_die)

    for second_die_tuple in combinations(faces.difference(used), d):
        second_die = set(second_die_tuple)
        # print('\t%s' % second_die)
        score = compare_dice(first_die, second_die)
        if score <= threshold:
            continue
        # if score != 24:
        #     continue
        used = first_die.union(second_die)

        for third_die_tuple in combinations(faces.difference(used), d):
            third_die = set(third_die_tuple)
            temp23 = compare_dice(second_die, third_die)
            # print('\t\t%s %i %i' % (third_die, score, temp23))
            if temp23 != score:
                continue
            used = first_die.union(second_die.union(third_die))

            fourth_die = faces.difference(used)
            temp34 = compare_dice(third_die, fourth_die)
            temp41 = compare_dice(fourth_die, first_die)

            # print((score, temp23, temp34, temp41))

            if temp34 != score:
                continue
            if temp41 != score:
                continue
            res = (first_die, second_die, third_die, fourth_die)
            if score in hits:
                hits[score].append(res)
            else:
                hits[score] = [res]


# with open('intrans_dice/'+ '4d6' + '.pkl', 'wb') as f:
#     pickle.dump(hits, f, pickle.HIGHEST_PROTOCOL)


# SAT strategy
import pysat
from pysat.solvers import Minisat22
from pysat.pb import PBEnc
from itertools import combinations_with_replacement, product

d = 36
score = 864
cross_score = 648
A_vars = ['A%i' % i for i in range(1,d+1)]
B_vars = ['B%i' % i for i in range(1,d+1)]
C_vars = ['C%i' % i for i in range(1,d+1)]
D_vars = ['D%i' % i for i in range(1,d+1)]


AB_vars = list(product(A_vars, B_vars))
BC_vars = list(product(B_vars, C_vars))
CD_vars = list(product(C_vars, D_vars))
DA_vars = list(product(D_vars, A_vars))

CA_vars = list(product(C_vars, A_vars))
DB_vars = list(product(D_vars, B_vars))
AC_vars = list(product(A_vars, C_vars))
BD_vars = list(product(B_vars, D_vars))

var_list = AB_vars + BC_vars + CD_vars + DA_vars + CA_vars + DB_vars + AC_vars + BD_vars
vars = dict((v,k) for k,v in enumerate(var_list, 1))
vpool = pysat.formula.IDPool(start_from = len(vars)+1)

# Cardinality clauses
## Scores for adjacent pairs of dice
cardinality_clauses = []
cnf = PBEnc.equals(lits=[vars[v] for v in AB_vars], bound=score, vpool=vpool, encoding=0)
cardinality_clauses += cnf.clauses
cnf = PBEnc.equals(lits=[vars[v] for v in BC_vars], bound=score, vpool=vpool, encoding=0)
cardinality_clauses += cnf.clauses
cnf = PBEnc.equals(lits=[vars[v] for v in CD_vars], bound=score, vpool=vpool, encoding=0)
cardinality_clauses += cnf.clauses
cnf = PBEnc.equals(lits=[vars[v] for v in DA_vars], bound=score, vpool=vpool, encoding=0)
cardinality_clauses += cnf.clauses

## Scores for non-adjacent pairs of dice
cnf = PBEnc.equals(lits=[vars[v] for v in CA_vars], bound=cross_score, vpool=vpool, encoding=0)
cardinality_clauses += cnf.clauses
cnf = PBEnc.equals(lits=[vars[v] for v in DB_vars], bound=cross_score, vpool=vpool, encoding=0)
cardinality_clauses += cnf.clauses
cnf = PBEnc.equals(lits=[vars[v] for v in AC_vars], bound=cross_score, vpool=vpool, encoding=0)
cardinality_clauses += cnf.clauses
cnf = PBEnc.equals(lits=[vars[v] for v in BD_vars], bound=cross_score, vpool=vpool, encoding=0)
cardinality_clauses += cnf.clauses

# Sorting clauses
sorting_clauses = []
## Horizonal implications
## These clauses caputure the implication that if (A3 > B2) then (A3 > B1)
for x,y in (('A','B'), ('B','C'), ('C', 'D'), ('D', 'A'), ('C', 'A'), ('D', 'B'), ('A', 'C'), ('B', 'D')):
    for i in range(1,(d+1)):
        for j in range(2,(d+1)):
            v1 = vars[(x+('%i' %i), y+('%i' % j))]
            v2 = vars[(x+('%i' %i), y+('%i' % (j-1)))]
            sorting_clauses.append([-v1, v2])

## Vertical implications
## These clauses capture the implication that if (A2 > B3) then (A3 > B3)
for x,y in (('A','B'), ('B','C'), ('C', 'D'), ('D', 'A'), ('C', 'A'), ('D', 'B'), ('A', 'C'), ('B', 'D')):
    for i in range(1,d):
        for j in range(1,(d+1)):
            v1 = vars[(x+('%i' % i), y+('%i' % j))]
            v2 = vars[(x+('%i' % (i+1)), y+('%i' % j))]
            sorting_clauses.append([-v1, v2])

# Transitivity Clauses
## These clauses caputure the implication that if (A1 > B1) and (B1 > C1) then (A1 > C1)
transitivity_clauses = []
for x,y,z in (('A', 'B', 'C'), ('B', 'C', 'D'), ('C', 'D', 'A'), ('D', 'A', 'B')):
    for i in range(1,(d+1)):
        for j in range(1,(d+1)):
            for k in range(1,(d+1)):
                v1 = vars[(x+('%i' %i), y+('%i' % j))]
                v2 = vars[(y+('%i' %j), z+('%i' % k))]
                v3 = vars[(z+('%i' %k), x+('%i' % i))]
                transitivity_clauses.append([v1, v2, v3])
                transitivity_clauses.append([-v1, -v2, -v3])

# Converse Clauses
## These clauses capture the implication that if (A1 > C1), then ~(C1 > A1)
converse_clauses = []
for i in range(1,(d+1)):
    for j in range(1, (d+1)):
        cnf = PBEnc.equals(lits=[vars[('A%i' % i, 'C%i' % j)], vars[('C%i' % j, 'A%i' % i)]], bound=1, vpool=vpool, encoding=0)
        converse_clauses += cnf.clauses
        cnf = PBEnc.equals(lits=[vars[('B%i' % i, 'D%i' % j)], vars[('D%i' % j, 'B%i' % i)]], bound=1, vpool=vpool, encoding=0)
        converse_clauses += cnf.clauses

m = Minisat22()
for clause in cardinality_clauses:
    m.add_clause(clause)

for clause in sorting_clauses:
    m.add_clause(clause)

for clause in transitivity_clauses:
    m.add_clause(clause)

for clause in converse_clauses:
    m.add_clause(clause)


counter = 0
is_solvable = m.solve()
print(is_solvable)

while is_solvable:
    counter += 1
    print(counter)

    res = m.get_model()
    print(res[:(8*d**2)])

    elim = [-1*r for r in res[:(8*d**2)]]
    m.add_clause(elim)

    is_solvable = m.solve()

# =============================================================================
# Five dice sets
# =============================================================================
# SAT strategy
import pysat
from pysat.solvers import Minisat22
from pysat.pb import PBEnc
from itertools import combinations_with_replacement, product

# d = 3
# adj_score = 5
# acr_score = 5

d = 6
adj_score = 24
acr_score = 16

A_vars = ['A%i' % i for i in range(1,d+1)]
B_vars = ['B%i' % i for i in range(1,d+1)]
C_vars = ['C%i' % i for i in range(1,d+1)]
D_vars = ['D%i' % i for i in range(1,d+1)]
E_vars = ['E%i' % i for i in range(1,d+1)]

AB_vars = list(product(A_vars, B_vars))
BC_vars = list(product(B_vars, C_vars))
CD_vars = list(product(C_vars, D_vars))
DE_vars = list(product(D_vars, E_vars))
EA_vars = list(product(E_vars, A_vars))

CA_vars = list(product(C_vars, A_vars))
AD_vars = list(product(A_vars, D_vars))
DB_vars = list(product(D_vars, B_vars))
BE_vars = list(product(B_vars, E_vars))
EC_vars = list(product(E_vars, C_vars))

var_list = []
var_list += AB_vars + BC_vars + CD_vars + DE_vars + EA_vars
var_list += CA_vars + AD_vars + DB_vars + BE_vars + EC_vars
vars = dict((v,k) for k,v in enumerate(var_list, 1))
vpool = pysat.formula.IDPool(start_from = len(vars)+1)

# Cardinality clauses
## Scores for adjacent pairs of dice
cardinality_clauses = []
var_lists = []
var_lists += [AB_vars, BC_vars, CD_vars, DE_vars, EA_vars]
for var_list in var_lists:
    cnf = PBEnc.equals(
            lits=[vars[v] for v in var_list],
            bound=adj_score,
            vpool=vpool,
            encoding=0,
          )
    cardinality_clauses += cnf.clauses

var_lists = []
var_lists += [CA_vars, AD_vars, DB_vars, BE_vars, EC_vars]
for var_list in var_lists:
    cnf = PBEnc.equals(
            lits=[vars[v] for v in var_list],
            bound=acr_score,
            vpool=vpool,
            encoding=0,
          )
    cardinality_clauses += cnf.clauses


# Sorting clauses
sorting_clauses = []
## Horizonal implications
## These clauses caputure the implication that if (A3 > B2) then (A3 > B1)
var_pairs = []
var_pairs += [('A','B'), ('B','C'), ('C','D'), ('D','E'), ('E','A')]
var_pairs += [('C','A'), ('A','D'), ('D','B'), ('B','E'), ('E','C')]
for x,y in var_pairs:
    for i in range(1,(d+1)):
        for j in range(2,(d+1)):
            v1 = vars[(x+('%i' %i), y+('%i' % j))]
            v2 = vars[(x+('%i' %i), y+('%i' % (j-1)))]
            sorting_clauses.append([-v1, v2])

## Vertical implications
## These clauses capture the implication that if (A2 > B3) then (A3 > B3)
for x,y in var_pairs:
    for i in range(1,d):
        for j in range(1,(d+1)):
            v1 = vars[(x+('%i' % i), y+('%i' % j))]
            v2 = vars[(x+('%i' % (i+1)), y+('%i' % j))]
            sorting_clauses.append([-v1, v2])

# Transitivity Clauses
## These clauses caputure the implication that if (A1 > B1) and (B1 > C1) then (A1 > C1)
var_triplets = []
var_triplets += [('A', 'B', 'C'), ('B', 'C', 'D'), ('C', 'D', 'E'), ('D', 'E', 'A'), ('E', 'A', 'B')]
transitivity_clauses = []
for x,y,z in var_triplets:
    for i in range(1,(d+1)):
        for j in range(1,(d+1)):
            for k in range(1,(d+1)):
                v1 = vars[(x+('%i' %i), y+('%i' % j))]
                v2 = vars[(y+('%i' %j), z+('%i' % k))]
                v3 = vars[(z+('%i' %k), x+('%i' % i))]
                transitivity_clauses.append([v1, v2, v3])
                transitivity_clauses.append([-v1, -v2, -v3])

m = Minisat22()
for clause in cardinality_clauses:
    m.add_clause(clause)

for clause in sorting_clauses:
    m.add_clause(clause)

for clause in transitivity_clauses:
    m.add_clause(clause)

counter = 0
is_solvable = m.solve()
print(is_solvable)
res = np.array(m.get_model())
signs = res[:(10 * d**2)] > 0

AB_constraints = signs[:(d**2)].reshape(d,d)
BC_constraints = signs[(d**2):2*(d**2)].reshape(d,d)
CD_constraints = signs[2*(d**2):3*(d**2)].reshape(d,d)
DE_constraints = signs[3*(d**2):4*(d**2)].reshape(d,d)
EA_constraints = signs[4*(d**2):5*(d**2)].reshape(d,d)

CA_constraints = signs[5*(d**2):6*(d**2)].reshape(d,d)
AD_constraints = signs[6*(d**2):7*(d**2)].reshape(d,d)
DB_constraints = signs[7*(d**2):8*(d**2)].reshape(d,d)
BE_constraints = signs[8*(d**2):9*(d**2)].reshape(d,d)
EC_constraints = signs[9*(d**2):10*(d**2)].reshape(d,d)

def update_bounds(lower_bounds, upper_bounds, constraints, faces):
    for i in range(d):
        if i > 0:
            lower_bounds[i] = max(lower_bounds[i], lower_bounds[i-1])
        for j in range(d):
            if constraints[j,i]:
                upper_bounds[i] = min(faces[j], upper_bounds[i])
            else:
                lower_bounds[i] = max(faces[j], lower_bounds[i])
    return (lower_bounds, upper_bounds)

def find_faces(lower_bounds, upper_bounds):
    faces = []
    faces.append((lower_bounds[0] + upper_bounds[0]) / 2)
    for i in range(1, d):
        lower_bounds[i] = max(lower_bounds[i], faces[-1])
        faces.append((lower_bounds[i] + upper_bounds[i])/2)
    return faces

def collapse_values(dice):
    T = {}
    for i,die in enumerate(dice):
        T.update({k: i for k in die})
    n = len(T.keys())
    T_list = [T[i] for i in range(n)]
    current_value = 0
    current_die = T_list[0]
    collapsed_dice = [[] for _ in dice]
    collapsed_dice[current_die].append(current_value)
    for i in range(1,n):
        previous_die = current_die
        current_die = T_list[i]
        if current_die != previous_die:
            current_value += 1
        collapsed_dice[current_die].append(current_value)
    return collapsed_dice


A_faces = list(np.arange(1, d+1))

lower_bounds = np.zeros(d)
upper_bounds = (d+1) * np.ones(d)
lower_bounds, upper_bounds = update_bounds(lower_bounds, upper_bounds, AB_constraints, A_faces)

B_faces = find_faces(lower_bounds, upper_bounds)

lower_bounds = np.zeros(d)
upper_bounds = (d+1) * np.ones(d)
lower_bounds, upper_bounds = update_bounds(lower_bounds, upper_bounds, BC_constraints, B_faces)
lower_bounds, upper_bounds = update_bounds(lower_bounds, upper_bounds, (CA_constraints ^ True).transpose(), A_faces)

C_faces = find_faces(lower_bounds, upper_bounds)

lower_bounds = np.zeros(d)
upper_bounds = (d+1) * np.ones(d)
lower_bounds, upper_bounds = update_bounds(lower_bounds, upper_bounds, CD_constraints, C_faces)
lower_bounds, upper_bounds = update_bounds(lower_bounds, upper_bounds, (DB_constraints ^ True).transpose(), B_faces)
lower_bounds, upper_bounds = update_bounds(lower_bounds, upper_bounds, AD_constraints, A_faces)

D_faces = find_faces(lower_bounds, upper_bounds)

lower_bounds = np.zeros(d)
upper_bounds = (d+1) * np.ones(d)
lower_bounds, upper_bounds = update_bounds(lower_bounds, upper_bounds, DE_constraints, D_faces)
lower_bounds, upper_bounds = update_bounds(lower_bounds, upper_bounds, (EC_constraints ^ True).transpose(), C_faces)
lower_bounds, upper_bounds = update_bounds(lower_bounds, upper_bounds, BE_constraints, B_faces)
lower_bounds, upper_bounds = update_bounds(lower_bounds, upper_bounds, (EA_constraints ^ True).transpose(), A_faces)

E_faces = find_faces(lower_bounds, upper_bounds)

all_faces = dict(zip(sorted(A_faces + B_faces + C_faces + D_faces + E_faces), range(5*d)))
A = [all_faces[A_faces[i]] for i in range(d)]
B = [all_faces[B_faces[i]] for i in range(d)]
C = [all_faces[C_faces[i]] for i in range(d)]
D = [all_faces[D_faces[i]] for i in range(d)]
E = [all_faces[E_faces[i]] for i in range(d)]

A,B,C,D,E = collapse_values([A,B,C,D,E])

print(compare_dice(A, B))
print(compare_dice(B, C))
print(compare_dice(C, D))
print(compare_dice(D, E))
print(compare_dice(E, A))
print()
print(compare_dice(C, A))
print(compare_dice(A, D))
print(compare_dice(D, B))
print(compare_dice(B, E))
print(compare_dice(E, C))






A = np.array([1,9,11])
B = np.array([4,7,10])
C = np.array([2,5,14])
D = np.array([0,8,13])
E = np.array([3,6,12])
