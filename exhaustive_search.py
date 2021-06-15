import numpy as np
from itertools import combinations
import numba
from numba.core import types
from numba.typed import Dict

from utils import compare_dice

int_array = types.int64[:]
int_2d_array = types.int64[:,:]

from scipy.special import binom

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
            comp12 = compare_dice2(first_die, second_die)
            if comp12 > threshold:
                comp23 = compare_dice2(second_die, third_die)
                comp31 = compare_dice2(third_die, first_die)
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

%time hits3 = find_hits3(6)
%time hits3b = find_hits3b(6)

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
