import numpy as np
from scipy.special import factorial, comb
from copy import copy
from collections import Counter
import random
from itertools import combinations, permutations, product
from tqdm import tqdm

from utils import dice_to_constraints, permute_letters
from utils import word_to_dice, dice_to_word


def score_perm2s(word, verbose=False):
    dice = word_to_dice(word)
    dice_names = list(dice.keys())
    constraints = dice_to_constraints(dice, dtype=np.float)
    scores = dict()
    for x, y in permutations(dice_names, 2):
        score = np.sum(constraints[(x, y)])
        scores[(x, y)] = score
        if verbose:
            print(score)
    return scores


def score_perm3s(word, verbose=False):
    dice = word_to_dice(word)
    dice_names = list(dice.keys())
    constraints = dice_to_constraints(dice, dtype=np.float)
    scores = dict()
    for x, y, z in permutations(dice_names, 3):
        score = np.sum(constraints[(x, y)] @ constraints[(y, z)])
        scores[(x, y, z)] = score
        if verbose:
            print(score)
    return scores


def score_perm4s(word, verbose=False):
    dice = word_to_dice(word)
    dice_names = list(dice.keys())
    constraints = dice_to_constraints(dice, dtype=np.float)
    scores = dict()
    for w, x, y, z in permutations(dice_names, 4):
        score = np.sum(constraints[(w, x)] @ constraints[(x, y)] @ constraints[(y, z)])
        scores[(w, x, y, z)] = score
        if verbose:
            print(score)
    return scores


def score_perm5s(word, verbose=False):
    dice = word_to_dice(word)
    dice_names = list(dice.keys())
    constraints = dice_to_constraints(dice, dtype=np.float)
    scores = dict()
    for v, w, x, y, z in permutations(dice_names, 5):
        score = np.sum(
            constraints[(v, w)]
            @ constraints[(w, x)]
            @ constraints[(x, y)]
            @ constraints[(y, z)]
        )
        scores[(v, w, x, y, z)] = score
        if verbose:
            print(score)
    return scores


perms_list_6 = [
    (0, 1, 2, 3, 4),
    (3, 2, 1, 0, 4),
    (4, 1, 2, 0, 3),
    (0, 2, 1, 3, 4),
    (3, 1, 2, 0, 4),
    (4, 2, 1, 0, 3),
]


perms_list_6b = [
    (0, 2, 3, 4, 1),
    (3, 2, 1, 0, 4),
    (4, 3, 1, 2, 0),
    (0, 1, 3, 2, 4),
    (1, 2, 3, 4, 0),
    (4, 2, 1, 3, 0),
]

perms_list_6c = [
    (0, 1, 2, 3, 4),
    (0, 4, 2, 1, 3),
    (1, 4, 2, 3, 0),
    (2, 4, 1, 0, 3),
    (3, 2, 1, 4, 0),
    (3, 4, 1, 2, 0),
]


perms_list_15 = [
    (0, 1, 2, 3, 4),
    (3, 1, 4, 0, 2),
    (2, 1, 0, 3, 4),
    (4, 1, 0, 2, 3),
    (2, 4, 0, 1, 3),
    (3, 0, 4, 1, 2),
    (3, 2, 4, 0, 1),
    (0, 4, 2, 1, 3),
    (1, 4, 2, 0, 3),
    (0, 3, 2, 1, 4),
    (1, 0, 2, 3, 4),
    (1, 3, 2, 0, 4),
    (2, 3, 0, 1, 4),
    (4, 2, 0, 1, 3),
    (4, 3, 0, 1, 2),
]


def apply_perm(perm, iterable):
    return [iterable[p] for p in perm]


def invert_perm(perm):
    inv_perm = [perm.index(i) for i in range(len(perm))]
    return tuple(inv_perm)


def aggregate_scores(*args):
    aggregator = dict()
    for arg in args:
        for x in arg:
            if x in aggregator:
                aggregator[x] += arg[x]
            else:
                aggregator[x] = arg[x]
    return aggregator


def normalize_score(score):
    mu = np.mean(np.array(list(score.values())))
    normalized_score = {x: score[x] - mu for x in score}
    return normalized_score


def max_norm(score):
    return max([abs(score[x]) for x in score])


def l0_norm(score):
    return sum(True ^ np.isclose(np.array(list(score.values())), 0))


def l1_norm(score):
    return sum([abs(score[x]) for x in score])


def l2_norm(score):
    return np.sqrt(sum([score[x] ** 2 for x in score]))


def l4_norm(score):
    return (sum([score[x] ** 4 for x in score])) ** (1 / 4)


def rotl(x, i):
    return x[i:] + x[:i]


def rotr(x, i):
    return x[-i:] + x[:-i]


lift_23_perms = [
    (0, 1, 2, 3, 4),
    (3, 2, 1, 0, 4),
    (4, 1, 2, 0, 3),
    (0, 2, 1, 3, 4),
    (3, 1, 2, 0, 4),
    (4, 2, 1, 0, 3),
]

lift_34_perms = [
    [(0, 1, 2, 3, 4), (4, 2, 1, 3, 0)],
    [(0, 1, 2, 3, 4), (3, 2, 1, 0, 4)],
    [(0, 1, 2, 3, 4), (4, 2, 1, 3, 0)],
    [(0, 1, 2, 3, 4), (4, 2, 1, 3, 0)],
    [(0, 1, 2, 3, 4), (3, 2, 1, 0, 4)],
    [(0, 1, 2, 3, 4), (4, 2, 1, 3, 0)],
]

lift_45_perms = [
    (0, 1, 2, 3, 4),
    (0, 1, 3, 2, 4),
    (0, 1, 4, 2, 3),
    (2, 0, 1, 3, 4),
    (0, 2, 3, 1, 4),
    (0, 2, 4, 1, 3),
    (0, 3, 4, 1, 2),
    (1, 0, 3, 2, 4),
    (1, 0, 2, 3, 4),
    (1, 0, 4, 2, 3),
]

# ============================================================================

m = 5

dice_names = "abcde"
word_a = "abcdeedcba"
word_b = "abcdeedcba"
A = "".join([permute_letters(word_a, perm, relative=False) for perm in perms_list_6])
A_inv = A[::-1]
AA = "abcdeedcba"  # A + A_inv
AA = "abcdebcdeacdeabdeabceabcd"

all_perms = list(permutations(range(m)))
# roots = [AA, BB]

big_words = []
# for word_a in permutations(dice_names):
# AA = ''.join(word_a + word_a[::-1])
foo = []
bar = []
candidates = dict()
for root in tqdm([AA]):  # roots):  # + [BB]):
    foo.extend([permute_letters(root, perm) for perm in all_perms])
    perm_words = {(root, perm): permute_letters(root, perm) for perm in all_perms}
    perm_scores = {
        (root, perm): score_perm5s(perm_words[(root, perm)]) for perm in all_perms
    }
    candidates.update(
        {
            (root, perm): normalize_score(perm_scores[(root, perm)])
            for perm in all_perms
            if permute_letters(root, perm) not in bar
        }
    )
    bar.extend([permute_letters(root, perm) for root, perm in candidates])


used_perms = [(AA, tuple(range(m)))]
current_score = normalize_score(score_perm5s(used_perms[0][0]))
current_norm = l2_norm(current_score)
print(current_norm)
while not np.isclose(current_norm, 0, atol=0.05):
    next_scores = {
        k: aggregate_scores(current_score, c)
        for k, c in candidates.items()
        if k not in used_perms
    }
    next_norms = {k: l2_norm(v) for k, v in next_scores.items()}
    next_perms = sorted(next_norms.items(), key=lambda x: x[1])
    # next_perm = random.choice([x for x in next_perms if x[0] == next_perms[0][0]])
    next_perm = next_perms[0]
    # if l2_norm(next_scores[next_perm]) <= current_norm:
    used_perms.append(next_perm[0])
    current_score = next_scores[next_perm[0]]
    current_norm = l2_norm(current_score)
    print(current_norm)
    # if len(used_perms) == 2:
    #     break


big_word = "".join([permute_letters(word, perm) for word, perm in used_perms])
big_words.append(big_word)

check = score_perm4s(big_word, verbose=True)


word = "abcdeedcba"
foo = [permute_letters(word, perm) for perm in perms_list_6c]
# foo = [permute_letters(word, perm) for perm in perms_list_15]
bar = "".join(foo + foo[::-1])

counts = score_perm5s(bar)
test = normalize_score(counts)
blarg = {x: v // 16 for x, v in test.items()}

for x in permutations(dice_names):
    print(sum([counts[rotl(x, i)] for i in range(m)]))


words = [rotl("abcde", i) + rotr("edcba", i) for i in range(5)]
# words += [rotl("edcba", i) + rotr("abcde", i) for i in range(5)]
#
# words += [rotl("acebd", i) + rotr("dbeca", i) for i in range(5)]
# words += [rotl("dbeca", i) + rotr("acebd", i) for i in range(5)]
#
# words = [rotl("aedcb", i) + rotr("bcdea", i) for i in range(5)]
# words += [rotl("bcdea", i) + rotr("aedcb", i) for i in range(5)]
#
# words += [rotl("adbec", i) + rotr("cebda", i) for i in range(5)]
# words += [rotl("cebda", i) + rotr("adbec", i) for i in range(5)]
#
# words = [rotl("abced", i) + rotr("decba", i) for i in range(5)]
# words += [rotl("abecd", i) + rotr("dceba", i) for i in range(5)]
# words += [rotl("acedb", i) + rotr("bdeca", i) for i in range(5)]
# words += [rotl("adbec", i) + rotr("cebda", i) for i in range(5)]

foos = [[permute_letters(word, perm) for perm in perms_list_6] for word in words]
bars = ["".join(f + f[::-1]) for f in foos]
# bars = ["".join(f) for f in foos] + ["".join(f[::-1]) for f in foos[::-1]]
ram = "".join(bars)
counts = score_perm5s(ram)

roots = list(permutations(dice_names))
families = dict()
for x in roots:
    y = "".join(x)
    z = frozenset(
        [rotl(x, i) for i in range(5)] + [rotr(x[::-1], 4 - i) for i in range(5)]
    )
    if not z in families:
        families[z] = set()

for z in families:
    for x in z:
        families[z].add(counts[x])

for z in families:
    print(z)
    print(families[z])
    print()


rev_families = {frozenset(v): [] for v in families.values()}
for k, v in families.items():
    rev_families[frozenset(v)].append(k)


rot_perms = [rotl((0, 1, 2, 3, 4), i) for i in range(5)]
# rot_perms += [rotl((0,2,4,1,3), i) for i in range(5)]
rot_perms += [x[::-1] for x in rot_perms]
test = [permute_letters("abcde", perm, relative=False) for perm in rot_perms]
for i, t in enumerate(test):
    print(i, rot_perms[i], t, counts[tuple(t)])


hits = dict()
for x in permutations(dice_names):
    hits[x] = 0
    for i in range(len(ram) - 4):
        if "".join(x) == ram[i : i + 5]:
            hits[x] += 1

for x in hits:
    temp = [hits[rotl(x, i)] for i in range(5)]
    if len(set(temp)) == 1:
        print(x, temp)
print()

target = ((len(ram) // 5) ** 5) / 120
for x in counts:
    if counts[x] == target:
        print(x, counts[x])
print()

no_revs = set()
for x in permutations(dice_names):
    if x[::-1] not in no_revs:
        no_revs.add(x)

iterable = list(combinations(no_revs, 4))
iterable = [tuple([rotl("abcde", i) for i in range(1, 5)])]
hit_counts = dict()
for xs in tqdm(iterable):
    words = ["edcbaabcde"] + ["".join(x) + "".join(x[::-1]) for x in xs]
    foos = [[permute_letters(word, perm) for perm in perms_list_6] for word in words]
    bars = ["".join(f + f[::-1]) for f in foos]
    ram = "".join(bars)
    target = ((len(ram) // 5) ** 5) / 120
    counts = score_perm5s(ram)
    if target in counts.values():
        counter = Counter(counts.values())
        if counter[target] >= 20:
            print("hit")
            print(xs)
            hit_counts[xs] = counts
            print(Counter(counts.values()))


def rotate_segments(word, m, x):
    num_segments = len(word) // (2 * m)
    rotated_segments = []
    for i in range(num_segments):
        first = word[(2 * m * i) : (2 * m * i + m)]
        second = word[(2 * m * i + m) : (2 * m * i + 2 * m)]
        rot_first = rotl(first, x)
        rot_second = rotr(second, x)
        rotated_segments.append(rot_first + rot_second)
    return rotated_segments


bar2 = [rotate_segments(bar, m, x) for x in range(5)]
counts2 = score_perm5s(bar2)

bar2 = rotl(bar, 1)
counts2 = score_perm5s(bar2)

perms_list_6_2 = [rotl(p, 1) for p in perms_list_6]
word2 = "eabcddcbae"
foo2 = [permute_letters(word2, perm) for perm in perms_list_6_2]
bar2 = "".join(foo2 + foo2[::-1])
counts2 = score_perm5s(bar2)

word2 = "abcdebcdeacdeabdeabceabcd"
word2 = word2 + word2[::-1]
foo2 = [permute_letters(word2, perm) for perm in perms_list_6]


target = {x: counts[rotl(x, 1)] for x in counts}
all_perms = list(permutations(range(m)))
for p in tqdm(all_perms):
    word2 = permute_letters(word, p)
    foo2 = [permute_letters(word, perm) for perm in perms_list_6c]
    seg_perms = list(permutations(foo2))
    for f in tqdm(seg_perms):
        bar2 = "".join(f + f[::-1])
        for q in all_perms:
            counts2 = score_perm5s(permute_letters(bar2, q))
            if counts2 == target:
                print("hit!")
                print(p, f, q)


bar2 = "".join(foo)
perms_list_10 = [
    (1, 0, 2, 3, 4),
    (2, 1, 0, 3, 4),
    (3, 1, 2, 0, 4),
    (4, 1, 2, 3, 0),
    (0, 2, 1, 3, 4),
    (0, 3, 2, 1, 4),
    (0, 4, 2, 3, 1),
    (0, 1, 3, 2, 4),
    (0, 1, 4, 3, 2),
    (0, 1, 2, 4, 3),
]
bar3s = [permute_letters(bar2, perm) for perm in perms_list_10]


counts = score_perm5s(bar)
hit_groups = [tuple([x for x in counts if counts[x] == v]) for x, v in counts.items()]


qerms = []
for q in permutations(range(5)):
    if q[::-1] not in qerms:
        qerms.append(q)

big_word = "".join([permute_letters(bar, q, relative=False) for q in qerms])

dice_names = "abcde"
lut = {}
for x in permutations(dice_names):
    temp = [dice_names.index(i) for i in x]
    # print(permute_letters(temp, x))
    lut[x] = temp


used = []
qerms = []
qerms2 = []
for x in lut:
    if x[::-1] not in used:
        qerms.append(lut[x])
        qerms2.append(lut[x[::-1]])
        used.append(x)

big_word = "".join([permute_letters(bar, q) for q in qerms])


bar2 = permute_letters(bar, (0, 1, 2, 4, 3))

ram = [permute_letters(bar, perm) for perm in perms_list_15]
big_word = "".join(sum([r + r[::-1] for r in ram], []))


# ============================================================================

word = "abcdeedcba"
foo = [permute_letters(word, perm) for perm in lift_23_perms]

foo2 = ["".join(foo[i:] + foo[:i]) for i in range(6)]

bars = ["".join(foo2[i]) for i in range(6)]
rams = [
    "".join([permute_letters(bars[i], perm) for perm in lift_34_perms[i][1:]])
    for i in range(6)
]

ram = "".join(rams)

sol = "".join(permute_letters(ram, perm) for perm in lift_45_perms)
counts = score_perm5s(sol)

for x in counts:
    print(x, counts[x])
