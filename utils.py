def compare_dice(first, second):
    hits = 0
    for x in first:
        for y in second:
            if y < x:
                hits += 1
    return hits

def natural_faces(d, *args):
    temp = sum(args, [])
    n = len(args)
    all_faces = dict(zip(sorted(temp), range(len(temp))))
    ret = [[all_faces[args[i][j]] for j in range(d)] for i in range(n)]
    return ret

def collapse_values(dice):
    T = {}
    for i, die in enumerate(dice):
        T.update({k: i for k in die})
    n = len(T.keys())
    T_list = [T[i] for i in range(n)]
    current_value = 0
    current_die = T_list[0]
    collapsed_dice = [[] for _ in dice]
    collapsed_dice[current_die].append(current_value)
    for i in range(1, n):
        previous_die = current_die
        current_die = T_list[i]
        if current_die != previous_die:
            current_value += 1
        collapsed_dice[current_die].append(current_value)
    return collapsed_dice
