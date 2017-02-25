import numpy as np
from multiprocessing import Pool
from datetime import datetime


def read_input(input_file):
    line1 = input_file.readline()
    rows, cols, min_sum, max_size = [int(x) for x in line1.strip().split(' ')]
    pizza = np.ones((rows, cols), dtype=np.int)
    for i, line in enumerate(input_file.readlines()):
        pizza[i, :] = np.array([int(c == 'T') for c in line.strip()])
    return min_sum, max_size, pizza


def get_summed_area(a):
    """
    Return summed table area for a.
    :param a: ndarray of size h, w
    :return: s ndarray of size h + 1, w + 1 such as
        s[i, j] = sum(a[l, k] for 0 <= l <= i, 0 <= k <= j)
    """
    s = a.cumsum(axis=0).cumsum(axis=1)
    s1 = np.vstack((np.zeros(s.shape[1], dtype=np.int), s))
    s2 = np.hstack((np.zeros((s1.shape[0], 1), dtype=np.int), s1))
    return s2


def fast_sum(summed, x, y, w, h):
    """
    Compute the sum in a given slice, using a summed area table.
    """
    D = summed[y + h, x + w]
    B = summed[y, x + w]
    C = summed[y + h, x]
    A = summed[y, x]
    return D + A - B - C


def check_score(pizza, solution, min_sum, max_size):
    """
    Compute the score for a given solution (just to check)
    :return: -1 if solution is not valid, the score otherwise.
    """
    summed_area_table = get_summed_area(pizza)
    occupied = np.zeros(pizza.shape, dtype=np.int)
    total_score = 0
    for (x, y, w, h) in solution:
        if w * h > max_size:
            print("Slice is too big: ", x, y, w, h)
            return -1

        nHam= fast_sum(summed_area_table, x, y, w, h)
        nMushrooms = w * h - nHam
        if nHam < min_sum or nMushrooms < min_sum:
            print(" Slice is not valid: ", x, y, w, h, nHam, nMushrooms)
            return -1
        if occupied[y:y + h, x:x + w].sum() != 0:
            print("Slice intersect: ", x, y, w, h)
            return -1
        occupied[y:y + h, x:x + w] = 1
        total_score += w * h
    return total_score


def gen_block_with_params(pizza, min_sum, max_size, block_size=(60, 60)):
    """
    Split the pizza in block and yield the block and the parameter for the
    guillotine solver.
    """
    h, w = pizza.shape
    n_blocks_hori = int(np.ceil(w/block_size[1]))
    n_blocks_vert = int(np.ceil(h/block_size[0]))
    for bhori in range(n_blocks_hori):
        for bvert in range(n_blocks_vert):
            block = pizza[bvert*block_size[0]:(bvert + 1)*block_size[0],
                          bhori*block_size[1]:(bhori + 1)*block_size[1]]
            yield (block, min_sum, max_size, bhori*block_size[1], bvert*block_size[0])


def translate_solution(sol, dx, dy):
    """
    Translate the solution of a sub_block starting at (dx, dy)
    to the right coordinate in the full pizza
    """
    return [(x + dx, y + dy, w, h) for (x, y, w, h) in sol]


def get_subregions(r, c):
    """
    Get all possible subregions in a pizza of shape (r, c)
    """
    return [(w * h, i, j, w, h) for i in range(r)
            for j in range(c)
            for h in range(1, r - i + 1)
            for w in range(1, c - j + 1)]


def guillotine_worker(params):
    """
    Iterate over region to find the best cut pattern.
    """
    pizza, min_sum, max_size, dx, dy = params
    print("Start block ({}, {})".format(dx, dy))
    r, c = pizza.shape
    # start = datetime.now()
    subarea_scores = dict()
    subarea_cuts = dict()
    summed_pizza = get_summed_area(pizza)

    sub_regions = sorted(get_subregions(r, c))
    print(len(sub_regions), "subregions to consider")
    for _, i, j, w, h in sub_regions:
        ham = fast_sum(summed_pizza, j, i, w, h)
        mushroom = w * h - ham
        if ham < min_sum or mushroom < min_sum:
            subarea_scores[i, j, w, h] = 0
            subarea_cuts[i, j, w, h] = 0
            continue

        area = w * h
        if area <= max_size:
            subarea_scores[i, j, w, h] = area
            subarea_cuts[i, j, w, h] = 0
            continue

        subarea_score = 0
        subarea_cut = 0
        for cut in range(1, w):  # vertical cut
            candidate_score = subarea_scores[i, j, cut, h] + subarea_scores[i, j + cut, w - cut, h]
            if candidate_score > subarea_score:
                subarea_score = candidate_score
                subarea_cut = cut
        for cut in range(1, h):  # horizontal_cut
            candidate_score = subarea_scores[i, j, w, cut] + subarea_scores[i + cut, j, w, h - cut]
            if candidate_score > subarea_score:
                subarea_score = candidate_score
                subarea_cut = -cut
        subarea_cuts[i, j, w, h] = subarea_cut
        subarea_scores[i, j, w, h] = subarea_score

    sol = []
    stack = [(0, 0, c, r)]
    # Go through all the subregions by decreasing size
    while stack:
        i, j, w, h = stack.pop()
        cut = subarea_cuts[i, j, w, h]
        if cut == 0:  # Undivisable subregion
            if subarea_scores[i, j, w, h] > 0:
                sol.append((j, i, w, h))
        elif cut > 0:  # Subregion divided horizontally
            stack.append((i, j, cut, h))
            stack.append((i, j + cut, w - cut, h))
        else:  # Subregion divided vertically
            stack.append((i, j, w, -cut))
            stack.append((i - cut, j, w, h + cut))
    return translate_solution(sol, dx, dy)


def solve_by_block_parallel(pizza, min_sum, max_size, block_size=(60, 60), n_workers=50):
    """
    Solve the problem by block.
    Each block in solve in a separate process.
    """
    print("Solve using {} workers".format(n_workers))
    start = datetime.now()
    pool = Pool(n_workers)
    params = gen_block_with_params(pizza, min_sum, max_size, block_size)

    result = pool.map(guillotine_worker, params)
    result = [s for sub_solution in result for s in sub_solution]
    pool.close()
    pool.join()
    score = check_score(pizza, result, min_sum, max_size)
    print("Total time: {}".format(datetime.now() - start))
    return score, result


def print_solution_google(solution, out_file):
    out_file.write(str(len(solution)) + "\n")
    for x, y, w, h in solution:
        out_file.write("{r1} {c1} {r2} {c2}\n".format(r1=y, c1=x, r2=y + h - 1, c2=x + w - 1))


if __name__ == '__main__':
    import os
    import sys
    in_filename = sys.argv[1]
    basename, _ = os.path.splitext(in_filename)
    out_filename = basename + ".out"

    with open(in_filename, "r") as in_file,\
         open(out_filename, "w") as out_file:
        min_sum, max_size, pizza = read_input(in_file)
        print("Solve problem with min_sum={} and max_size={}".format(min_sum, max_size))
        print("Pizza size is {}".format(pizza.shape))
        score, solution = solve_by_block_parallel(pizza, min_sum, max_size, (50, 50))
        print("Score: {}".format(score))
        print_solution_google(solution, out_file)
