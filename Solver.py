import numpy as np
import sys

import numpy as np
import sys
import datetime


def read_input(input_file):
    line1 = input_file.readline()
    rows, cols, min_sum, max_size = [int(x) for x in line1.strip().split(' ')]
    pizza = np.ones((rows, cols), dtype=np.int)
    for i, line in enumerate(input_file.readlines()):
        pizza[i, :] = np.array([int(c != 'T') for c in line.strip()])
    return min_sum, max_size, pizza


def fast_sum(summed, x, y, w, h):
    """
    Compute the sum in a given slice, using a summed area table.
    (not very useful for small slices...)
    """
    r, c = summed.shape
    if not (0 <= x + w - 1 < c and 0 < y + h - 1 < r):
        return -1
    D = summed[y + h - 1, x + w - 1] if 0 <= y + h - 1 < r and 0 <= x + w - 1 < c else 0
    B = summed[y - 1, x + w - 1] if 0 <= y - 1 < r and 0 <= x + w - 1 < c else 0
    C = summed[y + h - 1, x - 1] if 0 <= y + h - 1 < r and 0 <= x - 1 < c else 0
    A = summed[y - 1, x - 1] if 0 <= y - 1 < r and 0 <= x - 1 < c else 0
    return D + A - B - C


def score(pizza, solution, min_sum, max_size):
    """
    Compute the score for a given solution (just to check)
    :return: -1 if solution is not valid, the score otherwise.
    """
    summed_area_table = pizza.cumsum(axis=0).cumsum(axis=1)
    occupied = np.zeros(pizza.shape, dtype=np.int)
    total_score = 0
    for (x, y, w, h) in solution:
        if w * h > max_size:
            print("Slice is too big: ", x, y, w, h)
            return -1

        n = fast_sum(summed_area_table, x, y, w, h)
        if n < min_sum:
            print(" Slice is not valid: ", x, y, w, h, n)
            return -1
        if occupied[y:y + h, x:x + w].sum() != 0:
            print("Slice intersect: ", x, y, w, h)
            return -1
        occupied[y:y + h, x:x + w] = 1
        total_score += w * h
    return total_score


def generate_slices(pizza, min_sum, max_size):
    """
    Generate all valid slices.
    """
    summed_area_table = pizza.cumsum(axis=0).cumsum(axis=1)
    r, c = pizza.shape
    for y in range(r):
        for x in range(c):
            w_max = min(c - x, max_size)
            for w in range(1, w_max + 1):
                h_max = min(max_size // w, r - y)
                for h in range(1, h_max + 1):
                    n_ham = fast_sum(summed_area_table, x, y, w, h)
                    if h * w <= max_size and n_ham >= min_sum:
                        n_tomato = h * w - n_ham
                        yield (n_tomato, x, y, w, h)


def greedy(pizza, min_sum, max_size, n_iter):
    """
    Random greedy solver.
    Generate all pairs, shuffle them and add them in order.
    We do this n_iter times, and keep the best solution.

    :return: best_score and solution found (as a List of slice)
    """
    possibles = [s for s in generate_slices(pizza, min_sum, max_size)]
    best_score = 0
    best_solution = []
    for _ in range(n_iter):
        possibles = np.random.permutation(possibles)
        occupied = np.zeros(pizza.shape, dtype=np.int)
        solution = []
        current_score = 0
        for x, y, w, h in possibles:
            if occupied[y:y + h, x:x + w].sum() == 0:
                solution.append((x, y, w, h))
                occupied[y:y + h, x:x + w] = 1
                current_score += w * h
        if current_score > best_score:
            best_score = current_score
            best_solution = list(solution)
        elif current_score == best_score and len(solution) < len(best_solution):
            best_solution = list(solution)
    return best_score, best_solution


def greedy_tomato(pizza, min_sum, max_size):
    """
    """
    possibles = [s for s in generate_slices(pizza, min_sum, max_size)]
    possibles = sorted(possibles)[::-1]

    occupied = np.zeros(pizza.shape, dtype=np.int)
    solution = []
    current_score = 0
    for _, x, y, w, h in possibles:
        if occupied[y:y + h, x:x + w].sum() == 0:
            solution.append((x, y, w, h))
            occupied[y:y + h, x:x + w] = 1
            current_score += w * h

    return current_score, solution


def greedy_add(occupied, possibles, old, score):
    new_slices = []
    new_score = score
    for (_, x, y, w, h) in possibles:
        if (x, y, w, h) != old and occupied[y:y + h, x:x + w].sum() == 0:
            new_slices.append((x, y, w, h))
            occupied[y:y + h, x:x + w] = 1
            new_score += w * h
    return new_score, new_slices


def hillclimbing(solution, original_score, occupied, possibles, tmax_s=600):
    start_search = datetime.datetime.now()
    while 1:
        permuted_ids = np.random.permutation(len(solution))
        solution = [solution[r] for r in permuted_ids]
        possibles_ids = np.random.permutation(len(possibles))
        possibles = [possibles[r] for r in possibles_ids]

        i = 0
        while i < len(solution):
            x, y, w, h = solution[i]
            occupied_test = occupied.copy()
            # remove the slice
            occupied_test[y:y + h, x:x + w] = 0
            # add new slices greedily
            old_score = original_score - w * h
            score, new_slices = greedy_add(occupied_test, possibles, solution[i], old_score)
            if score > original_score:
                original_score = score
                solution = solution[:i] + new_slices + solution[i + 1:]
                occupied = occupied_test
                i += len(new_slices)
                #print(original_score)
            else:
                i += 1
            now = datetime.datetime.now()
            if (now - start_search).total_seconds() > tmax_s:
                return original_score, solution
    return original_score, solution


def print_solution_google(solution, out_file):
    out_file.write(str(len(solution)) + "\n")
    for x, y, w, h in solution:
        out_file.write("{x0} {y0} {x1} {y1}\n".format(x0=x, y0=y, x1=x + w - 1, y1=y + h - 1))


def print_solution_primers(solution, out_file):
    for x, y, w, h in solution:
        out_file.write("{x0},{y0},{w},{h}\n".format(x0=x, y0=y, w=w, h=h))


def solve(in_file, out_file, n_iter, print_solution=print_solution_google):
    min_sum, max_size, pizza = read_input(in_file)
    best_score, solution = greedy_tomato(pizza, min_sum, max_size)
    r, c = pizza.shape
    max_score = r * c
    sys.stderr.write("{s}/{m}".format(s=best_score, m=max_score))
    print_solution(solution, out_file)
    assert (best_score == score(pizza, solution, min_sum, max_size))


def solve_hillclimbing(in_file, out_file, time_max_s, print_solution=print_solution_primers):
    min_sum, max_size, pizza = read_input(in_file)
    occupied = np.zeros(pizza.shape, dtype=np.int)
    possibles = [s for s in generate_slices(pizza, min_sum, max_size)]
    possibles = sorted(possibles)[::-1]

    # Greedy with max tomato first
    original_score, original_solution = greedy_add(occupied, possibles, None, 0)

    best_score, best_solution = hillclimbing(original_solution, original_score, occupied, possibles, time_max_s)

    r, c = pizza.shape
    max_score = r * c
    sys.stderr.write("{s}/{m}".format(s=best_score, m=max_score))
    print_solution(best_solution, out_file)
    assert (best_score == score(pizza, best_solution, min_sum, max_size))


if __name__ == '__main__':
    solve_hillclimbing(sys.stdin, sys.stdout, 60, print_solution_primers)
