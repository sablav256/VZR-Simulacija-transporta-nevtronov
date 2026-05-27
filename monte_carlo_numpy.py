import numpy as np
import time
import csv
import os


L = 10.0
sigma_t = 1.0
p_abs = 0.7
N = 100000


def save_result(method, cores, t):
    file_exists = os.path.isfile("times.csv")

    with open("times.csv", "a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["method", "cores", "time"])

        writer.writerow([method, cores, t])


def simulate_numpy(N, L, sigma_t, p_abs):

    x = np.zeros(N)
    alive = np.ones(N, dtype=bool)

    transmitted = 0
    absorbed = 0
    reflected = 0

    while np.any(alive):

        idx = np.where(alive)[0]

        s = -np.log(np.random.rand(len(idx))) / sigma_t
        direction = np.random.choice([-1, 1], size=len(idx))

        x[idx] += s * direction

        out_right = x[idx] >= L
        out_left = x[idx] <= 0

        transmitted += np.sum(out_right)
        reflected += np.sum(out_left)

        alive[idx[out_right | out_left]] = False

        still_inside = ~(out_right | out_left)
        inside_idx = idx[still_inside]

        if len(inside_idx) > 0:
            absorb_event = np.random.rand(len(inside_idx)) < p_abs
            absorbed += np.sum(absorb_event)
            alive[inside_idx[absorb_event]] = False

    return transmitted, absorbed, reflected


if __name__ == "__main__":

    start = time.time()
    T, A, R = simulate_numpy(N, L, sigma_t, p_abs)
    end = time.time()

    print("NUMPY RESULTS")
    print("TRANSMISSION:", T / N)
    print("ABSORPTION:", A / N)
    print("REFLECTION:", R / N)
    print("TIME:", end - start)

    save_result("numpy", 1, end - start)
