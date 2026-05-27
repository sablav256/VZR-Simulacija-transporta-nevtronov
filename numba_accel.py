import numpy as np
from numba import njit
import time
import csv
import os


@njit
def simulate_batch(n, thickness, mean_free_path, absorption_prob):

    absorbed = 0
    transmitted = 0
    reflected = 0

    for _ in range(n):

        x = 0.0
        angle = 0.0

        while True:

            step = -mean_free_path * np.log(np.random.random())
            x += step * np.cos(angle)

            if x >= thickness:
                transmitted += 1
                break

            if x <= 0:
                reflected += 1
                break

            if np.random.random() < absorption_prob:
                absorbed += 1
                break

            angle = -np.pi + 2 * np.pi * np.random.random()

    return absorbed, transmitted, reflected


def save_result(method, cores, t):
    file_exists = os.path.isfile("times.csv")

    with open("times.csv", "a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["method", "cores", "time"])

        writer.writerow([method, cores, t])


if __name__ == "__main__":

    start = time.time()

    result = simulate_batch(
        200000,
        10.0,
        1.0,
        0.25
    )

    end = time.time()

    print("RESULT:", result)
    print("TIME:", end - start)

    save_result("numba", 1, end - start)
