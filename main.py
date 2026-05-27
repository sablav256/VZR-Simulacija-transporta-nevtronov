from mpi_simulation import run_simulation
from mpi4py import MPI
import time
import csv
import os

material = {
    "thickness": 10.0,
    "mean_free_path": 1.0,
    "absorption_prob": 0.25
}

N = 100000


def save_result(method, cores, time_val):
    file_exists = os.path.isfile("times.csv")

    with open("times.csv", "a", newline="") as f:
        writer = csv.writer(f)

        if not file_exists:
            writer.writerow(["method", "cores", "time"])

        writer.writerow([method, cores, time_val])


if __name__ == "__main__":

    start = time.time()
    result = run_simulation(N, material)
    end = time.time()

    if MPI.COMM_WORLD.Get_rank() == 0:

        absorbed, transmitted, reflected = result
        total = absorbed + transmitted + reflected

        print("TRANSMISSION:", transmitted / total)
        print("ABSORPTION:", absorbed / total)
        print("REFLECTION:", reflected / total)
        print("TIME:", end - start)

        save_result("serial", 1, end - start)
