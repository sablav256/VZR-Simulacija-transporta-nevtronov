from mpi4py import MPI
from mpi_simulation import run_simulation
import csv
import os

material = {
    "thickness": 10.0,
    "mean_free_path": 1.0,
    "absorption_prob": 0.25
}

N = 200000

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()


def measure():
    comm.Barrier()
    start = MPI.Wtime()

    run_simulation(N, material)

    comm.Barrier()
    return MPI.Wtime() - start


def save_result(method, cores, time):
    file_exists = os.path.isfile("times.csv")

    with open("times.csv", "a", newline="") as f:
        writer = csv.writer(f)

        if not file_exists:
            writer.writerow(["method", "cores", "time"])

        writer.writerow([method, cores, time])


if __name__ == "__main__":

    results = []

    for _ in range(3):
        t = measure()
        if rank == 0:
            results.append(t)

    if rank == 0:
        avg = sum(results) / len(results)

        save_result("mpi", size, avg)

        print("AVG TIME:", avg)
