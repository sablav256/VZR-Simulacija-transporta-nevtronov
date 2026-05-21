from mpi4py import MPI
import csv
from mpi_simulation import run_simulation


# Parametri materiala
material = {
    # debelina materiala
    "thickness": 10.0,
    # povprečna prosta pot nevtrona
    "mean_free_path": 1.0,
    # verjetnost absorpcije pri trku
    "absorption_prob": 0.25
}

# skupno število nevtronov
N = 200000

# inicializacija MPI komunikatorja
comm = MPI.COMM_WORLD

# ID trenutnega procesa
rank = comm.Get_rank()

# skupno število MPI procesov
size = comm.Get_size()


# funkcija za merjenje časa izvajanja
def measure():

    # sinhronizacija vseh procesov - istočasen začetek merjenja
    comm.Barrier()

    # začetek meritve časa
    start = MPI.Wtime()

    # zagon MPI simulacije
    run_simulation(N, material)

    # sinhronizacija po zaključku simulacije - istočasen konec merjenja
    comm.Barrier()

    # skupni čas izvajanja
    return MPI.Wtime() - start


if __name__ == "__main__":

    # seznam časov izvajanja
    results = []

    # večkrat ponovimo - zmanjšamo vpliva sistemskega šuma
    for _ in range(3):
        t = measure()

        # samo root proces shrani rezultate
        if rank == 0:
            results.append(t)

    # samo root proces zapisuje rezultate
    if rank == 0:

        # povprečni čas izvajanja
        avg = sum(results) / len(results)

        # zapis rezultatov v times.csv
        with open("times.csv", "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([size, avg])
        # izpis rezultata meritve
        print("AVG TIME:", avg)