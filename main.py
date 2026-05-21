from mpi_simulation import run_simulation
from mpi4py import MPI

# Definicija materiala za simulacijo
material = {
    # debelina materiala
    "thickness": 10.0,
    # povprečna prosta pot nevtrona (razdalja med trki)
    "mean_free_path": 1.0,
    # verjetnost absorpcije pri posameznem trku
    "absorption_prob": 0.25
}

# skupno število simuliranih nevtronov
N = 100000


if __name__ == "__main__":

    # zagon MPI paralelizirane Monte Carlo simulacije
    result = run_simulation(N, material)

    # rezultate izpiše samo root proces
    if MPI.COMM_WORLD.Get_rank() == 0:

        absorbed, transmitted, reflected = result

        # skupno število simuliranih nevtronov
        total = absorbed + transmitted + reflected

        # relativne verjetnosti posameznih dogodkov
        print("TRANSMISSION:", transmitted / total)
        print("ABSORPTION:", absorbed / total)
        print("REFLECTION:", reflected / total)