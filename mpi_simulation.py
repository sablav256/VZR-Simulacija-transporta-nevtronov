from mpi4py import MPI
import numpy as np
from neutron_model import simulate_neutron

# MPI paralelizirana Monte Carlo simulacija transporta nevtronov skozi material.
# Vsak MPI proces simulira svoj del nevtronov neodvisno od ostalih procesov.
# Na koncu se rezultati združijo z MPI.Reduce().
def run_simulation(n_neutrons, material, seed=42):

    # inicializacija MPI komunikatorja
    comm = MPI.COMM_WORLD

    # ID trenutnega procesa
    rank = comm.Get_rank()
    # skupno število procesov
    size = comm.Get_size()

    # enakomerna porazdelitev dela med procese
    base = n_neutrons // size
    remainder = n_neutrons % size

    # prvih nekaj procesov dobi dodatni nevtron
    local_n = base + (1 if rank < remainder else 0)

    # vsak proces uporablja lasten RNG seed - neodvisnost simulacije
    rng = np.random.default_rng(seed + rank)

    # izid nevtronov: [absorbed, transmitted, reflected]
    local_results = np.zeros(3, dtype=int)

    # Monte Carlo simulacija nevtronov
    for _ in range(local_n):
        r = simulate_neutron(material, rng)
        local_results[r] += 1

    # globalno združevanje rezultatov
    #MPI.Reduce se izvede samo na root procesu
    global_results = np.zeros(3, dtype=int)

    comm.Reduce(
        local_results,
        global_results,
        op=MPI.SUM,
        root=0
    )

    # samo root proces vrne rezultate
    return global_results if rank == 0 else None