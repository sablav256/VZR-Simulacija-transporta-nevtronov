import numpy as np
from numba import njit
import time


# Numba - optimizirana Monte Carlo simulacija transporta nevtronov
# Simulacija nevtronov znotraj enega procesa
@njit
def simulate_batch(n, thickness, mean_free_path, absorption_prob):

    # začetne vrednosti za opazovanje izida nevtronov
    absorbed = 0
    transmitted = 0
    reflected = 0

    # simulacija posameznih nevtronov
    for _ in range(n):

        # začetni položaj nevtrona
        x = 0.0

        # začetna smer gibanja (+x smer)
        angle = 0.0

        while True:

            # generiranje proste poti
            # eksponentna porazdelitev
            step = -mean_free_path * np.log(np.random.random())

            # premik nevtrona v trenutni smeri
            x += step * np.cos(angle)

            # TRANSMISIJA
            # nevtron zapusti material
            if x >= thickness:
                transmitted += 1
                break

            # REFLEKSIJA
            # nevtron se odbije nazaj
            if x <= 0:
                reflected += 1
                break

            # ABSORPCIJA
            # material absorbira nevtron
            if np.random.random() < absorption_prob:
                absorbed += 1
                break

            # SIPANJE
            # nova naključna smer gibanja
            angle = -np.pi + 2 * np.pi * np.random.random()

    return absorbed, transmitted, reflected


if __name__ == "__main__":

    # parametri materiala
    # debelina materiala
    thickness = 10.0
    # povprečna prosta pot nevtrona (razdalja med trki)
    mean_free_path = 1.0
    # verjetnost absorpcije pri posameznem trku
    absorption_prob = 0.25

    # število simuliranih nevtronov
    n = 1000000

    start = time.time()

    result = simulate_batch(
        n,
        thickness,
        mean_free_path,
        absorption_prob
    )

    end = time.time()

    print("RESULT:", result)
    print("TIME:", end - start)