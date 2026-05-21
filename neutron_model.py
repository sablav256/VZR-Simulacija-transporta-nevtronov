import numpy as np

# Monte Carlo simulacija transporta enega nevtrona skozi material
def simulate_neutron(material, rng):

    # parametri materiala
    thickness = material["thickness"]
    mean_free_path = material["mean_free_path"]
    absorption_prob = material["absorption_prob"]

    # začetni položaj nevtrona
    x = 0.0

    # začetna smer gibanja (+x smer)
    angle = 0.0

    while True:

        # generiranje dolžine proste poti
        # eksponentna porazdelitev
        step = -mean_free_path * np.log(rng.random())

        # premik nevtrona v trenutni smeri gibanja
        x += step * np.cos(angle)

        # TRANSMISIJA
        # nevtron zapusti material
        if x >= thickness:
            return 1

        # REFLEKSIJA
        # nevtron se odbije nazaj
        if x <= 0:
            return 2

        # ABSORPCIJA
        # material absorbira nevtron
        if rng.random() < absorption_prob:
            return 0

        # SIPANJE
        # nova smer gibanja po trku
        angle = rng.uniform(-np.pi, np.pi)