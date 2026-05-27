import pandas as pd
import matplotlib.pyplot as plt

# branje CSV
df = pd.read_csv("times.csv")

# odstrani presledke (če jih imaš)
df.columns = [c.strip() for c in df.columns]


# MPI DATA
mpi = df[df["method"] == "mpi"].copy()
mpi = mpi.sort_values("cores")

T1_mpi = mpi[mpi["cores"] == 1]["time"].values[0]

mpi["speedup"] = T1_mpi / mpi["time"]
mpi["efficiency"] = mpi["speedup"] / mpi["cores"]

def karp_flatt(p, S):
    if p == 1:
        return 0
    return ((1 / S) - (1 / p)) / (1 - 1 / p)

mpi["karp_flatt"] = mpi.apply(
    lambda r: karp_flatt(r["cores"], r["speedup"]),
    axis=1
)


# SPEEDUP MPI
plt.figure()
plt.plot(mpi["cores"], mpi["speedup"], marker="o")
plt.title("MPI Speedup")
plt.xlabel("Cores")
plt.ylabel("Speedup")
plt.grid()
plt.tight_layout()
plt.savefig("mpi_speedup.png")
plt.show()

# EFFICIENCY MPI
plt.figure()
plt.plot(mpi["cores"], mpi["efficiency"], marker="o")
plt.title("MPI Efficiency")
plt.xlabel("Cores")
plt.ylabel("Efficiency")
plt.grid()
plt.tight_layout()
plt.savefig("mpi_efficiency.png")
plt.show()


# KARP-FLATT MPI
plt.figure()
plt.plot(mpi["cores"], mpi["karp_flatt"], marker="o")
plt.title("MPI Karp-Flatt")
plt.xlabel("Cores")
plt.ylabel("f(p)")
plt.grid()
plt.tight_layout()
plt.savefig("mpi_karp_flatt.png")
plt.show()


# BAR GRAF: MPI + NUMBA + NUMPY + vrednosti
df_bar = pd.read_csv("times.csv")
df_bar = df_bar.dropna()
df_bar = df_bar.groupby(["method", "cores"], as_index=False)["time"].mean()
df_bar["label"] = df_bar["method"].str.upper() + " " + df_bar["cores"].astype(int).astype(str)
df_bar = df_bar.sort_values("time")

plt.figure(figsize=(10, 5))

bars = plt.bar(
    df_bar["label"],
    df_bar["time"]
)


# DODAMO VREDNOST NAD STOLPCE

for bar in bars:
    height = bar.get_height()
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        height,
        f"{height:.2f}s",
        ha="center",
        va="bottom",
        fontsize=9
    )

plt.title("Primerjava časa izvedb (log skala)")
plt.ylabel("Čas [s]")
plt.xticks(rotation=45)

plt.grid(axis="y", linestyle="--", alpha=0.5)

plt.tight_layout()
plt.savefig("bar_times.png")
plt.show()
