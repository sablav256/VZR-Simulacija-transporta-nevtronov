import pandas as pd
import matplotlib.pyplot as plt

# branje rezultatov benchmark meritev
df = pd.read_csv("times.csv", header=None)
df.columns = ["cores", "time"]

# urejanje po številu procesov
# pomembno za pravilni prikaz grafov
df = df.sort_values("cores")

# referenčni čas izvajanja na enem procesu
T1 = df[df["cores"] == 1]["time"].values[0]

# SPEEDUP:
# S(p) = T1 / Tp
df["speedup"] = T1 / df["time"]

# EFFICIENCY:
# E(p) = S(p) / p
df["efficiency"] = df["speedup"] / df["cores"]


# KARP–FLATT METRIKA
# ocenjuje delež sekvenčnega dela programa
def karp_flatt(p, S):
    return ((1 / S) - (1 / p)) / (1 - 1 / p)


df["karp_flatt"] = df.apply(
    lambda row: karp_flatt(row["cores"], row["speedup"]),
    axis=1
)

# SPEEDUP GRAF
plt.figure()

plt.plot(
    df["cores"],
    df["speedup"],
    marker="o"
)

plt.title("Speedup MPI simulacije")
plt.xlabel("Število procesov")
plt.ylabel("Speedup")

plt.grid(True)
plt.tight_layout()

plt.savefig("speedup.png")
plt.show()

# EFFICIENCY GRAF
plt.figure()

plt.plot(
    df["cores"],
    df["efficiency"],
    marker="o"
)

plt.title("Efficiency MPI simulacije")
plt.xlabel("Število procesov")
plt.ylabel("Efficiency")

plt.grid(True)
plt.tight_layout()

plt.savefig("efficiency.png")
plt.show()

# KARP–FLATT GRAF
plt.figure()

plt.plot(
    df["cores"],
    df["karp_flatt"],
    marker="o"
)

plt.title("Karp–Flatt parameter")
plt.xlabel("Število procesov")
plt.ylabel("e(p)")

plt.grid(True)
plt.tight_layout()

plt.savefig("karp_flatt.png")
plt.show()

# izpis rezultatov v tabeli
print(df)