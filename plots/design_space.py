import matplotlib
import matplotlib.pyplot as plt
import math
import numpy as np

N_FPGA = 8

def main():
    labelsize = 10
    matplotlib.rc("text", usetex=True)
    matplotlib.rc('font',**{
        'family':'sans-serif',
        'sans-serif':['Helvetica'],
    })
    matplotlib.rcParams.update({'font.size': labelsize, "font.family": "serif"})
    matplotlib.rc('xtick', labelsize=labelsize)
    matplotlib.rc('ytick', labelsize=labelsize)

    fig, ax = plt.subplots(1)

    x = []
    y = []
    legends = []
    for p in range(0, 10 + 2, 2):
        x.append(p)
        y.append(p)

    x.append(20)
    y.append(10)

    x.append(25)
    y.append(8)

    x.append(31)
    y.append(7)

    plt.step(x, y, where="post")
    plt.axvline(x=10, linestyle="--", color="#00000070")
    plt.axhline(10, xmax=10, linestyle="--", color="#00000070")

    plt.axvline(x=25, linestyle="--", color="#00000070")
    plt.axhline(8, xmax=31, linestyle="--", color="#00000070")

    plt.ylim([0, 15])

    plt.grid()

    ax.set_xticks([10, 25])
    ax.set_xticklabels([r"$\mathcal{L}_1$", r"$\mathcal{L}_2$"])
    ax.set_yticks([14, 12, 10, 8])
    ax.set_yticklabels([r"$r_1$", r"$r_2$", r"$r_3$", r"$r_4$"])

    plt.title(r"$N_{in}^{(ref)}$ vs Effective Throughput (image per second) with using Fixed FPGAs")
    plt.ylabel(r"Effective throughput")
    plt.xlabel(r"$N_{in}^{(ref)}$")
    plt.show()

    # y = []
    # arr.reverse()
    # legends.reverse()

    # hs = []
    # for a in arr:
    #     hs.append(plt.plot(a["x"], a["y"], label=" FPGA")[0])
    # legends.append("Toxic $N_{in}^{(ref)}$ Values")
    # hs.append(None)

    # for l, r in zip(arr, arr[1:]):
    #     x_l = l["x"][-1]
    #     x_r = float(l["y"][-1] / r["m"])
    #     print x_l, x_r
    #     hs[-1] = plt.hlines(l["y"][-1], x_l, x_r, linestyle="--", color="#00000080")


if __name__ == "__main__":
    main()
