import matplotlib
import matplotlib.pyplot as plt
import math
import numpy as np

N_FPGA = 8

def main():
    labelsize = 10
    matplotlib.rc('font',**{
        'family':'sans-serif',
        'sans-serif':['Helvetica'],
    })
    matplotlib.rcParams.update({'font.size': labelsize, "font.family": "serif"})
    matplotlib.rc('xtick', labelsize=labelsize)
    matplotlib.rc('ytick', labelsize=labelsize)
    arr = []
    legends = []
    for num_used in reversed(range(1, N_FPGA + 1)):
        if num_used != N_FPGA and math.floor(N_FPGA / float(num_used)) == math.floor(N_FPGA / (num_used + 1.0)):
            continue
        x = np.array(range(0, int(((num_used ** 1.8) * 1000))))
        m = float(math.floor(N_FPGA / float(num_used)))
        y = x * m
        arr.append({"n": num_used, "x": x, "y": y, "m": m})
        legends.append("Used %d FPGA" % num_used)
    arr.reverse()
    legends.reverse()

    hs = []
    for a in arr:
        hs.append(plt.plot(a["x"], a["y"], label=" FPGA")[0])
    legends.append("Toxic $N_{in}^{(ref)}$ Values")
    hs.append(None)

    for l, r in zip(arr, arr[1:]):
        x_l = l["x"][-1]
        x_r = float(l["y"][-1] / r["m"])
        print x_l, x_r
        hs[-1] = plt.hlines(l["y"][-1], x_l, x_r, linestyle="--", color="#00000080")

    plt.grid()
    plt.title(r"$N_{in}^{(ref)}$ vs Effective Throughput (image per second) with %d available FPGAs" % N_FPGA)
    plt.ylabel(r"Effective throughput")
    plt.xlabel(r"$N_{in}^{(ref)}$")
    plt.legend(hs, legends)
    plt.show()

if __name__ == "__main__":
    main()
