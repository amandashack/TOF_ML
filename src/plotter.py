import math
import matplotlib.pyplot as plt
import numpy as np

def plot_opt(ax_nm):#lbl_x,lbl_y
    ax_nm.axhline(y = 0, color = 'b', label = 'E_F',linestyle='--'); ax_nm.axvline(0,color='black',linestyle = '--')
    ax_nm.yaxis.set_ticks_position('both'); ax_nm.xaxis.set_ticks_position('both')
    ax_nm.set_xlabel('Kx (deg)',fontsize=16); ax_nm.set_ylabel('Kz (deg)',fontsize=16)
    fig.patch.set_facecolor('white'); fig.patch.set_alpha(0.95)


def get_cmap(n, name='seismic'):
    '''Returns a function that maps each index in 0, 1, ..., n-1 to a distinct
    RGB color; the keyword argument name must be a standard mpl colormap name.'''
    return plt.cm.get_cmap(name, n)


def multi_plot(spec_dict):
    n = len(spec_dict.keys())
    h = math.ceil(n/3)
    row = math.ceil(n/3)
    f = plt.figure(figsize=(15,5*row))
    for i in range(1, n+1):
        keys = list(spec_dict.keys())
        spec = spec_dict[keys[i-1]]
        ax = f.add_subplot(row, 3, i)
        spec.plot(x='slit', y='photon_energy', ax=ax, add_colorbar=False, cmap='viridis')
        ax.set_xlabel('$k_{x}$ ($\AA^{-1}$)', fontsize=18)
        ax.set_ylabel('hv', fontsize=18)
        ax.set_title(keys[i-1], fontsize=20)
    plt.tight_layout()
    plt.show()


def one_plot_multi_scatter(ax, spec_dict, multi_title, xlabel, ylabel, logarithm=True, fit=True):
    cmap = get_cmap(len(spec_dict.keys()))
    for i in range(len(spec_dict.keys())):
        dvals = list(spec_dict.keys())
        xx = spec_dict[dvals[i]][0]
        yy = spec_dict[dvals[i]][1]
        if logarithm:
            ax.scatter(x=np.log2(xx), y=np.log2(yy), color = cmap(i), label=f'Retardation: {dvals[i]}')
        else:
            ax.scatter(x=xx, y=yy, color=cmap(i), label=f'Retardation: {dvals[i]}')
    if fit:
        X1 = np.log2(spec_dict[0][0][5])
        X2 = np.log2(spec_dict[0][0][-5])
        Y1 = np.log2(spec_dict[0][1][5])
        Y2 = np.log2(spec_dict[0][1][-5])
        slope = (Y2 - Y1)/(X2 - X1)
        b = Y1 - X1*slope

        Xmin = np.log2(np.min(spec_dict[0][0]))
        Xmax = np.log2(np.max(spec_dict[0][0]))
        XX = np.linspace(Xmin, Xmax, num = 10)
        YY = slope*XX + b
        ax.scatter(x=XX, y=YY, color="orange", label=f'Fit: y = {slope:.4g}x+{b:.4g}')
    ax.set_title(multi_title, fontsize=18)
    ax.set_xlabel(xlabel, fontsize=16)
    ax.set_ylabel(ylabel, fontsize=16)
    return(ax)