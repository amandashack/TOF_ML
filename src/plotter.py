import math
import matplotlib.pyplot as plt
import numpy as np
import collections
from sklearn.linear_model import LinearRegression

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


def pass_versus_counts(spec, retardation):
    cmap = get_cmap(len(retardation))
    fig, ax = plt.subplots()
    for i in range(len(retardation)):
        c = cmap(i)
        r = retardation[i]
        counter_pass = collections.Counter(spec[r][0])
        ax.scatter(counter_pass.keys(), counter_pass.values(), color=c)
    ax.set_xlabel("pass_energy")
    ax.set_ylabel("counts")
    plt.show()


def one_plot_multi_scatter(ax, df, multi_title, xlabel, ylabel, logarithm=True, fit=True):
    cmap = get_cmap(len(df.keys()))
    params = {'mathtext.default': 'regular'}
    plt.rcParams.update(params)
    for i in range(len(df.keys())):
        dvals = list(df.keys())
        xx = df[dvals[i]]
        yy = df[dvals[i]][2]
        if logarithm:
            ax.scatter(x=np.log2(xx), y=np.log2(yy), color = cmap(i), label=f'Retardation: {dvals[i]}')
        else:
            ax.scatter(x=xx, y=yy, color=cmap(i), label=f'Retardation: {dvals[i]}')
    if fit:
        X_vals = np.log2(df[0][1][:]).reshape((-1, 1))
        Y_vals = np.log2(df[0][2][:])
        print(X_vals.shape, Y_vals.shape)
        model = LinearRegression().fit(X_vals, Y_vals)
        r_sq = "{:.5f}".format(model.score(X_vals, Y_vals))
        slope = model.coef_[0]
        intercept = model.intercept_

        Xmin = np.log2(np.min(df[0][1]))
        Xmax = np.log2(np.max(df[0][1]))
        XX = np.linspace(Xmin, Xmax, num = 10)
        YY = slope*XX + intercept
        ax.scatter(x=XX, y=YY, color="orange", label=f'Fit: y = {slope:.4g}x+{intercept:.4g}')
        ax.set_title(multi_title + f", $R^{2}$ = {r_sq}", fontsize=18)
    else:
        ax.set_title(multi_title, fontsize=18)
    ax.set_xlabel(xlabel, fontsize=16)
    ax.set_ylabel(ylabel, fontsize=16)
    return ax


def plot_residuals(ax, df, multi_title, xlabel, ylabel):
    params = {'mathtext.default': 'regular'}
    plt.rcParams.update(params)
    dvals = list(df.keys())
    xx = df[dvals[0]]
    yy = df[dvals[1]]
    ax.scatter(x=xx, y=yy)
    ax.set_title(multi_title, fontsize=18)
    ax.set_xlabel(xlabel, fontsize=16)
    ax.set_ylabel(ylabel, fontsize=16)


def heatmap_plot(m):
    print(m)
