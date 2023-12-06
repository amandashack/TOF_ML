"""
Created on Oct 16 2023
this code opens .csv files or h5 files that have already been cleaned up

@author: Amanda
"""
import matplotlib.pyplot as plt
import numpy as np
import h5py
import os
import re
import copy
from sklearn.preprocessing import KBinsDiscretizer


def requires_invalid_indxs(func):
    def func_wrapper(*args, **kwargs):
        if getattr(args[0], 'invalid_indxs') is None:
            raise AttributeError("invalid indices are not defined yet.")
        return func(*args, **kwargs)
    return func_wrapper


def interpolator(tofs, pass_ens):
    min_pe = 0
    max_pe = 1200
    x_axis = np.linspace(min_pe, max_pe, num=30000)
    interp_y = []
    for i in range(len(tofs)):
        interp_y.append(np.interp(x_axis, pass_ens[i], tofs[i]))
    return np.array([y for y in interp_y]), x_axis


def train_test_val_loader(fp):
    df = np.array([])
    fp = os.path.abspath(fp)
    for file in os.listdir(fp):
        if file.endswith("h5"):
            path1 = os.path.join(fp, file)
            with h5py.File(path1, 'r') as f:
                df = np.append(df, f['data1']['elevation'][:])
                df = np.append(df, f['data1']['pass'][:])
                df = np.append(df, f['data1']['retardation'][:])
                df = np.append(df, f['data1']['ele*ret'][:])
                df = np.append(df, f['data1']['ele*pass'][:])
                df = np.append(df, f['data1']['pass*ret'][:])
                df = np.append(df, f['data1']['residuals'][:])
                df = np.reshape(df, (-1, len(f['data1']['elevation'][:])))
    return df


class MRCOLoader(object):
    """
    Implements data loading and storage for use plotting
    """

    _TOLERATED_EXTENSIONS = {
        ".h5"
    }

    def __init__(self, filepath: str):
        self.filepath = filepath
        self.dataframe = None
        self.data_masked = None
        self.spec_dict = {}
        self.spec_masked = {}
        self.mask = {}
        self.tof_values = []
        self.pass_energy = []
        self.initial_ke = []
        self.retardation = []
        self.elevation = []
        self.x_tof = []
        self.y_tof = []
        self.p_bins = []

    def load(self):
        fp = os.path.abspath(self.filepath)
        for file in os.listdir(fp):
            if file.endswith("h5"):
                path1 = os.path.join(fp, file)
                self.retardation.append(int(re.findall(r'_R(\d+)_', file)[0]))
                with h5py.File(path1, 'r') as f:
                    x_tof = f['data1']['x'][:]
                    y_tof = f['data1']['y'][:]
                    tof_values = f['data1']['tof'][:]
                    initial_ke = f['data1']['initial_ke'][:]
                    elevation = f['data1']['elevation'][:]
                    pass_energy = list(ke - self.retardation[-1] for ke in initial_ke)
                    self.initial_ke.append(initial_ke.tolist())
                    self.pass_energy.append(pass_energy)
                    self.tof_values.append(tof_values.tolist())
                    self.elevation.append(elevation.tolist())
                    self.x_tof.append(x_tof.tolist())
                    self.y_tof.append(y_tof.tolist())
                    self.spec_dict[self.retardation[-1]] = [self.elevation[-1],
                                                            self.pass_energy[-1],
                                                            self.tof_values[-1]]
            else:
                pass
        self.organize_data()

    def organize_data(self):
        tof_c = copy.deepcopy(self.tof_values)
        pass_c = copy.deepcopy(self.pass_energy)
        ele_c = copy.deepcopy(self.elevation)
        r_c = copy.deepcopy(self.retardation)
        ke_c = copy.deepcopy(self.initial_ke)
        x_tof = copy.deepcopy(self.x_tof)
        y_tof = copy.deepcopy(self.y_tof)
        s = sorted(range(len(r_c)),
                   key=r_c.__getitem__)
        tof_c = [tof_c[i] for i in s]
        pass_c = [pass_c[i] for i in s]
        ele_c = [ele_c[i] for i in s]
        r_c = [r_c[i] for i in s]
        ke_c = [ke_c[i] for i in s]
        x_tof = [x_tof[i] for i in s]
        y_tof = [y_tof[i] for i in s]
        self.dataframe = self.gen_dataframe(ele_c, pass_c, r_c, tof_c)
        self.spec_dict = self.gen_spec(ele_c, pass_c, r_c, tof_c)
        self.initial_ke = ke_c
        self.retardation = r_c
        self.pass_energy = pass_c
        self.tof_values = tof_c
        self.elevation = ele_c
        self.x_tof = x_tof
        self.y_tof = y_tof

    @staticmethod
    def gen_spec(elevation, pass_energy, retardation, tof):
        spec = {}
        for i in range(len(retardation)):
            spec[retardation[i]] = [elevation[i], pass_energy[i], tof[i]]
        return spec

    @staticmethod
    def gen_dataframe(elevation, pass_energy, retardation, tof):
        pass_flat = []
        tof_flat = []
        r_flat = []
        ele_flat = []
        for i in range(len(retardation)):
            r_flat += len(pass_energy[i]) * [retardation[i]]
            pass_flat += pass_energy[i]
            ele_flat += elevation[i]
            tof_flat += tof[i]
        data = np.array((np.asarray(ele_flat), np.log2(np.asarray(pass_flat)),
                         np.asarray(r_flat), np.log2(np.asarray(tof_flat))))
        return data

    def create_mask(self, x, y, min_pass, mask_name):
        # used to generate a mask for a part of the data
        # x and y are tuples with the first value being the minimum and second the maximum
        m = []
        for i in range(len(self.retardation)):
            xtof = np.asarray(self.x_tof[i])[:].astype(float)
            ytof = np.abs(np.asarray(self.y_tof[i])[:].astype(float))
            pass_en = np.asarray(self.pass_energy[i])[:].astype(float)
            xmin_mask = xtof > x[0]
            xmax_mask = xtof < x[1]
            ymin_mask = ytof > y[0]
            ymax_mask = ytof < y[1]
            pass_mask = pass_en > min_pass
            mask = xmin_mask & xmax_mask & ymin_mask & ymax_mask & pass_mask
            m.append(mask)
        self.mask[mask_name] = m
        self.apply_mask()

    def apply_mask(self):
        tof_c = copy.deepcopy(self.tof_values)
        pass_c = copy.deepcopy(self.pass_energy)
        ele_c = copy.deepcopy(self.elevation)
        for key in self.mask.keys():
            m = self.mask[key]
            for i in range(len(self.retardation)):
                # this does not currently apply multiple masks...
                tof_c[i] = np.asarray(tof_c[i])[m[i]].tolist()
                pass_c[i] = np.asarray(pass_c[i])[m[i]].tolist()
                ele_c[i] = np.asarray(ele_c[i])[m[i]].tolist()
        self.data_masked = self.gen_dataframe(ele_c, pass_c, self.retardation, tof_c)
        self.spec_masked = self.gen_spec(ele_c, pass_c, self.retardation, tof_c)

    def check_rebalance(self):
        hb = self.p_bins == 2
        high_bin = np.count_nonzero(hb)
        mb = self.p_bins == 1
        mid_bin = np.count_nonzero(mb)
        lb = self.p_bins == 0
        low_bin = np.count_nonzero(lb)
        return high_bin, mid_bin, low_bin, high_bin + mid_bin + low_bin

    def rebalance(self, ratio):
        pass_en = []
        for key in self.spec_masked.keys():
            pass_en += self.spec_masked[key][1]
        p = np.log2(pass_en).reshape((-1, 1))
        print(np.max(p), np.min(p))
        est = KBinsDiscretizer(n_bins=3, encode='ordinal', strategy='uniform',
                               subsample=None)
        est.fit(p)
        self.p_bins = est.transform(p)
        hb, mb, lb, total = self.check_rebalance()
        n_lb = (1-ratio) * lb
        hb_mask = self.p_bins == 2
        pass_en_hb = np.log2(np.asarray(pass_en)[hb_mask.flatten()]).reshape((-1, 1))
        #est2 = KBinsDiscretizer(n_bins=10, encode='ordinal', strategy='uniform',
        #                       subsample=None)
        #est2.fit(pass_en_hb)
        #p_bins_hb = est2.transform(pass_en_hb)
        #unique, frequency = np.unique(p_bins_hb, return_counts=True)
        #count = np.asarray(frequency)
        #probs = count / count.sum()
        #full_probs = np.zeros_like(p_bins_hb)

        #for i in range(10):
        #    full_probs = np.where(p_bins_hb==i, probs[i], full_probs)
        # now construct the uniform distribution for training set
        probs = pass_en_hb / pass_en_hb.sum()
        print(probs.flatten(), np.flip(probs.flatten()))
        hb_train = np.random.choice(pass_en_hb.flatten(), size=int(n_lb), replace=False)#, p=np.flip(probs.flatten()))
        fig, ax = plt.subplots(1, 1)
        values, bins, bars = plt.hist(hb_train, edgecolor='white')
        plt.title("histogram of randomly selected values")
        ax.set_xlabel("Discretization")
        ax.set_ylabel("Number of values")
        ax.locator_params(axis='x', integer=True)
        plt.bar_label(bars, fontsize=20, color='navy')
        plt.margins(x=0.01, y=0.1)
        plt.tight_layout()
        plt.show()


