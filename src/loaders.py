"""
Created on Oct 16 2023
this code opens .csv files or h5 files that have already been cleaned up

@author: Amanda
"""

import numpy as np
import h5py
import os
import re
import copy


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
        self.mask = {}
        self.tof_values = []
        self.pass_energy = []
        self.initial_ke = []
        self.retardation = []
        self.x_tof = []
        self.y_tof = []

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
                    pass_energy = list(ke - self.retardation[-1] for ke in initial_ke)
                    self.initial_ke.append(initial_ke.tolist())
                    self.pass_energy.append(pass_energy)
                    self.tof_values.append(tof_values.tolist())
                    self.x_tof.append(x_tof.tolist())
                    self.y_tof.append(y_tof.tolist())
            else:
                pass
        self.organize_data()

    def organize_data(self):
        tof_c = copy.deepcopy(self.tof_values)
        pass_c = copy.deepcopy(self.pass_energy)
        r_c = copy.deepcopy(self.retardation)
        ke_c = copy.deepcopy(self.initial_ke)
        x_tof = copy.deepcopy(self.x_tof)
        y_tof = copy.deepcopy(self.y_tof)
        s = sorted(range(len(r_c)),
                   key=r_c.__getitem__)
        tof_c = [tof_c[i] for i in s]
        pass_c = [pass_c[i] for i in s]
        r_c = [r_c[i] for i in s]
        ke_c = [ke_c[i] for i in s]
        x_tof = [x_tof[i] for i in s]
        y_tof = [y_tof[i] for i in s]
        self.dataframe = self.gen_dataframe(pass_c, tof_c, r_c)
        self.initial_ke = ke_c
        self.retardation = r_c
        self.pass_energy = pass_c
        self.tof_values = tof_c
        self.x_tof = x_tof
        self.y_tof = y_tof

    @staticmethod
    def gen_dataframe(pass_energy, tof, retardation):
        pass_flat = []
        tof_flat = []
        r_flat = []
        for i in range(len(retardation)):
            r_flat += len(pass_energy[i]) * [retardation[i]]
            pass_flat += pass_energy[i]
            tof_flat += tof[i]
        data = np.array((np.asarray(r_flat), np.log2(np.asarray(pass_flat)), np.log2(np.asarray(tof_flat))))
        return(data)

    def create_mask(self, x, y, mask_name):
        # used to generate a mask for a part of the data
        # x and y are tuples with the first value being the minimum and second the maximum
        m = []
        for i in range(len(self.retardation)):
            xtof = np.asarray(self.x_tof[i])[:].astype(float)
            ytof = np.abs(np.asarray(self.y_tof[i])[:].astype(float))
            xmin_mask = xtof > x[0]
            xmax_mask = xtof < x[1]
            ymin_mask = ytof > y[0]
            ymax_mask = ytof < y[1]
            mask = xmin_mask & xmax_mask & ymin_mask & ymax_mask
            m.append(mask)
        self.mask[mask_name] = m
        self.apply_mask()

    def apply_mask(self):
        tof_c = copy.deepcopy(self.tof_values)
        pass_c = copy.deepcopy(self.pass_energy)
        for key in self.mask.keys():
            m = self.mask[key]
            for i in range(len(self.retardation)):
                # this does not currently apply multiple masks...
                tof_c[i] = np.asarray(tof_c[i])[m[i]].tolist()
                pass_c[i] = np.asarray(pass_c[i])[m[i]].tolist()
        data = self.gen_dataframe(pass_c, tof_c, self.retardation)
        self.data_masked = data

