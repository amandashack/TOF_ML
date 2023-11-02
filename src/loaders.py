"""
Created on Oct 16 2023
this code opens .csv files or h5 files that have already been cleaned up

@author: Amanda
"""

import numpy as np
import xarray as xr
import h5py
import matplotlib.pyplot as plt
import os
import re
import arpys
import copy


def interpolator(tofs, pass_ens):
    min_pe = 0
    max_pe = 1200
    x_axis = np.linspace(min_pe, max_pe, num=30000)
    interp_y = []
    for i in range(len(tofs)):
        interp_y.append(np.interp(x_axis, pass_ens[i], tofs[i]))
    return(np.array([y for y in interp_y]), x_axis)


class MRCOLoader(object):
    """
    Implements data loading and storage for use plotting
    """

    _TOLERATED_EXTENSIONS = {
        ".h5"
    }

    def __init__(self, filepath: str):
        self.filepath = filepath
        self.spec_dict = {}
        self.tof_values = []
        self.pass_energy = []
        self.initial_ke = []
        self.retardation = []

    def load(self):
        fp = os.path.abspath(self.filepath)
        for file in os.listdir(fp):
            if file.endswith("h5"):
                path1 = os.path.join(fp, file)
                self.retardation.append(int(re.findall(r'_R(\d+)_', file)[0]))
                with h5py.File(path1, 'r') as f:
                    tof_values = f['data1']['tof'][:]
                    initial_ke = f['data1']['initial_ke'][:]
                    pass_energy = list(ke - self.retardation[-1] for ke in initial_ke)
                    self.spec_dict[self.retardation[-1]] = [tof_values[-1].tolist(), pass_energy[-1]]
                    self.initial_ke.append(initial_ke.tolist())
                    self.pass_energy.append(pass_energy)
                    self.tof_values.append(tof_values.tolist())
            else:
                print("your file is not an h5 file. Please input a different filepath")

    def organize_data(self):
        tof_c = copy.deepcopy(self.tof_values)
        pass_c = copy.deepcopy(self.pass_energy)
        r_c = copy.deepcopy(self.retardation)
        ke_c = copy.deepcopy(self.initial_ke)
        s = sorted(range(len(r_c)),
                   key=r_c.__getitem__)
        tofs = [tof_c[i] for i in s]
        pass_ens = [pass_c[i] for i in s]
        r = [r_c[i] for i in s]
        ke = [ke_c[i] for i in s]
        spec = {r[i]: [pass_ens[i], tofs[i]] for i in range(len(r_c))}
        self.spec_dict = spec
        self.initial_ke = ke
        self.retardation = r
        self.pass_energy = pass_ens
        self.tof_values = tofs
