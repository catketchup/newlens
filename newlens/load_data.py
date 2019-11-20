import numpy as np
import matplotlib.pyplot as plt
from newlens import path
import pandas as pd
import importlib
importlib.reload(path)


class unlensed:
    def __init__(self, *args):
        self.lmin = args[0]
        self.lmax = args[1]
        self.estimator = args[2]
        self.data = np.loadtxt(path.file(1, 'unlensed').file_data())

    def spectra(self):
        ell = np.arange(self.lmin, self.lmax + 1)
        if self.estimator == 'TT':
            return np.concatenate([
                np.zeros(self.lmin),
                self.data[0:(self.lmax - self.lmin + 1), 1] /
                (ell * (ell + 1.)) * 2 * np.pi
            ])

        if self.estimator == 'EE':
            return np.concatenate([
                np.zeros(self.lmin),
                self.data[0:(self.lmax - self.lmin + 1), 2] /
                (ell * (ell + 1.)) * 2 * np.pi
            ])


class lensed:
    def __init__(self, *args):
        self.lmin = args[0]
        self.lmax = args[1]
        self.estimator = args[2]
        self.data = np.loadtxt(path.file(1, 'lensed').file_data())

    def spectra(self):
        ell = np.arange(self.lmin, self.lmax + 1)
        if self.estimator == 'TT':
            return np.concatenate([
                np.zeros(self.lmin),
                self.data[0:(self.lmax - self.lmin + 1), 1] /
                (ell * (ell + 1.)) * 2 * np.pi
            ])

        if self.estimator == 'EE':
            return np.concatenate([
                np.zeros(self.lmin),
                self.data[0:(self.lmax - self.lmin + 1), 2] /
                (ell * (ell + 1.)) * 2 * np.pi
            ])

        if self.estimator == 'BB':
            return np.concatenate([
                np.zeros(self.lmin),
                self.data[0:(self.lmax - self.lmin + 1), 3] /
                (ell * (ell + 1.)) * 2 * np.pi
            ])


class input_cldd:
    def __init__(self, *args):
        self.lmin = args[0]
        self.lmax = args[1]
        self.data = np.loadtxt(path.file(1, 'input_cldd').file_data())

    def spectra(self):
        ell = np.arange(self.lmin, self.lmax + 1)
        # return np.concatenate(
        #     [np.zeros(self.lmin), self.data[0:(self.lmax - self.lmin + 1), 5]])
        return np.concatenate([
            np.zeros(self.lmin), self.data[0:(self.lmax - self.lmin + 1), 5] /
            (ell * (ell + 1.)) * 2 * np.pi
        ])
