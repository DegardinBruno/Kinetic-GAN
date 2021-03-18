import os,re
import numpy as np
from scipy.io import loadmat, savemat


def humanSort(text):  # Sort function for strings w/ numbers
    convText = lambda seq: int(seq) if seq.isdigit() else seq.lower()
    arrayKey = lambda key: [convText(s) for s in re.split('([0-9]+)', key)]  # Split numbers and chars, base function for sorted
    return sorted(text, key=arrayKey)


def check_runs(method, id=None):
    exps_path = os.path.join('runs', method)
    if not os.path.exists(exps_path): os.makedirs(exps_path)

    exps = [exp for exp in humanSort(os.listdir(exps_path)) if 'exp' in exp]
    out  = os.path.join(exps_path,'exp'+str(len(exps)+1)) if not id else os.path.join(exps_path, exps[id])
    if not id: os.makedirs(os.path.join(exps_path,'exp'+str(len(exps)+1)))
    return out


def save(method, data, name, run_id=-1):
    out = check_runs(method, run_id)
    
    savemat(os.path.join(out,name+'.mat'), data)


def load(method, name, run_id=-1):
    out = check_runs(method, run_id)
    return loadmat(os.path.join(out,name+'.mat'))