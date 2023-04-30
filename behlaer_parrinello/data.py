import h5py
import pandas as pd
from tqdm import tqdm
import json
import numpy as np



def traverse_datasets(hdf_file):

    def h5py_dataset_iterator(g, prefix=''):
        for key in g.keys():
            item = g[key]
            path = f'{prefix}/{key}'
            if isinstance(item, h5py.Dataset): # test for dataset
                yield (path, item)
            elif isinstance(item, h5py.Group): # test for group (go down)
                yield from h5py_dataset_iterator(item, path)

    for path, _ in h5py_dataset_iterator(hdf_file):
        yield path

dic = {}
cc=0
with h5py.File('fortnet-data22/data/01_Taining_on_Global_Energy_Terms/Training/dft.hdf5', 'r') as f:
    for dset in tqdm(traverse_datasets(f)):
        d = dset.split('/')
        datapoint = d[3]
        col = '_'.join(d[4:])
        dataset = f[dset][:]
        datatype = f[dset].dtype
        if isinstance(dataset, np.ndarray):
            dataset = dataset.tolist()
        if not 'datapoint' in datapoint:
            continue
        if not col in ['geometry_coordinates', 'targets']:
            continue
        if not datapoint in dic:
            dic[datapoint] = {}
        if datapoint in dic:
            dic[datapoint][col] = dataset
        # cc+=1
        # if cc == 100:
        #     break


with open("df.json", "w") as outfile:
    json.dump(dic, outfile)