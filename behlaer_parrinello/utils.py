import math
import numpy as np
import json
import pandas as pd
from tqdm import tqdm
import torch



def magnitude(vector):
    return math.sqrt(sum(pow(element, 2) for element in vector))

def read_data(data_path): return json.load(open(data_path, 'r'))


class SymetryFunctions:

    def __init__(self, dataset, R_c, R_s, eta, la, zeta):
        self.dataset    = dataset
        self.R_c        = R_c
        self.R_s        = R_s
        self.eta        = eta
        self.la         = la
        self.zeta       = zeta


    def cutoff(self, R):
        r = magnitude(R)
        if r <= self.R_c:
            return 0.5 * (math.cos(math.pi*r/self.R_c)+1)
        else:
            return 0
        

    def G1(
        self,
        R_ij    : np.array
        ):
        return math.pow(math.e, -self.eta * math.pow(magnitude(R_ij)-self.R_s, 2)) * self.cutoff(R_ij)

    def G2(
        self,
        R_ij    : np.array,
        R_ik    : np.array,
        R_jk    : np.array
        ):

        theta_ijk = np.dot(R_ij, R_ik)/(magnitude(R_ij) * magnitude(R_ik))

        return math.pow(1 + self.la * math.cos(theta_ijk), self.zeta) * \
            math.pow(math.e, -self.eta * (math.sqrt(magnitude(R_ij))+math.sqrt(magnitude(R_ik))+math.sqrt(magnitude(R_jk)))) * \
            self.cutoff(R_ij) * self.cutoff(R_ik) * self.cutoff(R_jk)


    def G1_sum(self):
        ls = []
        for k,v in tqdm(self.dataset.items()):
            gs = []
            for i in range(len(v['geometry_coordinates'])):
                g = [
                    self.G1(np.subtract(v['geometry_coordinates'][i], v['geometry_coordinates'][j]))
                    for j in range(len(v['geometry_coordinates'])) if j!=i
                    ]
                gs.append(sum(g))
            ls.append(gs)
        return ls

    def G2_sum(self):
        ls = []
        for k,v in tqdm(self.dataset.items()):
            gs = []
            for i in range(len(v['geometry_coordinates'])):
                g = [math.pow(1-self.zeta, 2) * self.G2(
                    np.subtract(v['geometry_coordinates'][i], v['geometry_coordinates'][j]),
                    np.subtract(v['geometry_coordinates'][i], v['geometry_coordinates'][k]),
                    np.subtract(v['geometry_coordinates'][j], v['geometry_coordinates'][k])
                )
                        for j in range(len(v['geometry_coordinates']))
                        for k in range(len(v['geometry_coordinates']))
                        if j!=i and k!=i]
                
                gs.append(sum(g))
            ls.append(gs)
        return ls

dataset = read_data('sample.json')
sf = SymetryFunctions(
    dataset=dataset,
    R_c=6,
    R_s=1,
    eta=1,
    la=1,
    zeta=0.1,
)
print(sf.G2_sum())


exit()
cols = [
    'datapoint',
    'atomicweights',
    'geometry_basis_x',
    'geometry_basis_y',
    'geometry_basis_z',
    'geometry_coordinates',
    'geometry_localattoatnum',
    'geometry_localattoglobalsp',
    'geometry_localattolocalsp',
    'targets'
]
df = pd.DataFrame(columns=cols)

for k,v in tqdm(dataset.items()):
    datapoint = k
    num = len(v['atomicweights'])

    for i in range(num):
        d = {}
        d['datapoint'] = datapoint

        for kk in v.keys():
            if kk == 'geometry_basis':
                gbx = v[kk][0]
                gby = v[kk][1]
                gbz = v[kk][2]
                d['geometry_basis_x'], d['geometry_basis_y'], d['geometry_basis_z'] = gbx, gby, gbz
                continue
            if kk == 'targets':
                e = v[kk][0][0]
                d[kk] = e
                continue
            d[kk] = v[kk][i]
        df = df.append(d, ignore_index=True)

df.to_csv('df.csv')