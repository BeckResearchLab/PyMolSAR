try:
    import urllib.request as urllib2
except ImportError:
    import urllib2
import os.path as op
import inspect
import pandas as pd

df_x = pd.read_csv('SMILES.csv',index_col='Unnamed: 0')

import topology
topology_descriptors = topology.GetTopology(df_x)
topology_descriptors.to_csv('Topology_Descriptors.csv',encoding='utf-8')
topology_descriptors.head()

import constitution
constitution_descriptors = constitution.GetConstitutional(df_x)
constitution_descriptors.to_csv('Constitutional_Descriptors.csv',encoding='utf-8')
constitution_descriptors.head()

import bcut
burden_descriptors = bcut.GetBurden(df_x)
burden_descriptors.to_csv('Burden_Descriptors.csv',encoding='utf-8')
burden_descriptors.head()

