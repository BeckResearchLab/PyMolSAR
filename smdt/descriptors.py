try:
    import urllib.request as urllib2
except ImportError:
    import urllib2
import os.path as op
import inspect
import pandas as pd
from smdt import topology
from smdt import constitution
from smdt import bcut
from smdt import cats2d
from smdt import charge
from smdt import connectivity
from smdt import estate
from smdt import geary
from smdt import kappa
from smdt import moe
from smdt import moran
from smdt import moreaubroto

def Calculate_AllDescriptors(df_x):

    topology_descriptors = topology.GetTopology(df_x)
    constitution_descriptors = constitution.GetConstitutional(df_x)
    burden_descriptors = bcut.GetBurden(df_x)
    cats2d_descriptors = cats2d.CATS2D(df_x)
    charge_descriptors = charge.GetCharge(df_x)
    connectivity_descriptors = connectivity.GetConnectivity(df_x)
    estate_descriptors = estate.GetEstate(df_x)
    gearyauto_descriptors = geary.GetGearyAuto(df_x)
    kappa_descriptors = kappa.GetKappa(df_x)
    moe_descriptors = moe.GetMOE(df_x)
    moran_descriptors = moran.GetMoranAuto(df_x)
    moreaubroto_descriptors = moreaubroto.GetMoreauBrotoAuto(df_x)

    names = [topology_descriptors,constitution_descriptors,burden_descriptors,cats2d_descriptors,charge_descriptors,
             connectivity_descriptors,estate_descriptors,gearyauto_descriptors,kappa_descriptors,moe_descriptors,
             moran_descriptors,moreaubroto_descriptors]

    for i in names:
        df_x = df_x.join(i)

    return df_x
