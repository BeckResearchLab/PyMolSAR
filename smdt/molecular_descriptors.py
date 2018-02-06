from multiprocessing import Process
from smdt.descriptors import topology
from smdt.descriptors import constitution
from smdt.descriptors import bcut
from smdt.descriptors import cats2d
from smdt.descriptors import charge
from smdt.descriptors import connectivity
from smdt.descriptors import estate
from smdt.descriptors import geary
from smdt.descriptors import kappa
from smdt.descriptors import moe
from smdt.descriptors import moran
from smdt.descriptors import moreaubroto
try:
    import urllib.request as urllib2
except ImportError:
    import urllib2
import os.path as op
import inspect
import pandas as pd


def calc_descriptors(df_x):

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

    descriptor_list = [topology_descriptors,constitution_descriptors,burden_descriptors,cats2d_descriptors,
                       charge_descriptors,connectivity_descriptors,estate_descriptors,gearyauto_descriptors,kappa_descriptors,
                       moe_descriptors,moran_descriptors,moreaubroto_descriptors]

    joined_df = topology_descriptors
    for i in descriptor_list[1:]:
        joined_df = joined_df.join(i)

    return joined_df