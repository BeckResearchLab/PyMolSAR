"""
This module is to calculate the ghosecrippen descriptor
"""
import string
import os
from rdkit import Chem
import os.path as op
import inspect

def _ReadPatts(fileName):
    """
    *Internal Use Only*

    parses the pattern list from the data file
    """
    patts = {}
    order = []
    with open(fileName, 'r') as f:
        lines = f.readlines()
    for line in lines:
        if line[0] != '#':
            splitLine = line.split('\t')
            if len(splitLine) >= 4 and splitLine[0] != '':
                sma = splitLine[1]
                if sma != 'SMARTS':
                    sma.replace('"', '')
                    p = Chem.MolFromSmarts(sma)
                    if p:
                        cha = string.strip(splitLine[0])
                        if cha not in order:
                            order.append(cha)
                        l = patts.get(cha, [])
                        l.append((sma, p))
                        patts[cha] = l
                else:
                    print('Problems parsing smarts: %s' % (sma))
    return order, patts


def GhoseCrippenFingerprint(mol, count=False):
    """
    #################################################################
    Ghose-Crippen substructures or counts based on the definitions of

    SMARTS from Ghose-Crippen's paper. (110 dimension)

    The result is a dict format.
    #################################################################
    """
    path = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
    path = op.join(path, 'crippen.txt')
    order, patts = _ReadPatts(os.path.dirname(path))
    GCres = dict()
    for sma in patts:
        match = mol.GetSubstructMatches(patts[sma][0][1], False, False)
        temp = len([i[0] for i in match])
        GCres.update({sma: temp})

    res = {}
    if count == False:
        for i in GCres:
            if GCres[i] > 0:
                res.update({i: 1})
            else:
                res.update({i: 0})
    else:
        res = GCres

    return res