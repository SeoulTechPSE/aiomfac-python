''' this script includes code snippets for loading pickle files and returning Tg prediction
authored by Gianluca Armeli'''

# available model_name = ['fg_cho','fg_nhal','fg_cho_no_tm','fg_nhal_no_tm','sm','sm_no_tm']

''' input formats for feat '''
# feat needs to be list of list, i.e. [[...],[...]] . Keep in mind if you only have one sample.
# fg_cho: feat = [CH3, CH2, CH, C, OH, COC, =O, DBE, O:C, M, Tm]
# fg_cho_no_tm: X_pred = [CH3, CH2, CH, C, OH, COC, =O, DBE, O:C, M]

# fg_nhal: feat = [CH3, CH2, CH, C, OH, COC, =O, DBE, N, Hal, O:C, M, Tm]
# fg_nhal_no_tm: feat = [CH3, CH2, CH, C, OH, COC, =O, DBE, N, Hal, O:C, M]

# sm: feat = [smiles1, smiles2, ...], Tm = [[Tm1], [Tm2], ...]
# sm_no_tm: feat = [smiles1, smiles2, ...]

# for the fg modes the Tm needs to be included in the feat list
# for the sm modes the Tm needs to be delivered separately as described above 

import pickle
import numpy as np
from deepchem.feat.base_classes import MolecularFeaturizer
from rdkit.Chem import Descriptors, AllChem, MolFromSmiles
from rdkit import Chem
from rdkit import DataStructs
from deepchem.utils.typing import RDKitMol
import logging

class RDKitDescriptors(MolecularFeaturizer):
    def __init__(self, use_fragment=True, ipc_avg=True):
	    self.use_fragment = use_fragment
	    self.ipc_avg = ipc_avg
	    self.descriptors = []
	    self.descList = []
        
    def _featurize(self, mol: RDKitMol) -> np.ndarray:
	    # initialize
	    if len(self.descList) == 0:
	        try:
	            for descriptor, function in Descriptors.descList:
		            if self.use_fragment is False and descriptor.startswith('fr_'):
		                continue
		            self.descriptors.append(descriptor)
		            self.descList.append((descriptor, function))
	        except ModuleNotFoundError:
	            raise ImportError("This class requires RDKit to be installed.")

	    # check initialization
	    assert len(self.descriptors) == len(self.descList)
	    features = []
	    for desc_name, function in self.descList:
	        if desc_name == 'Ipc' and self.ipc_avg:
	            feature = function(mol, avg=True)
	        else:
	            feature = function(mol)
	        features.append(feature)
	    return np.asarray(features)

def rd_descriptor(smiles):
    mol = Chem.MolFromSmiles(smiles)
    featurizer = RDKitDescriptors()
    fp = featurizer.featurize(mol)
    fp = fp.reshape((208,))
    return fp

def rd_descriptor_list(list_of_smiles):
    fingerprints = []
    featurizer = RDKitDescriptors()
    for smiles in list_of_smiles:
        mol = Chem.MolFromSmiles(smiles)
        fp = featurizer.featurize(mol)
        fp = fp.reshape((208,))
        fingerprints.append(fp)
    return fingerprints 

def load(model_name):
    pickle_in = open('pickle/{}'.format(model_name),'rb')
    model = pickle.load(pickle_in)
    return model

def predict(model_name, X_pred):
    model = load(model_name)
    y_pred = model.predict(X_pred)
    return y_pred

def make_X_pred_sm(Tm, smiles):
    fp = rd_descriptor_list(smiles)
    if Tm == None:
        X_pred = fp
    else:
        X_pred = np.concatenate((Tm, fp), axis=1)
    return X_pred

def main(model_name, feat, Tm=None):
    if model_name in ['sm','sm_no_tm']:
        X_pred = make_X_pred_sm(Tm, feat)
    else:
        X_pred = feat
    return predict(model_name, X_pred)


# examples
print('Tg =', main('fg_nhal', feat=[[0, 0, 5, 1, 0, 0, 0, 4, 1, 0, 0.0, 93, 200],
                                       [0, 0, 5, 1, 0, 0, 0, 4, 1, 0, 0.0, 93, 200]]))

print('Tg =', main('sm', feat=['C1=CC=CC=C1','C1=CC=CC=C1'], Tm=[[279],[279]]))

print('Tg =', main('sm_no_tm', feat=['C1=CC=CC=C1','C1=CC=CC=C1'], Tm=None))
