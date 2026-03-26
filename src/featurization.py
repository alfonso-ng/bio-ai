import deepchem as dc

def get_featurizer(model):
    if model == "gat":
        return dc.feat.MolGraphConvFeaturizer()
    else:
        return dc.feat.CircularFingerprint(size=2048, radius=2)