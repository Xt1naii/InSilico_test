import pandas as pd
import numpy as np
import pickle
import torch

def inverse_kegg_dict(file) -> dict[any, any]:
    """
    Return two dictionaries with KEGG pathways - default and reversed.
    """
    kegg = {}
    with open(file) as f:
        for line in f:
            pathway = line.split('\t')[0]
            pathway_genes = line.replace('\n', '').split('\t')[2:]
            kegg[pathway] = [int(i) for i in pathway_genes]

    inverse_kegg = {}
    for k,v in kegg.items():
        for x in v:
            inverse_kegg.setdefault(x,[]).append(k)

    return kegg, inverse_kegg


def get_dict(df, dict: dict[str, int]) -> dict:
    """Return dict with pathway vectors.

    Arguments:
    df -- pandas.DataFrame
    dict -- default dict with KEGG pathways (not reversed!)
    """
    list_of_keys = list(dict.keys())
    vector_dict = {}
    for n, j in enumerate(df.pathway):
        zeros = np.zeros((186,), dtype=int)
        if j is None:
            vector_dict[df.ensp[n]] = torch.from_numpy(np.array(zeros)).type(torch.float32)
            continue
        if j is not None:
            for l in j:
                for i, path in enumerate(list_of_keys):
                    if path == l:
                        zeros[i] = 1
            vector_dict[df.ensp[n]] = torch.from_numpy(np.array(zeros)).type(torch.float32)
    
    return vector_dict


def get_kegg_dict():
    """
    Save dict with pathway vectors for all proteins from STRING PPI network.
    """
    all_keggs = pd.read_parquet('./data/all_proteins_kegg.parquet')
    all_keggs['kegg_id'] = all_keggs['kegg_id'].str.replace('hsa:', '')
    m = all_keggs["kegg_id"].str.contains('ENST', na=False)
    all_keggs.loc[m,'kegg_id'] = None
    all_keggs['kegg_id'] = all_keggs['kegg_id'].astype('Int64')

    kegg, inverse_kegg = inverse_kegg_dict('./data/c2.cp.kegg.v2023.1.Hs.entrez.gmt.txt')
    all_keggs['pathway'] = all_keggs.kegg_id.map(inverse_kegg)
    all_keggs = all_keggs.where(pd.notnull(all_keggs), None)

    vector_dict = get_dict(all_keggs, kegg)

    with open('./artifacts/vector_dict.pkl', 'wb') as f:
        pickle.dump(vector_dict, f)
