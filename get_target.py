import pandas as pd
import numpy as np
import pickle
from functools import partial

from preprocess import remove_words

def process_mechanism(mechanism_list: list[str], mechanism_ensp: dict[str, int]) -> list[str]:
    """
    Return list of ENSP.
    
    Arguments:
    mechanism_list -- list of mechanisms of action
    mechanism_ensp -- dict (dict of mechanism: ENSP)
    """
    processed_list = []
    for item in mechanism_list:
        if item in ['Immunomodulators', 'Immunostimulants']:
            continue
        item = remove_words(item)
        item = item.lower()
        item = item.replace('(', '')
        item = item.replace(')', '')
        item = item.replace('+', '')
        item = mechanism_ensp.get(item)
        if item is None:
            continue
        processed_list.append(item)
    return processed_list

def get_target(data):
    """
    Save dictionary with targets for all proteins from STRING PPI network.
    """
    d = pd.read_parquet('./data/mechanisms_ensg.parquet')
    d[~d.ensp.str.contains('ENSP').fillna(True)]
    mechanism_ensp = d[['mechanism', 'ensp']].set_index('mechanism')['ensp'].to_dict()

    tmp = partial(process_mechanism, mechanism_ensp=mechanism_ensp)
    data['ensp'] = data['mechanism'].apply(tmp)
    data = data.loc[~data['ensp'].str.len().eq(0)]


    ensp_target = data.set_index('passed_to_3')['ensp'].explode() \
        .reset_index() \
        .groupby('ensp') \
        .agg({'passed_to_3': pd.Series.max})['passed_to_3'] \
        # .apply(lambda x: max(x) if type(x) != np.int64 else x)

    ensp_target_dict = ensp_target.to_dict()
    
    all_keggs = pd.read_parquet('./data/all_proteins_kegg.parquet')
    target = dict((ensp,-1) for ensp in all_keggs['ensp'])
    target.update(ensp_target_dict)

    with open('./artifacts/target_dict.pkl', 'wb') as f:
        pickle.dump(target, f)

    return data
