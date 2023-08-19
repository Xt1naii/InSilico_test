from concurrent.futures import ThreadPoolExecutor
from functools import partial

import requests
import pandas as pd

from preprocess import remove_words

def get_ensg(mechanism: str) -> str:
    """
    Return ENSG, ENSP and kegg_id from UNIPROT db.
    """
    response = requests.get(f"https://rest.uniprot.org/uniprotkb/search?query={mechanism} AND (taxonomy_id:9606)&fields=xref_ensembl,protein_name,xref_kegg")
    response = response.json()
    try:
        ensg = response["results"][0]['uniProtKBCrossReferences'][0]['properties'][1]['value'].split('.')[0]
    except:
        ensg = None
    try:
        ensp = response["results"][0]['uniProtKBCrossReferences'][0]['properties'][0]['value'].split('.')[0]
    except:
        ensp = None
    try:
        kegg_id = response['results'][0]['uniProtKBCrossReferences'][-1]['id']
    except:
        kegg_id = None
    return {
        'mechanism': mechanism,
        'ensg': ensg,
        'ensp':  ensp,
        'kegg_id': kegg_id
    }
 
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

def get_ensg_for_mechanism(data):
    mechanisms = data.mechanism.explode()
    mechanisms = mechanisms[~mechanisms.isin(['Immunomodulators', 'Immunostimulants'])]
    mechanisms = mechanisms.apply(remove_words)
    mechanisms = mechanisms.str.lower()
    mechanisms = mechanisms.str.replace('(', '')
    mechanisms = mechanisms.str.replace(')', '')
    mechanisms = mechanisms.str.replace('+', '')
    mechanisms = mechanisms.unique() 

    with ThreadPoolExecutor() as executor:
        result = list(executor.map(get_ensg, mechanisms))
    result = [x for x in result if x]
    result = pd.DataFrame(result)
    result.to_parquet('./data/mechanisms_ensg.parquet') 

 