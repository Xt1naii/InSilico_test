from concurrent.futures import ThreadPoolExecutor
from functools import partial

import requests
import pandas as pd

def get_kegg(ensp: str) -> str:
    """
    Return the result of the request.
    Return the kegg_id value. 
    """
    response = requests.get(f"https://rest.uniprot.org/uniprotkb/search?query={ensp} AND (taxonomy_id:9606)&fields=xref_ensembl,protein_name,xref_kegg")
    response = response.json()
    try:
        ensg = response["results"][0]['uniProtKBCrossReferences'][0]['properties'][1]['value'].split('.')[0]
    except:
        ensg = None
    try:
        kegg_id = response['results'][0]['uniProtKBCrossReferences'][-1]['id']
    except:
        kegg_id = None
    return {
        'ensg': ensg,
        'ensp':  ensp,
        'kegg_id': kegg_id
    }
 

def get_kegg_for_string():
    """
    Save kegg_id for all proteins from STRING PPI network. 
    """
    string = pd.read_csv('./data/9606.protein.links.v12.0.txt', sep=' ', skiprows=0)
    string['protein1'] = string['protein1'].str.replace('9606.', '')
    string['protein2'] = string['protein2'].str.replace('9606.', '')
    all_proteins = pd.concat([string.protein1, string.protein2]).dropna().reset_index(drop=True).to_list()
    all_proteins = list(set(all_proteins))

    with ThreadPoolExecutor() as executor:
        result = list(executor.map(get_kegg, all_proteins))
    result = [x for x in result if x]
    result = pd.DataFrame(result)
    result.to_parquet('./data/all_proteins_kegg.parquet')
