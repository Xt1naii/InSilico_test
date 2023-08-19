import os
import gzip
import shutil
import pickle

import requests
import torch
import pandas as pd

from get_adinsight_data import get_adisinsight_data
from preprocess import prepare_dataset
from get_ensg import get_ensg_for_mechanism 
from get_kegg_for_string import get_kegg_for_string
from get_kegg_dict import get_kegg_dict
from get_target import get_target
from model import make_graph_dataset, train_model, get_embeddings, get_preds

URL_STRING = "https://stringdb-downloads.org/download/protein.links.v12.0/9606.protein.links.v12.0.txt.gz"
URL_KEGG = "https://www.gsea-msigdb.org/gsea/msigdb/download_file.jsp?filePath=/msigdb/release/2023.1.Hs/c2.cp.kegg.v2023.1.Hs.entrez.gmt"

def main():
    if not os.path.exists('./data/scrape_result.parquet'):
        print("Downloading drug data from adisinsight..")
        print("This might take a while")
        get_adisinsight_data()
    data_path = "./data/scrape_result.parquet"

    if not os.path.exists('./data/9606.protein.links.v12.0.txt'):
        print("Downloading data from string db")
        filename = URL_STRING.split("/")[-1]
        with open(f"./data/{filename}", "wb") as f:
            r = requests.get(URL_STRING)
            f.write(r.content)
        with gzip.open('./data/9606.protein.links.v12.0.txt.gz', 'rb') as f_in:
            with open('./data/9606.protein.links.v12.0.txt', 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)
        os.remove('./data/9606.protein.links.v12.0.txt.gz')

    if not os.path.exists('./data/c2.cp.kegg.v2023.1.Hs.entrez.gmt.txt'):
        print("Downloading kegg data")
        filename = URL_KEGG.split("/")[-1]
        with open(f"./data/{filename}", "wb") as f:
            r = requests.get(URL_KEGG)
            f.write(r.content)

    data = prepare_dataset(data_path)
    if not os.path.exists('./data/mechanisms_ensg.parquet'):
        get_ensg_for_mechanism(data)

    if not os.path.exists('./data/all_proteins_kegg.parquet'):
        get_kegg_for_string()

    get_kegg_dict()
    data = get_target(data)
    
    pyg_graph, G, train_idx, test_idx = make_graph_dataset()
    if not os.path.exists('./artifacts/model_dict.pt'):
        print("Torch model not found, starting training")
        model = train_model(pyg_graph)
    else:
        model = torch.load('./artifacts/model_dict.pt')  

    get_embeddings(G, model, pyg_graph)
    with open("./artifacts/node_embeddings.pkl", "rb") as f:
        node_embeddings = pickle.load(f)
    node_embeddings = pd.DataFrame(node_embeddings).T 

    train_nodes = [list(G.nodes)[i] for i in train_idx]
    X_train = node_embeddings.loc[train_nodes, :]
    y_train = pyg_graph.y.numpy()[train_idx]

    test_nodes = [list(G.nodes)[i] for i in test_idx]
    X_test = node_embeddings.loc[test_nodes, :]
    y_test = pyg_graph.y.numpy()[test_idx]

    train_preds, test_preds, logreg = get_preds(X_train, y_train, X_test, y_test, data)
    return train_preds, y_train, test_preds, y_test, pyg_graph 

if __name__ == "__main__":
    train_preds, y_train, test_preds, y_test, pyg_graph = main() 
