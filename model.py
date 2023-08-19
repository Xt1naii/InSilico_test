import pandas as pd
import numpy as np
import networkx as nx

import pickle

from sklearn.linear_model import LogisticRegression
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.utils.convert import from_networkx
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.nn import Linear
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
class GCN(torch.nn.Module):
    def __init__(self, pyg_graph):
        super().__init__()
        self.conv1 = GCNConv(pyg_graph.num_features, 64)
        self.conv2 = GCNConv(64, 32)
        self.conv3 = GCNConv(32, 16)
        self.out = Linear(16, 1)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = self.conv3(x, edge_index)
        x = F.relu(x)
        x = self.out(x)
        
        return x
    def get_embeddings(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = self.conv3(x, edge_index)
        return x


def make_graph_dataset():
    string = pd.read_csv('./data/9606.protein.links.v12.0.txt', sep=' ', skiprows=0)
    string["edge_attr"] = pd.to_numeric(string["combined_score"]) / 1000.0
    string.drop(columns=['combined_score'])
    string['protein1'] = string['protein1'].str.replace('9606.', '')
    string['protein2'] = string['protein2'].str.replace('9606.', '')
    string = string.loc[string.protein1 != 'ENSP00000396068']
    string = string.loc[string.protein2 != 'ENSP00000396068']
    string = string.loc[string.protein1 != 'ENSP00000196061']
    string = string.loc[string.protein2 != 'ENSP00000196061']
    G = nx.from_pandas_edgelist(string, 'protein1', 'protein2', ['edge_attr'])

    with open("./artifacts/vector_dict.pkl", "rb") as f:
        vector_dict = pickle.load(f)
    with open('./artifacts/target_dict.pkl', 'rb') as f:
        target = pickle.load(f)
    nx.set_node_attributes(G, vector_dict, 'x')
    nx.set_node_attributes(G, target, 'y')

    pyg_graph = from_networkx(G)

    idx = pyg_graph.y != -1
    idx_labeled = np.where(idx)[0]
    # Node train-test split
    train_idx, test_idx = train_test_split(idx_labeled, random_state=2, train_size=0.8)
    train_mask = np.full(len(idx), False)
    train_mask.put(train_idx, True)
    test_mask = np.full(len(idx), False)
    test_mask.put(test_idx, True)

    pyg_graph.train_mask = train_mask
    pyg_graph.test_mask = test_mask
    return pyg_graph, G, train_idx, test_idx


def train_model(pyg_graph):
    model = GCN(pyg_graph).to(device)
    data = pyg_graph.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=0)
    loss_fn = torch.nn.BCEWithLogitsLoss()
    scheduler = ReduceLROnPlateau(optimizer, 'min', verbose=True)

    EPOCHS = 200

    for epoch in range(EPOCHS):
        model.train()
        optimizer.zero_grad()
        output = model(data)
        loss = loss_fn(output[data.train_mask][:, 0], data.y[data.train_mask].float())
        loss.backward()
        optimizer.step()
        train_loss = loss.item()
        model.eval()
        val_loss = loss_fn(output[data.test_mask][:, 0], data.y[data.test_mask].float()).item()
        scheduler.step(val_loss)
        print(f"Epoch: {epoch} | train loss: {train_loss} | val loss: {val_loss}")
    
    torch.save(model.state_dict(), "./artifacts/model_dict.pt")
    # pred = model(data).detach().numpy()
    # roc_auc_score(data.y[data.test_mask], pred[data.test_mask])
    return model
 
def get_embeddings(G, model, data):
    model = GCN(data)
    model.load_state_dict(torch.load("./artifacts/model_dict.pt"))
    data = data.to(device)
    node_embeddings = model.get_embeddings(data).detach().numpy()
    dict_embeddings = {}
    for idx, node in enumerate(list(G.nodes)):
        if data.y[idx] in [0, 1]:
            dict_embeddings[node] = node_embeddings[idx, :]

    with open("./artifacts/node_embeddings.pkl", "wb") as f:
        pickle.dump(dict_embeddings, f)

def get_preds(X_train, y_train, X_test, y_test, data):
    ensp_drug_counts = data.join(data.ensp.explode().rename('ensp_')).groupby('ensp_', as_index=False) \
        .agg({'name': 'count'}) \
        .query('ensp_ != "-"' ) \
        .rename(columns={"ensp_": "ensp", "name": "num_drugs"}) \
        .set_index("ensp")
    X_train = X_train.join(ensp_drug_counts, how='inner')
    X_test = X_test.join(ensp_drug_counts, how='inner')
    X_train.columns = [str(x) for x in X_train.columns]
    X_test.columns = [str(x) for x in X_test.columns]

    logreg = LogisticRegression()
    logreg.fit(X_train, y_train)

    train_preds = logreg.predict_proba(X_train)[:, 1]
    test_preds = logreg.predict_proba(X_test)[:, 1]
    train_auc = roc_auc_score(y_train, train_preds)
    test_auc = roc_auc_score(y_test, test_preds)
    print(f"Train AUC: {train_auc:.2f}")
    print(f"Test AUC: {test_auc:.2f}")
    return train_preds, test_preds, logreg