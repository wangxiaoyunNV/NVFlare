# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os.path
import time

import torch
import torch.nn.functional as F
import tqdm
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import f1_score
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from torch_geometric.data import Batch
from torch_geometric.datasets import PPI
from torch_geometric.loader import DataLoader, LinkNeighborLoader
from torch_geometric.nn import GraphSAGE
from read_data import process_ellipitc


# (0) import nvflare client API
import nvflare.client as flare

# Create elliptic dataset for training.

df_features = pd.read_csv('../data/elliptic_bitcoin_dataset/elliptic_txs_features.csv', header=None)
df_edges = pd.read_csv("../data/elliptic_bitcoin_dataset/elliptic_txs_edgelist.csv")
df_classes =  pd.read_csv("../data/elliptic_bitcoin_dataset/elliptic_txs_classes.csv")
node_features, classified_idx, edge_index, weights, labels, y_train = process_ellipitc(df_features, df_edges, df_classes)

# converting data to PyGeometric graph data format
data_train = Data(x=node_features, edge_index=edge_index, edge_attr=weights,
                               y=torch.tensor(labels, dtype=torch.double)) #, adj= torch.from_numpy(np.array(adj))
X_train, X_valid, y_train, y_valid, train_idx, valid_idx = train_test_split(node_features[classified_idx], y_train, classified_idx, test_size=0.15, random_state=42, stratify=y_train)

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

model = GraphSAGE(in_channels = data_train.num_node_features, 
                  hidden_channels=64, num_layers=3,
                  out_channels = 2)
model.double()
data_train = data_train.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-2, weight_decay=5e-4)


# (1) initializes NVFlare client API
flare.init()
print("finish init")

# (2) gets FLModel from NVFlare
input_model = flare.receive()
print("finish receive")

# (3) loads model from NVFlare
model.load_state_dict(input_model.params)
model.to(device)
print("finish load_state_dict")


def train():
    model.train()
    optimizer.zero_grad()
    out = model(data_train)
    loss = F.nll_loss(out[train_idx], data_train.y[train_idx].T.to(torch.long))
    loss.backward()
    optimizer.step()

    return loss.item()




@torch.no_grad()
def test():
    model.eval()
    out = model(data_train)
    y_pred = torch.argmax(out,dim=1).detach().cpu().numpy()
    y_true = data_train.y.detach().cpu().numpy() 
    train_auc = roc_auc_score(y_true[train_idx], y_pred[train_idx])
    valid_auc = roc_auc_score(y_true[valid_idx], y_pred[valid_idx]) 

    return train_auc, valid_auc
  

times = []
_, _, global_test_auc = test()
print(f"Global Test AUC: {global_test_auc:.4f}")


# (optional) calculate total steps
steps = number_epochs = 100

for epoch in range(1, number_epochs):
    start = time.time()
    loss = train()
    print(f"Epoch: {epoch:02d}, Loss: {loss:.4f}")
    train_auc, val_auc= test()
    print(f"Train AUC: {train_auc:.4f}, Val AUC: {val_auc:.4f} ")
    times.append(time.time() - start)


print(f"Median time per epoch: {torch.tensor(times).median():.4f}s")


# (5) construct the FLModel to returned back
output_model = flare.FLModel(
    params=model.cpu().state_dict(),
    params_type="FULL",
    metrics={"test_auc": global_test_auc},
    meta={"NUM_STEPS_CURRENT_ROUND": steps},
)

# (6) send back model
flare.send(output_model)
