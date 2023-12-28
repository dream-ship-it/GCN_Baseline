import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.utils import to_dense_adj
import csv
from data_loader import load_data

# class GCN(torch.nn.Module):
#     def __init__(self, num_features, hidden_channels, nclasses):
#         super(GCN, self).__init__()
#         self.conv1 = GCNConv(num_features, hidden_channels)
#         self.conv2 = GCNConv(hidden_channels, nclasses)
#
#     def forward(self, x, edge_index, edge_weight=None):
#         x = F.dropout(x, p=0.5, training=self.training)
#         x = self.conv1(x, edge_index, edge_weight).relu()
#         x = F.dropout(x, p=0.5, training=self.training)
#         x = self.conv2(x, edge_index, edge_weight)
#         return x

class GCN(torch.nn.Module):
    # hidden_layers： GNN的层数，GRU层数 = GNN + 1
    def __init__(self, num_features, hidden_channels, hidden_layers, nclasses):
        super(GCN, self).__init__()
        self.conv = nn.Sequential()
        self.gru = nn.Sequential()
        self.hidden_channels = hidden_channels
        for i in range(hidden_layers):
            if i == 0:
                self.conv.append(GCNConv(num_features, hidden_channels))
                self.gru.append(nn.GRUCell(num_features, hidden_channels))
            self.conv.append(GCNConv(hidden_channels, hidden_channels))
            self.gru.append(nn.GRUCell(hidden_channels, hidden_channels))
        self.gru.append(nn.GRUCell(hidden_channels, hidden_channels))
        self.fc = nn.Linear(hidden_channels, nclasses)

    def forward(self, x, edge_index, edge_weight=None):
        h = torch.zeros(x.size(0), self.hidden_channels, device=x.device)
        x = F.dropout(x, p=0.5, training=self.training)
        h = self.gru[0](x, h)
        for i in range(len(self.conv)):
            x = self.conv[i](x, edge_index, edge_weight).relu()
            h = self.gru[i+1](x, h)
            x = F.dropout(x, p=0.5, training=self.training)
        out = self.fc(h)
        return out

# 设置随机数生成器的种子
seed = 42
torch.manual_seed(seed)
dataset = ['cora', 'citeseer', 'pubmed']
# dataset = ['cora','pubmed','citeseer','photo','computers','actor','cs','cornell','texas','wisconsin','chameleon','physics','wikics','squirrel']
for dataset_name in dataset:

    features, edges, train_mask, val_mask, test_mask, labels, nnodes, nfeats, nclasses = load_data(dataset_name)
    train_mask = train_mask[:, 0]
    val_mask = val_mask[:, 0]
    test_mask = test_mask[:, 0]
    adj = to_dense_adj(edges).squeeze(0)
    results = []
    num_layers = 2
    best_acc = 0
    epochs = 200
    hid_dim = 16
    hidden_layers = 2

    early_stop = 10
    lr = 0.001

    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    num_features = nfeats

    test_accs = []  # List to store test accuracy of each run

    best_model = 0

    for runs in range(100):  # Perform 10 runs

        best_acc = 0
        model = GCN(num_features=num_features, hidden_channels=hid_dim, hidden_layers=hidden_layers, nclasses=nclasses).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)  # Only perform weight-decay on first convolution.


        def train():
            model.train()
            optimizer.zero_grad()
            out = model(features, edges)
            loss = F.cross_entropy(out[train_mask], labels[train_mask])
            loss.backward()
            optimizer.step()


        @torch.no_grad()
        def run():
            model.eval()
            out = model(features, edges)
            pred = out.argmax(dim=1)
            accs = []
            losses = []
            for mask in (train_mask, val_mask, test_mask):
                acc = pred[mask].eq(labels[mask]).sum().item() / mask.sum().item()
                loss = F.cross_entropy(out[mask], labels[mask])
                accs.append(acc)
                losses.append(loss.item())
            return accs, losses

        # res_file = "gcntest_" + dataset_name
        # for epoch in range(1, epochs):
        #     train()
        #     train_acc, val_acc, test_acc = run()
        #     if best_acc < test_acc:
        #         best_acc = test_acc
        #         if best_acc > best_model:
        #             best_model = best_acc
        #             torch.save(model, "gcn_res/" + res_file)
        #     print(
        #         f'Run: {str(runs + 1)}, Epoch: {epoch:03d}, Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}, Test Acc: {test_acc:.4f}')
        # print(best_model)
        # test_accs.append(best_acc)

        res_file = "gcntest_" + dataset_name
        cost_val = []
        for epoch in range(1, epochs):
            train()
            accs, losses = run()

            train_acc, val_acc, test_acc = accs
            train_loss, val_loss, test_loss = losses

            cost_val.append(val_loss)
            if best_acc < test_acc:
                best_acc = test_acc

            print(
                f'Run: {str(runs + 1)}, Epoch: {epoch:03d}, Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}, Test Acc: {test_acc:.4f}')

            if epoch > early_stop and cost_val[-1] > np.mean(cost_val[-(early_stop + 1): -1]):
                print(f"Epoch {epoch} Early stopping...")
                break

        if best_acc > best_model:
            best_model = best_acc
            torch.save(model, "gated_res/" + res_file)
        print(best_model)
        test_accs.append(best_acc)

    avg_test_acc = np.mean(test_accs)
    std_test_acc = np.std(test_accs)
    print(f'Average Best Test Acc: {avg_test_acc:.4f}, Std Dev: {std_test_acc:.4f}')

    results.append([dataset_name, epochs, hid_dim, avg_test_acc, std_test_acc, num_layers])

    with open('res/results_gated.csv', 'a', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerows(results)
        csv_writer.writerows('\n')
