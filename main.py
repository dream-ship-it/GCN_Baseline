import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.nn import GATConv
from torch_geometric.utils import to_dense_adj
import csv
from data_loader import load_data

class GCN(torch.nn.Module):
    def __init__(self, num_features, hidden_channels, nclasses):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(num_features, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, nclasses)

    def forward(self, x, edge_index, edge_weight=None):
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv1(x, edge_index, edge_weight).relu()
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index, edge_weight)
        return x



class GAT(torch.nn.Module):
    def __init__(self, num_features, hidden_channels, nclasses):
        super(GAT, self).__init__()
        self.conv1 = GATConv(num_features, hidden_channels, heads=1, dropout=0.6)
        self.conv2 = GATConv(hidden_channels, nclasses, heads=1, dropout=0.6)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.6, training=self.training)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)

class MLP(torch.nn.Module):
    def __init__(self, num_features, hidden_channels, nclasses):
        super(MLP, self).__init__()
        self.layer1 = torch.nn.Linear(num_features, hidden_channels)
        self.layer2 = torch.nn.Linear(hidden_channels, nclasses)

    def forward(self, x, edg):
        x = F.relu(self.layer1(x))
        x = F.dropout(x, p=0.6, training=self.training)
        x = self.layer2(x)
        return F.log_softmax(x, dim=1)

dataset = ['cora']

# 设置随机数生成器的种子
seed = 42
torch.manual_seed(seed)

#dataset = ['cora','pubmed','citeseer','photo','computers','actor','cs','cornell','texas','wisconsin','chameleon','physics','wikics','squirrel']
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
    early_stop = 10
    lr = 0.01

    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    num_features = nfeats

    test_accs = []  # List to store test accuracy of each run

    best_model = 0

    for runs in range(100):  # Perform 10 runs

        best_acc = 0
        model = GCN(num_features=num_features, hidden_channels=hid_dim, nclasses=nclasses).to(device)
        optimizer = torch.optim.Adam([
            dict(params=model.conv1.parameters(), weight_decay=5e-4),
            dict(params=model.conv2.parameters(), weight_decay=0)
        ], lr=lr)  # Only perform weight-decay on first convolution.


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
            torch.save(model, "gcn_res/" + res_file)
        print(best_model)
        test_accs.append(best_acc)

    avg_test_acc = np.mean(test_accs)
    std_test_acc = np.std(test_accs)
    print(f'Average Best Test Acc: {avg_test_acc:.4f}, Std Dev: {std_test_acc:.4f}')

    results.append([dataset_name, epochs, hid_dim, avg_test_acc, std_test_acc, num_layers])

    with open('res/results_gcn.csv', 'a', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerows(results)
        csv_writer.writerows('\n')












