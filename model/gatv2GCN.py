import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv, global_mean_pool as gap, global_max_pool as gmp, GCNConv

class GraphConvolutionalBlock(nn.Module):
    def __init__(self, input_dim, output_dim, dropout_rate):
        super(GraphConvolutionalBlock, self).__init__()
        self.gatv2_layer1 = GATv2Conv(input_dim, output_dim, heads=10, dropout=dropout_rate)
        self.gatv2_layer2 = GCNConv(output_dim * 10, output_dim * 10, dropout=dropout_rate)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x, edge_index, batch):
        x = F.dropout(x, p=0.2, training=self.training)
        x = F.elu(self.gatv2_layer1(x, edge_index))
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.gatv2_layer2(x, edge_index)
        x = self.activation(x)
        x = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)
        return x

class ProteinSequenceBlock(nn.Module):
    def __init__(self, xt_features, embed_dim, n_filters, output_dim):
        super(ProteinSequenceBlock, self).__init__()
        self.embedding = nn.Embedding(xt_features + 1, embed_dim)
        self.conv_protein = nn.Conv1d(in_channels=1000, out_channels=n_filters, kernel_size=8)
        self.bilstm = nn.LSTM(embed_dim, 64, 1, dropout=0.2, bidirectional=True)
        self.fc_protein = nn.Linear(32 * 121, output_dim)

    def forward(self, target):
        embedded_xt = self.embedding(target)
        embedded_xt, _ = self.bilstm(embedded_xt)
        conv_xt = self.conv_protein(embedded_xt)
        xt = conv_xt.view(-1, 32 * 121)
        return self.fc_protein(xt)

class CombinedDenseBlock(nn.Module):
    def __init__(self, input_dim, output_size, dropout_rate):
        super(CombinedDenseBlock, self).__init__()
        self.fc_combined1 = nn.Linear(input_dim, 1024)
        self.fc_combined2 = nn.Linear(1024, 512)
        self.output_layer = nn.Linear(512, output_size)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        x = self.fc_combined1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.fc_combined2(x)
        x = self.activation(x)
        x = self.dropout(x)
        return self.output_layer(x)

class GATv2GCNModel(nn.Module):
    def __init__(self, output_size=1, xd_features=78, xt_features=25,
                 filter_count=32, embedding_dim=128, dense_output_dim=128, dropout_rate=0.2):
        super(GATv2GCNModel, self).__init__()
        self.graph_block = GraphConvolutionalBlock(xd_features, xd_features, dropout_rate)
        self.protein_block = ProteinSequenceBlock(xt_features, embedding_dim, filter_count, dense_output_dim)
        self.combined_block = CombinedDenseBlock(256, output_size, dropout_rate)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        target = data.target

        graph_features = self.graph_block(x, edge_index, batch)
        protein_features = self.protein_block(target)

        combined_features = torch.cat((graph_features, protein_features), 1)
        return self.combined_block(combined_features)