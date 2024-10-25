import torch
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree, softmax, to_dense_batch
from torch_geometric.nn import global_add_pool, global_mean_pool, global_max_pool, GlobalAttention, Set2Set
import torch.nn.functional as F
# from torch_scatter import scatter_add
from torch_geometric.nn.inits import glorot, zeros
from torch_geometric.nn import HGTConv, SAGEConv, GINConv,GraphConv,GATv2Conv,GATConv, GCNConv, HeteroConv
from torch.nn import Linear, Sequential, ReLU
from torch_geometric.nn.aggr import MultiAggregation

input_dim = 14 # totally 17 types of nodes
"""
Homogeneous GNNs
"""
class GNN(torch.nn.Module):
    """
    Args:
        num_layer (int): the number of GNN layers
        emb_dim (int): dimensionality of embeddings
        JK (str): last, concat, max or sum.
        max_pool_layer (int): the layer from which we use max pool rather than add pool for neighbor aggregation
        drop_ratio (float): dropout rate
        gnn_type: gin, gcn, graphsage, gat

    Output:
        node representations

    """
    def __init__(self, num_layer, input_dim, emb_dim, JK = "last", drop_ratio = 0, gnn_type = "gin"):
        super(GNN, self).__init__()
        self.num_layer = num_layer
        self.drop_ratio = drop_ratio
        self.JK = JK
        self.input_dim = input_dim

        if self.num_layer < 2:
            raise ValueError("Number of GNN layers must be greater than 1.")

        # nn.embedding for node type, we do not need one-hot anymore
        self.type_embedding = torch.nn.Embedding(input_dim, emb_dim) # type embedding
        self.output_embedding = torch.nn.Embedding(2, emb_dim) # output embedding (whether the gate is a output)


        torch.nn.init.xavier_uniform_(self.type_embedding.weight.data)
        torch.nn.init.xavier_uniform_(self.output_embedding.weight.data)

        ###List of MLPs
        self.gnns = torch.nn.ModuleList()
        for layer in range(num_layer):
            if gnn_type == "gin":
                self.gnns.append(GINConv(Sequential(Linear(emb_dim, emb_dim),ReLU(),Linear(emb_dim, emb_dim))))
            elif gnn_type == "gcn":
                self.gnns.append(GCNConv(emb_dim,emb_dim))
            elif gnn_type == "gat":
                self.gnns.append(GATv2Conv(emb_dim, emb_dim))
            elif gnn_type == "graphsage":
                self.gnns.append(SAGEConv((-1,-1), emb_dim))
        
        self.pool = global_mean_pool

        ###List of batchnorms
        self.batch_norms = torch.nn.ModuleList()
        for layer in range(num_layer):
            self.batch_norms.append(torch.nn.BatchNorm1d(emb_dim, dtype = torch.float16))
        self.num_features = emb_dim
        self.cat_grep = True

    #def forward(self, x, edge_index, edge_attr):
    def forward(self, *argv):
        if len(argv) == 2:
            x, edge_index = argv[0], argv[1]
        elif len(argv) == 1:
            data = argv[0]
            x, edge_index, batch = data.x, data.edge_index, data.batch
        else:
            raise ValueError("unmatched number of arguments.")

        type_emeb = self.type_embedding(x[...,0])
        output_emeb = self.output_embedding(x[...,1])
        x = type_emeb + output_emeb

        h_list = [x]
        for layer in range(self.num_layer):
            h = self.gnns[layer](h_list[layer], edge_index)
            h = self.batch_norms[layer](h)
            #h = F.dropout(F.relu(h), self.drop_ratio, training = self.training)
            if layer == self.num_layer - 1:
                #remove relu for the last layer
                h = F.dropout(h, self.drop_ratio, training = self.training)
            else:
                h = F.dropout(F.relu(h), self.drop_ratio, training = self.training)
            h_list.append(h)

        ### Different implementations of Jk-concat
        if self.JK == "concat":
            node_representation = torch.cat(h_list, dim = 1)
        elif self.JK == "last":
            node_representation = h_list[-1]
        elif self.JK == "max":
            h_list = [h.unsqueeze_(0) for h in h_list]
            node_representation = torch.max(torch.cat(h_list, dim = 0), dim = 0)[0]
        elif self.JK == "sum":
            h_list = [h.unsqueeze_(0) for h in h_list]
            node_representation = torch.sum(torch.cat(h_list, dim=0), dim=0)[0]
        

        h_graph = self.pool(node_representation, batch) # shape = [B, D]
        batch_node, batch_mask = to_dense_batch(node_representation, batch) # shape = [B, n_max, D], 
        batch_mask = batch_mask.bool()

        if self.cat_grep:
            batch_node = torch.cat((h_graph.unsqueeze(1), batch_node), dim=1) # shape = [B, n_max+1, D]
            batch_mask = torch.cat([torch.ones((batch_mask.shape[0], 1), dtype=torch.bool, device=batch.device), batch_mask], dim=1)
            return batch_node, batch_mask
        else:
            return batch_node, batch_mask, h_graph


class HeteGNN(torch.nn.Module):
    """
    Args:
        num_layer (int): the number of GNN layers
        emb_dim (int): dimensionality of embeddings
        JK (str): last, concat, max or sum.
        max_pool_layer (int): the layer from which we use max pool rather than add pool for neighbor aggregation
        drop_ratio (float): dropout rate
        gnn_type: gin, gcn, graphsage, gat

    Output:
        node representations

    """
    def __init__(self, num_layer, emb_dim, JK = "last", drop_ratio = 0, gnn_type = "gin",final_aggr = 'multi-aggregate'):
        super(HeteGNN, self).__init__()
        self.num_layer = num_layer
        self.drop_ratio = drop_ratio
        self.JK = JK

        if self.num_layer < 2:
            raise ValueError("Number of GNN layers must be greater than 1.")

        # nn.embedding for node type, we do not need one-hot anymore
        self.type_embedding = torch.nn.Embedding(input_dim, emb_dim) # type embedding
        self.output_embedding = torch.nn.Embedding(2, emb_dim) # output embedding (whether the gate is a output)


        torch.nn.init.xavier_uniform_(self.type_embedding.weight.data)
        torch.nn.init.xavier_uniform_(self.output_embedding.weight.data)

        ###List of MLPs
        self.gnns = torch.nn.ModuleList()
        for layer in range(num_layer):
            if gnn_type == "gin":
                self.gnns.append(HeteroConv({
                    ('node', 'pos', 'node'): GINConv(Sequential(Linear(emb_dim, emb_dim),ReLU(),Linear(emb_dim, emb_dim))),
                    ('node', 'inv', 'node'): GINConv(Sequential(Linear(emb_dim, emb_dim),ReLU(),Linear(emb_dim, emb_dim))),
                }, aggr='sum'))
            elif gnn_type == "gcn":
                self.gnns.append(HeteroConv({
                    ('node', 'pos', 'node'): GCNConv(emb_dim,emb_dim),
                    ('node', 'inv', 'node'): GCNConv(emb_dim,emb_dim),
                }, aggr='sum'))
            elif gnn_type == "gat":
                self.gnns.append(HeteroConv({
                    ('node', 'pos', 'node'): GATv2Conv(emb_dim, emb_dim),
                    ('node', 'inv', 'node'): GATv2Conv(emb_dim, emb_dim),
                }, aggr='sum'))
            elif gnn_type == "graphsage":
                self.gnns.append(HeteroConv({
                    ('node', 'pos', 'node'): SAGEConv((-1,-1), emb_dim),
                    ('node', 'inv', 'node'): SAGEConv((-1,-1), emb_dim),
                }, aggr='sum'))
        
        self.pool = global_mean_pool

        ###List of batchnorms
        self.batch_norms = torch.nn.ModuleList()
        for layer in range(num_layer):
            self.batch_norms.append(torch.nn.BatchNorm1d(emb_dim))
            
        self.num_features = emb_dim
        self.cat_grep = True # cat the graph representation
        # self.aggregator = MultiAggregation(['sum', 'mean', 'max','min','softmax','median','std','var'])
        self.aggregator = MultiAggregation(['sum', 'mean', 'max','min'])

    #def forward(self, x, edge_index, edge_attr):
    def forward(self, *argv):
        # if len(argv) == 2:
        #     x, edge_index = argv[0], argv[1]
        if len(argv) == 1:
            data = argv[0]
            x_dict, edge_index_dict, batch = data.x_dict, data.edge_index_dict, data['node'].batch
        else:
            raise ValueError("unmatched number of arguments. The number is {}".format(len(argv)), argv)

        type_emeb = self.type_embedding(x_dict['node'][...,0])
        output_emeb = self.output_embedding(x_dict['node'][...,1])
        x_dict['node'] = type_emeb + output_emeb
        h_list = [x_dict['node']]
        for layer in range(self.num_layer):
            h = self.gnns[layer](x_dict, edge_index_dict)
            # h = self.batch_norms[layer](h['node'])
            h = h['node']
            #h = F.dropout(F.relu(h), self.drop_ratio, training = self.training)
            if layer == self.num_layer - 1:
                #remove relu for the last layer
                h = F.dropout(h, self.drop_ratio, training = self.training)
            else:
                h = F.dropout(F.relu(h), self.drop_ratio, training = self.training)
            h_list.append(h)
            x_dict['node'] = h

        ### Different implementations of Jk-concat
        if self.JK == "concat":
            node_representation = torch.cat(h_list, dim = 1)
        elif self.JK == "last":
            node_representation = h_list[-1]
        elif self.JK == "max":
            h_list = [h.unsqueeze_(0) for h in h_list]
            node_representation = torch.max(torch.cat(h_list, dim = 0), dim = 0)[0]
        elif self.JK == "sum":
            h_list = [h.unsqueeze_(0) for h in h_list]
            node_representation = torch.sum(torch.cat(h_list, dim=0), dim=0)[0]
        
        # print(node_representation.shape)  # shape = [B,N, D] B = batch size, N = # of nodes (1 million), D = embedding dim (512)
        h_graph = self.aggregator(node_representation, batch) # shape = [B,  D * 4], multi-aggregate all nodes using 8 different methods
        h_graph = h_graph.view(h_graph.size(0), -1, 4) # shape = [B, D, 4]
        # swap the last two dimensions
        h_graph = h_graph.permute(0, 2, 1) # shape = [B, 4, D]
        batch_mask = torch.ones((h_graph.shape[0],h_graph.shape[1]), dtype=torch.bool, device=batch.device)
        return h_graph, batch_mask
        exit()
        # h_graph = self.pool(node_representation, batch) # shape = [B, D]
        # batch_node, batch_mask = to_dense_batch(node_representation, batch) # shape = [B, n_max, D], 
        # batch_mask = batch_mask.bool()

        # if self.cat_grep:
        #     batch_node = torch.cat((h_graph.unsqueeze(1), batch_node), dim=1) # shape = [B, n_max+1, D]
        #     batch_mask = torch.cat([torch.ones((batch_mask.shape[0], 1), dtype=torch.bool, device=batch.device), batch_mask], dim=1)
        #     return batch_node, batch_mask
        # else:
        #     return batch_node, batch_mask, h_graph



class GNN_graphpred(torch.nn.Module):
    """
    Extension of GIN to incorporate edge information by concatenation.

    Args:
        num_layer (int): the number of GNN layers
        emb_dim (int): dimensionality of embeddings
        num_tasks (int): number of tasks in multi-task learning scenario
        drop_ratio (float): dropout rate
        JK (str): last, concat, max or sum.
        graph_pooling (str): sum, mean, max, attention, set2set
        gnn_type: gin, gcn, graphsage, gat
        
    See https://arxiv.org/abs/1810.00826
    JK-net: https://arxiv.org/abs/1806.03536
    """
    def __init__(self, num_layer, emb_dim, num_tasks, JK = "last", drop_ratio = 0, graph_pooling = "mean", gnn_type = "gin"):
        super(GNN_graphpred, self).__init__()
        self.num_layer = num_layer
        self.drop_ratio = drop_ratio
        self.JK = JK
        self.emb_dim = emb_dim
        self.num_tasks = num_tasks

        if self.num_layer < 2:
            raise ValueError("Number of GNN layers must be greater than 1.")

        self.gnn = GNN(num_layer, emb_dim, JK, drop_ratio, gnn_type = gnn_type)

        #Different kind of graph pooling
        if graph_pooling == "sum":
            self.pool = global_add_pool
        elif graph_pooling == "mean":
            self.pool = global_mean_pool
        elif graph_pooling == "max":
            self.pool = global_max_pool
        elif graph_pooling == "attention":
            if self.JK == "concat":
                self.pool = GlobalAttention(gate_nn = torch.nn.Linear((self.num_layer + 1) * emb_dim, 1))
            else:
                self.pool = GlobalAttention(gate_nn = torch.nn.Linear(emb_dim, 1))
        elif graph_pooling[:-1] == "set2set":
            set2set_iter = int(graph_pooling[-1])
            if self.JK == "concat":
                self.pool = Set2Set((self.num_layer + 1) * emb_dim, set2set_iter)
            else:
                self.pool = Set2Set(emb_dim, set2set_iter)
        else:
            raise ValueError("Invalid graph pooling type.")

        #For graph-level binary classification
        if graph_pooling[:-1] == "set2set":
            self.mult = 2
        else:
            self.mult = 1
        
        if self.JK == "concat":
            self.graph_pred_linear = torch.nn.Linear(self.mult * (self.num_layer + 1) * self.emb_dim, self.num_tasks)
        else:
            self.graph_pred_linear = torch.nn.Linear(self.mult * self.emb_dim, self.num_tasks)

    def from_pretrained(self, model_file):
        #self.gnn = GNN(self.num_layer, self.emb_dim, JK = self.JK, drop_ratio = self.drop_ratio)
        missing_keys, unexpected_keys = self.gnn.load_state_dict(torch.load(model_file))
        print(missing_keys)
        print(unexpected_keys)

    def forward(self, *argv):
        if len(argv) == 4:
            x, edge_index, edge_attr, batch = argv[0], argv[1], argv[2], argv[3]
        elif len(argv) == 1:
            data = argv[0]
            x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch
        else:
            raise ValueError("unmatched number of arguments.")

        node_representation = self.gnn(x, edge_index, edge_attr)

        return self.graph_pred_linear(self.pool(node_representation, batch))


if __name__ == "__main__":
    pass

