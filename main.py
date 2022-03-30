from torch_geometric.datasets import Planetoid   (dataset)
import torch
import torch.nn.functional as F
#Convolution functions, pooling funtions etc

from torch_geometric.nn import MessagePassing
#we can import data from here

from torch_geometric.utils import add_self_loops, degree
#Adds a self-loop  to every node  in the graph given by edge_index.
#Computes the (unweighted) degree of a given one-dimensional index tensor.

import networkx as nx
#studies function of complex networks
import numpy as np
#NumPy offers comprehensive mathematical functions

import matplotlib.pyplot as plt
#for visualization

class GCNConv(MessagePassing):
    def __init__(self, in_channels, out_channels): #(passing perameters)
        super(GCNConv, self).__init__(aggr='add')  # "Add" aggregation
        self.lin = torch.nn.Linear(in_channels, out_channels) #(recursive calling)
        #A recursive function is a function defined in terms of itself via self-referential
        #expressions. This means that the function will continue to call itself and repeat its behavior until some condition is met to return a result.
        #Applies a linear transformation to the incoming data

    def forward(self, x, edge_index):
        # Step 1: Add self-loops # (recursive call)
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))  #(function calling)
        #print(add_self_loops)
        # Step 2: Multiply with weights
        x = self.lin(x) #each value is assigned a weight randomly

        # Step 3: Calculate the normalization
        row, col = edge_index #
        deg = degree(row, x.size(0), dtype=x.dtype)
        deg_inv_sqrt = deg.pow(-0.5)
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]

        # Step 4: Propagate the embeddings to the next layer
        return self.propagate(edge_index, size=(x.size(0), x.size(0)), x=x,
                              norm=norm)

    #Propagate: Decides whether a log should be propagated to the logger's parent. By default, its value is True
    def message(self, x_j, norm):
        # Normalize node features.
        return norm.view(-1, 1) * x_j


class Net(torch.nn.Module):
    #Base class for all neural network modules.
    def __init__(self, dataset):
        super(Net, self).__init__()
        self.conv1 = GCNConv(dataset.num_node_features, 16)
        self.conv2 = GCNConv(16, dataset.num_classes)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)

        return F.log_softmax(x, dim=1)


def plot_dataset(dataset):
    edges_raw = dataset.data.edge_index.numpy()
    edges = [(x, y) for x, y in zip(edges_raw[0, :], edges_raw[1, :])]
    #The zip() function takes iterables (can be zero or more), aggregates them in a tuple
    labels = dataset.data.y.numpy()

    G = nx.Graph()
    G.add_nodes_from(list(range(np.max(edges_raw))))
    G.add_edges_from(edges)
    plt.subplot(111)
    ##There is only one subplot or graph
    `
    
    options = {
                'node_size': 30,
                'width': 0.2,
    }  # fine tuning
    nx.draw(G, with_labels=False, node_color=labels.tolist(), cmap=plt.cm.tab10, font_weight='bold', **options)
    plt.show()


def test(data, train=True):
    model.eval()

    correct = 0
    pred = model(data).max(dim=1)[1]

    if train:
        correct += pred[data.train_mask].eq(data.y[data.train_mask]).sum().item()
        return correct / (len(data.y[data.train_mask]))
    else:
        correct += pred[data.test_mask].eq(data.y[data.test_mask]).sum().item()
        return correct / (len(data.y[data.test_mask]))


def train(data, plot=False):
    train_accuracies, test_accuracies = list(), list()
    for epoch in range(100):
            model.train()
            optimizer.zero_grad()
            #Sets the gradients of all optimized torch. Tensor s to zero.
            
            out = model(data)
            loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
            loss.backward()
            #when we call loss. backward() , the whole graph is differentiated w.r.t. the loss
            optimizer.step()
            #makes the optimizer iterate over all parameters (tensors)

            train_acc = test(data)
            test_acc = test(data, train=False)

            train_accuracies.append(train_acc)
            test_accuracies.append(test_acc)
            print('Epoch: {:03d}, Loss: {:.5f}, Train Acc: {:.5f}, Test Acc: {:.5f}'.
                  format(epoch, loss, train_acc, test_acc))

    if plot:
        plt.plot(train_accuracies, label="Train accuracy")
        plt.plot(test_accuracies, label="Validation accuracy")
        plt.xlabel("# Epoch")
        plt.ylabel("Accuracy")
        plt.legend(loc='upper right')
        plt.show()


if __name__ == "__main__":
    dataset = Planetoid(root='/tmp/Cora', name='Cora')
    
    plot_dataset(dataset)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = Net(dataset).to(device)
    data = dataset[0].to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    #Adam is a replacement optimization algorithm for stochastic gradient descent for training deep learning models.
    #Adam combines the best properties of the AdaGrad and RMSProp algorithms to provide an optimization algorithm that can handle sparse gradients on noisy problems.
    #Weight decay is a regularization technique by adding a small penalty, usually the L2 norm of the weights (all the weights of the model), to the loss function.
    #it is also known as the Euclidean norm as it is calculated as the Euclidean distance from the origin
    train(data, plot=True)
