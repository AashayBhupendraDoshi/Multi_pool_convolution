# This is for graph classification tasks for graph datasets with 
# only node embeddings and class labels (no edge embeddings and edge labels).

#  Here we perfrom top_k pooling in the final layer two times and take
#  the mean pool from every top-k pool in the final layer
import sys
import os
import os.path as osp
import random
import numpy as np
import pandas as pd
from tqdm import tqdm
from collections import OrderedDict
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.transforms as T
from torch_geometric.datasets import TUDataset
from torch_geometric.data import DataLoader
# Ideally you should use InstanceNorm but since this is an older version of pytorch_geometric
# We are using batchnorm instead
from torch_geometric.nn import GraphConv, SAGEConv, TAGConv, TopKPooling, ChebConv, GlobalAttention 
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# For reproducibility
def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
set_seed(7)

#Argparase arguements
ap = argparse.ArgumentParser()
ap.add_argument('-NumBlocks', '--b', required=True,
                help = 'Number of Blocks on Network', type = int)
ap.add_argument('-NumHeads', '--h', required=True,
                help = 'Number of Heads in each block. NOTE: NumHeads MUST BE A FACTOR OF EMBEDDING SIZE', type = int)
ap.add_argument('-Embedding Size', '--emb', required=True,
                help = 'Embedding Size. NOTE: EMBEDDING SIZE MUST BE A MULTIPLE OF NumHeads', type = int)
ap.add_argument('-alpha', '--alpha', required=False, default = 0,
                help = 'Weightage for Orthogonality Loss', type = float)
ap.add_argument('-Dataset', '--dt', required=True,
                help = 'Dataset to be used', type = str)
arguments = ap.parse_args()


dataset_directory = {
                    "collab": "COLLAB" ,
                    "triangles": "TRIANGLES" ,
                    "deezer_ego_nets": "deezer_ego_nets" ,
                    "github_stargazers": "github_stargazers" ,
                    "imdb-binary": "IMDB-BINARY" ,
                    "imdb-multi": "IMDB-MULTI" ,
                    "reddit-binary": "REDDIT-BINARY",
                    "reddit-multi-5k": "REDDIT-MULTI-5K" ,
                    "reddit-multi-12k": "REDDIT-MULTI-12K" ,
                    "reddit_threads": "Reddit_threads" ,
                    "twitch_egos": "Twitch_egos" ,
                    "yeast": "Yeast" ,
                    "yeasth": "YeastH",
                    "reddit_threads": "reddit_threads",
                    "twitter-real-graph-partial": "TWITTER-Real-Graph-Partial"
                    }


if arguments.dt.lower() not in dataset_directory.keys():
    print("Dataset does not exist. Please choose from the given datasets:")
    print(dataset_directory.keys())
    sys.exit()
dataset_name = dataset_directory[arguments.dt.lower()]


# Here we are defining a subGraph Convolution Block
# Here there are two possible architextures possible:
# 1) Perform convolution -> Break the graph -> Different Convolution for every Sub-graph -> Combine
# 2) Break the graph -> Different Convolution for every Sub-graph -> Combine
# Here we perform the latter with resdual connections

# First we define a single head version, which is essentially Top-K pooling
# followed by convolution.
# Multi-head version of this will essentially be multiple single-head versions
# running in parallel
class single_head(torch.nn.Module):
    
    def __init__(self, in_channels, out_channels, ratio=0.7, min_score=0.7):
        super(single_head, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.ratio = ratio
        self.min_score = min_score
        self.pool = TopKPooling(self.in_channels, min_score = self.min_score)
        self.conv = GraphConv(self.in_channels, self.out_channels, aggr='max', bias=True)
        
    def forward(self, data_dict):
        # data_dict is a dictionary with the following keys
        # 'x': embeddings
        # 'edge_index': edge_index
        # 'batch': batch
        # 'new_x': new values. Will be initialized with x itself
        x, edge_index, _, batch, perm, score = self.pool(data_dict["x"], data_dict["edge_index"], None, data_dict["batch"])
        x = F.leaky_relu(self.conv(x, edge_index), negative_slope=0.01)
        
        # TopK pooling re-numbers graph after pruning it.
        # Hence the new node embeddings from the subgraph need to be scattered
        # to the bigger graph.
        new_x = torch.zeros(data_dict["x"].shape[0], self.out_channels).to(device)
        new_x.scatter(0,perm.repeat(self.out_channels,1).transpose(0,1), x)
        # Dimensions of x (node embeddins) will be [num_nodes, node_embeddins]
        # Batch is accounted for in num_nodes
        # Concatenating new_x to data_dict['new_x'] and returning the dict
        data_dict['new_x'] = torch.cat([data_dict['new_x'],new_x], dim = -1)
        return data_dict


# Define mult-head version along with orthogonal loss
# There are two ways to build this model
# 1) x -> Conv -> SubgraphConv -> new_x ,or
# 2) x -> SubgraphConv -> new_x
# Here we perform the latter
# To maintain constant dimensionality, we perform concatenation
# followed by batch-Normalization, similar to ResNet architecture
# A multi-head block is essentially multiple (num_heads) single-head blocks
# running in parallel.
# The method employs a dict structure where initial as well final values are stored (in series)
# Every head takes the single values from the dict, processes it and stores(concatenates) the
# outputs back into the dict. Hence the structure is effectively a parallel graph,
# even though it is processed sequentially(,i.e., in series)

class multi_head_block(torch.nn.Module):
    
    def __init__(self, in_channels, out_channels, num_heads):
        super(multi_head_block, self).__init__()
        # Make sure out_channels*num_heads = in_channels
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_heads = num_heads
        #self.normalization = InstanceNorm(in_channels)
        self.block_body = nn.Sequential(
                *[single_head(in_channels , out_channels) for _ in range(self.num_heads)]
            ) 
        # Adding batch_normalization. Since you already have outputs from the previous layers
        # stores, the total size of data_dict['new_x'] will be 2*embedding_size
        self.bn1 = nn.BatchNorm1d(1*self.in_channels)
        # intialize block orthogonality loss
        _ = self.block_orthogonality_loss()
        
    def forward(self, data_dict):
        # data_dict is a dictionary with the following keys
        # 'x': embeddings
        # 'edge_index': edge_index
        # 'batch': batch
        # 'new_x': new values. Will be initialized with x itself
        data_dict = self.block_body(data_dict)
        data_dict['new_x'][:,self.in_channels:] = self.bn1(data_dict['new_x'][:,self.in_channels:])
        # Adding residual connection by addition
        # Since you already have outputs from the previous layers
        # stores, the total size of data_dict['new_x'] will be 2*embedding_size
        # Hence data_dict['new_x'] needs to be reduced first to only carry new embeddings
        # Use data_dict['new_x'] to build x
        # Add residue layer in build data_dict['new_x'] itself
        data_dict['new_x'] = 0.5*(data_dict['new_x'][:,:self.in_channels] + 0.5*data_dict['new_x'][:,self.in_channels:])
        ########################################################################################
        # Embedding Updation
        data_dict['x'] = data_dict['new_x']
        #data_dict['x'] = self.normalization(data_dict['x'], data_dict['batch'])
        ########################################################################################
        return data_dict

    def block_orthogonality_loss(self):
        # Define orthogonality loss
        # Store the vectors of all the pooling functions into a list
        self.pooling_vectors = [self.block_body[i].state_dict()['pool.weight'][0] for i in range(self.num_heads)]
        # Convert the list into a tensor. This will give a tensor of dimensions [num_heads, in_channels]
        self.pooling_vectors = torch.stack(self.pooling_vectors)
        # Finding the of the vectors with themselves. We cannot directly
        # consider it to be equal to number of heads since the vectors are not normalized
        buff = torch.sum(self.pooling_vectors*self.pooling_vectors)
        # The above operations will lead to a consolidated matrix [v1, v2, v3, v4,......, vn]
        # of top-K pooling vectors of size [num_heads, in_features]
        # We will tile it to 'num_heads' times so that we get a tensor of size
        # [num_heads, num_heads, in_feature]
        # Then transpose it to get tensot of size [in_feature,num_heads, num_heads]
        self.pooling_vectors = self.pooling_vectors.repeat(self.num_heads,1,1).transpose(0,1)
        self.topK_orthogonality_loss = torch.sum(self.pooling_vectors* self.pooling_vectors) - buff
        return self.topK_orthogonality_loss




# Similar to single_head and multi_head block defination for the main network,
# we define a single head and multi_head version of the final classification block
# Here in the classification block we use multiple heads, but we do not merge them back
# Instead we use these heads for classification, each head
# pertaining to a particular class.

# Here we are defining a subGraph Convolution Block
# Here there are two possible architextures possible:
# 1) Perform convolution -> Break the graph -> Different Convolution for every Sub-graph
# 2) Break the graph -> Different Convolution for every Sub-graph
# Here we perform the latter with resdual connections

# First we define a single head version, which is essentially Top-K pooling
# followed by convolution.
# Multi-head version of this will essentially be multiple single-head versions
# running in parallel

class classification_block_single_head(torch.nn.Module):
    
    def __init__(self, in_channels, ratio=0.1, min_score=0.8):
        super(classification_block_single_head, self).__init__()
        self.in_channels = in_channels
        self.out_channels = 1
        self.ratio = ratio
        self.min_score = min_score
        self.pool = TopKPooling(self.in_channels, min_score = self.min_score)
        #self.conv = GraphConv(self.in_channels, self.out_channels)
        self.fc1 = nn.Sequential(
                    nn.Linear(self.in_channels, self.in_channels),
                    nn.LeakyReLU(negative_slope=0.01),
                    nn.Linear(self.in_channels, 1))
        
        self.attention_calc = nn.Sequential(OrderedDict([
                                          ('fc1', nn.Linear(self.in_channels, self.in_channels)),
                                          ('leaky_relu', nn.LeakyReLU()),
                                          ('fc2', nn.Linear(self.in_channels, 1))
                                            ]))
        # Here gate_nn should map the node embeddings to attention values
        # and nn furthur transform the node embeddings, although one must note that
        # the originalembeddings will be used for gate_nn
        self.GA = GlobalAttention(gate_nn = self.attention_calc, nn=None)
        
    def forward(self, data_dict):
        # data_dict is a dictionary with the following keys
        # 'x': embeddings
        # 'edge_index': edge_index
        # 'batch': batch
        # 'new_x': new values. Will be initialized with x itself
        x, edge_index, _, batch, perm, score = self.pool(data_dict["x"], data_dict["edge_index"], None, data_dict["batch"])
        #x = F.relu(self.conv(x, edge_index))
        x = self.GA(x, batch)
        x = self.fc1(x)
        data_dict['final_output'] = torch.cat([data_dict['final_output'],x], dim = -1)
        return data_dict

# Here we define the complete classifiation block
# We take the complete embeddings from the body of the network
# and pool out multiplesub-graphs; one subgraph per class
# Then we make classification purely based in this subgraph
class classification_block_multi_head(torch.nn.Module):
    
    def __init__(self, in_channels, num_classes):
        super(classification_block_multi_head, self).__init__()
        # Make sure out_channels*num_heads = in_channels
        self.in_channels = in_channels
        self.num_heads = num_classes
        #self.normalization = InstanceNorm(in_channels)
        self.block_body = nn.Sequential(
                *[classification_block_single_head(in_channels) for _ in range(self.num_heads)]
            ) 
        # intialize block orthogonality loss
        self.logSoftmax = nn.LogSoftmax(dim=-1)
        _ = self.block_orthogonality_loss()
        
    def forward(self, data_dict, batch_size):
        # data_dict is a dictionary with the following keys
        # 'x': embeddings
        # 'edge_index': edge_index
        # 'batch': batch
        # 'new_x': new values. Will be initialized with x itself
        # 'final_output': this is a new variable defined to store the final
        # score for all classes. 
        data_dict['final_output'] = torch.zeros(batch_size,1).to(device)
        data_dict = self.block_body(data_dict)
        data_dict['final_output'] = self.logSoftmax(data_dict['final_output'][:,1:])
        
        return data_dict

    def block_orthogonality_loss(self):
        # Define orthogonality loss
        # Store the vectors of all the pooling functions into a list
        self.pooling_vectors = [self.block_body[i].state_dict()['pool.weight'][0] for i in range(self.num_heads)]
        # Convert the list into a tensor. This will give a tensor of dimensions [num_heads, in_channels]
        self.pooling_vectors = torch.stack(self.pooling_vectors)
        # Finding the of the vectors with themselves. We cannot directly
        # consider it to be equal to number of heads since the vectors are not normalized
        buff = torch.sum(self.pooling_vectors*self.pooling_vectors)
        # The above operations will lead to a consolidated matrix [v1, v2, v3, v4,......, vn]
        # of top-K pooling vectors of size [num_heads, in_features]
        # We will tile it to 'num_heads' times so that we get a tensor of size
        # [num_heads, num_heads, in_feature]
        # Then transpose it to get tensot of size [in_feature,num_heads, num_heads]
        self.pooling_vectors = self.pooling_vectors.repeat(self.num_heads,1,1).transpose(0,1)
        self.topK_orthogonality_loss = torch.sum(self.pooling_vectors* self.pooling_vectors) - buff
        return self.topK_orthogonality_loss


class Net(torch.nn.Module):
    def __init__(self, dataset_num_features, embedding_size, num_heads, num_blocks, num_classes):
        super(Net, self).__init__()
        # NOTE: embeddins_size must be a multiple of num_heads
        # Here embedding size is the size of output you want
        # dataset_num_features is the feature size of the input data
        self.dataset_num_features = dataset_num_features
        self.in_channels = embedding_size
        self.out_channels = int(self.in_channels/num_heads)
        self.num_classes = num_classes
        self.num_heads = num_heads
        self.num_blocks = num_blocks
        # Lets define the network now
        # Here we first transform the eature size from dataset_num_features
        # to embedding_size (using a basic GCN layer). And then be apply blocks on top of this
        # You can also apply a multi_head_block instead of a conv block for the same
        self.conv0 = GraphConv(self.dataset_num_features , self.in_channels, aggr='max', bias=True)
        self.bn0 = nn.BatchNorm1d(self.in_channels)
        self.network_body = nn.Sequential(
                *[multi_head_block(self.in_channels , self.out_channels, self.num_heads) for _ in range(self.num_blocks)]
            )
        self.classification_block = classification_block_multi_head(self.in_channels, self.num_classes)
        # Initialize network orthogonality loss
        _ = self.network_orthogonality_loss()
        
    def forward(self, data, batch_size):
        # Packing data into a dict
        # Packing is necessary for nn.sequential
        data_dict = {
            'x':data.x,
            'edge_index':data.edge_index,
            'batch':data.batch,
            'new_x':data.x
        }
        data_dict['new_x'] = self.bn0(F.leaky_relu(self.conv0(data['x'], data['edge_index']), negative_slope=0.01))
        data_dict['x'] = data_dict['new_x']
        data_dict = self.network_body(data_dict)
        # Classification
        data_dict = self.classification_block(data_dict, batch_size)
        return data_dict["final_output"]

    def network_orthogonality_loss(self):
        # Ortho_loss for body
        self.ortho_network_body = sum([self.network_body[i].block_orthogonality_loss() for i in range(self.num_blocks)])
        # Ortho_loss for classification block
        self.ortho_loss_classifier_block = self.classification_block.block_orthogonality_loss()
        # Total
        self.ortho_loss = self.ortho_network_body + self.ortho_loss_classifier_block
        return self.ortho_loss



dataset = TUDataset('/home/aashay/Desktop/pytorch_geometric/DDP/Datasets/TU/'+ dataset_name, name = dataset_name, use_node_attr = True)
# Sine these datasets are purely topoligal, we need to add artificial features ourselves
# This is necessary for propogation of messages. Here we add a feature of [1] to every node
# Here we need to add a dataset.data.x for features
# and dataset.slices.x to define which nodes belong to which graph
# To calculate both we need to find number of nodes in each graph.
slices_x = [0]
num_nodes = 0
for i in range(len(dataset)):
    buff = slices_x[-1] + dataset[i].num_nodes
    slices_x += [buff]
    num_nodes += dataset[i].num_nodes
#print(num_nodes, num_nodes/len(dataset))
dataset.data["x"] = torch.ones(num_nodes,1)
dataset.slices["x"] = torch.tensor(slices_x)
########################################################################################################
dataset = dataset.shuffle()
n = len(dataset) // 5
test_dataset = dataset[:n]
validation_dataset = dataset[n:2*n]
train_dataset = dataset[2*n:]
test_loader = DataLoader(test_dataset, batch_size=64)
validation_loader = DataLoader(validation_dataset, batch_size=64)
train_loader = DataLoader(train_dataset, batch_size=64)


model = Net(int(dataset.num_features),
            arguments.emb, 
            arguments.h, 
            arguments.b, 
            int(dataset.num_classes)).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)
# Loss is degined as:
# Loss = accuracy_loss + alpha*orthogonality_loss
alpha = arguments.alpha
train_loss = []
train_list = []
test_list = []
validation_list = []
    

def train(epoch):
    model.train()

    loss_all = 0
    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        output = model(data, data.y.shape[0])
        loss = F.nll_loss(output, data.y) + alpha*model.network_orthogonality_loss()
        loss.backward()
        loss_all += data.num_graphs * loss.item()
        optimizer.step()
    return loss_all / len(train_dataset)


def test(loader):
    model.eval()

    correct = 0
    for data in loader:
        data = data.to(device)
        pred = model(data, data.y.shape[0]).max(dim=1)[1]
        correct += pred.eq(data.y).sum().item()
    return correct / len(loader.dataset)



for epoch in tqdm(range(1, 201)):
    loss = train(epoch)
    train_acc = test(train_loader)
    validation_acc = test(validation_loader)
    test_acc = test(test_loader)
    train_loss += [loss]
    train_list += [train_acc]
    test_list += [test_acc]
    validation_list += [validation_acc]
    #print('Epoch: {:03d}, Loss: {:.3f}, Train Acc: {:.3}, Validation Acc: {:.3f} Test Acc: {:.3f},'.
    #      format(epoch, loss, train_acc, validation_acc, test_acc))
    

df = pd.DataFrame({"Train_Loss": train_loss,
                    "Train Accuracy": train_list,
                    "Validation Accuracy": validation_list,
                    "Test Accuracy": test_list
                    })


name = str(arguments.b) + '_' + str(arguments.h) + '_' + str(arguments.emb) + '_' + str(int(100*arguments.alpha))
df.to_csv("./Performance/" + str(arguments.dt.upper()) + '/' + name + ".csv")