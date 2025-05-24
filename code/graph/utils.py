import sys
from torch_geometric.nn import GATConv, GCNConv, SAGEConv, GCN2Conv

gnn_layer_dict = {
    'gcn': GCNConv,
    'gcn2': GCN2Conv,
    'gat': GATConv,
    'sage': SAGEConv,
}

def print_weights(model):
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(name, param.shape)
    sys.stdout.flush()


def logger(info):
    fold, epoch = info['fold'], info['epoch']
    if epoch == 1 or epoch % 10 == 0:
        train_acc, test_acc = info['train_acc'], info['test_acc']
        print('{:02d}/{:03d}: Train Acc: {:.3f}, Test Accuracy: {:.3f}'.format(
            fold, epoch, train_acc, test_acc))
    sys.stdout.flush()


