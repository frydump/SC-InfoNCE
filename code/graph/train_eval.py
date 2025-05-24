import sys
import time
import random
import numpy as np
import torch
import torch.nn.functional as F
from torch import tensor
from torch.optim import Adam
from sklearn.model_selection import StratifiedKFold
# from sklearn.model_selection import KFold
from torch_geometric.data import DataLoader
from utils import print_weights
from GCL_trainer import GraphContrastiveLearningTrainer
from torch_geometric.data import Batch

def cross_validation_with_val_set(dataset,
                                  model_func,
                                  config,
                                  logger=None,
                                  result_PATH=None, result_feat=None):
    train_params = config['params']

    epoch_select = train_params['epoch_select']
    batch_size = train_params['batch_size']
    epochs = train_params['epochs']
    weight_decay = train_params['weight_decay']
    with_eval_mode = train_params['with_eval_mode']
    gpu = train_params['gpu']
    step_gcl = config["pre_params"]["aug_step"]

    device = f'cuda:{gpu}' if torch.cuda.is_available() and gpu >=0 else 'cpu'
    device = torch.device(device)

    # setting seeds
    random.seed(train_params['seed'])
    np.random.seed(train_params['seed'])
    torch.manual_seed(train_params['seed'])
    if device.type == 'cuda':
        torch.cuda.manual_seed(train_params['seed'])

    assert epoch_select in ['val_max', 'test_max'], epoch_select

    gcl_trainer = GraphContrastiveLearningTrainer(config)
    dataset.set_aug_mode(config["pre_params"]["aug_mode"])
    dataset.set_aug_ratio(config["pre_params"]["aug_ratio"])

    if config["pre_params"]["loss_type"]==1 or config["pre_params"]["loss_type"]==6:
        dataset.set_get_k(3)
    else:
        dataset.set_get_k(2)

    print(dataset)

    folds=10

    val_losses, train_accs, test_accs, durations = [], [], [], []
    all_mad = []
    for fold, (train_idx, test_idx, val_idx) in enumerate(zip(*k_fold(dataset, folds, epoch_select))):

        model = model_func(train_params['net'], dataset, config['net_params']).to(device)
        ref_model = model_func(train_params['net'], dataset, config['net_params']).to(device)

        # train from scratch
        gcl_trainer.run_contrastive_learning(dataset, model, ref_model, device)
        dataset.set_get_k(0)



        train_dataset = dataset[train_idx]
        test_dataset = dataset[test_idx]
        val_dataset = dataset[val_idx]

        train_loader = DataLoader(train_dataset, batch_size, shuffle=True, num_workers=16)
        val_loader = DataLoader(val_dataset, batch_size, shuffle=False, num_workers=16)
        test_loader = DataLoader(test_dataset, batch_size, shuffle=False, num_workers=16)

        if fold == 0:
            print_weights(model)
        optimizer = Adam(model.parameters(), lr=train_params['lr'], weight_decay=weight_decay)
        if train_params["scheduler"]:
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.8, patience=15, min_lr=0.00001)


        # if torch.cuda.is_available():
        #     torch.cuda.synchronize(device)

        t_start = time.perf_counter()

        for epoch in range(1, epochs+1):

            # if train_params["scheduler"]:
            #     print("Current learning rate:", optimizer.param_groups[0]['lr'])
            train_loss, train_acc = train(model, optimizer, train_loader, device)
            train_accs.append(train_acc)
            val_losses.append(eval_loss(
                model, val_loader, device, with_eval_mode))
            if train_params["scheduler"]:
                scheduler.step(val_losses[-1])
            test_accs.append(eval_acc(
                model, test_loader, device, with_eval_mode))
            eval_info = {
                'fold': fold,
                'epoch': epoch,
                'train_loss': train_loss,
                'train_acc': train_accs[-1],
                'val_loss': val_losses[-1],
                'test_acc': test_accs[-1],
            }

            if logger is not None:
                logger(eval_info)


        # if torch.cuda.is_available():
        #     torch.cuda.synchronize()


        t_end = time.perf_counter()
        durations.append(t_end - t_start)

        if config['eval_mad']:
            model.eval()
            loader = DataLoader(dataset, len(dataset), shuffle=False)
            mad = None
            for data in loader:
                mad = model.forward_node_mad(data.to(device))
            all_mad.append(mad)

    duration = tensor(durations)
    train_acc, test_acc = tensor(train_accs), tensor(test_accs)
    val_loss = tensor(val_losses)
    train_acc = train_acc.view(folds, epochs)
    test_acc = test_acc.view(folds, epochs)
    val_loss = val_loss.view(folds, epochs)
    if epoch_select == 'test_max':  # take epoch that yields best test results.
        _, selected_epoch = test_acc.mean(dim=0).max(dim=0)
        selected_epoch = selected_epoch.repeat(folds)
    else:  # take epoch that yields min val loss for each fold individually.
        _, selected_epoch = val_loss.min(dim=1)
    test_acc = test_acc[torch.arange(folds, dtype=torch.long), selected_epoch]
    train_acc_mean = train_acc[:, -1].mean().item()
    test_acc_mean = test_acc.mean().item()
    test_acc_std = test_acc.std().item()
    duration_mean = duration.mean().item()

    print(train_acc_mean, test_acc_mean, test_acc_std, duration_mean)
    sys.stdout.flush()

    if len(all_mad) > 0 and config['eval_mad']:
        print('node MAD: '+str(float(torch.tensor(all_mad).mean())))

    with open(result_PATH, 'a+') as f:
        f.write(result_feat + ' ' + str(test_acc_mean) + '\n')


def k_fold(dataset, folds, epoch_select, n_splits=10):
    skf = StratifiedKFold(folds, shuffle=True, random_state=12345)

    test_indices, train_indices = [], []
    for _, idx in skf.split(torch.zeros(len(dataset)), dataset.data.y):
        test_indices.append(torch.from_numpy(idx))

    if epoch_select == 'test_max':
        val_indices = [test_indices[i] for i in range(folds)]
    else:
        val_indices = [test_indices[i - 1] for i in range(folds)]

    skf_semi = StratifiedKFold(n_splits, shuffle=True, random_state=12345)
    for i in range(folds):
        train_mask = torch.ones(len(dataset), dtype=torch.uint8)
        train_mask[test_indices[i].long()] = 0
        train_mask[val_indices[i].long()] = 0
        idx_train = train_mask.nonzero(as_tuple=False).view(-1)

        for _, idx in skf_semi.split(torch.zeros(idx_train.size()[0]), dataset.data.y[idx_train]):
            idx_train = idx_train[idx]
            break

        train_indices.append(idx_train)

    return train_indices, test_indices, val_indices



def num_graphs(data):
    if data.batch is not None:
        return data.num_graphs
    else:
        return data.x.size(0)


def train(model, optimizer, loader, device):
    model.train()

    total_loss = 0
    correct = 0
    for data in loader:
        optimizer.zero_grad()
        if type(data) == list:
            data = data[0]
        #data = Batch.from_data_list(data).to(device)
        data = data.to(device)

        out = model(data)
        loss = F.nll_loss(out, data.y.long().view(-1))
        pred = out.max(1)[1]
        correct += pred.eq(data.y.view(-1)).sum().item()
        loss.backward()
        total_loss += loss.item() * num_graphs(data)
        optimizer.step()
    return total_loss / len(loader.dataset), correct / len(loader.dataset)


def eval_acc(model, loader, device, with_eval_mode):
    if with_eval_mode:
        model.eval()

    correct = 0
    for data in loader:
        data = data[0].to(device)
        with torch.no_grad():
            pred = model(data).max(1)[1]
        correct += pred.eq(data.y.view(-1)).sum().item()
    return correct / len(loader.dataset)


def eval_loss(model, loader, device, with_eval_mode):
    if with_eval_mode:
        model.eval()

    loss = 0
    for data in loader:
        data = data[0].to(device)
        with torch.no_grad():
            out = model(data)
        loss += F.nll_loss(out, data.y.long().view(-1), reduction='sum').item()
    return loss / len(loader.dataset)

