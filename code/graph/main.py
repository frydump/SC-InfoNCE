import json
import argparse
from utils import logger
from datasets import get_dataset
from gnn import GNN
from train_eval import cross_validation_with_val_set
import warnings

warnings.filterwarnings("ignore", category=UserWarning)


def get_model(model_name, dataset, net_params):

    if model_name == 'GNN':
        net_params['in_dim'] = dataset.num_features
        net_params['out_dim'] = dataset.num_classes
        if "xg" in dataset[0]:
            net_params['use_xg'] = True
            net_params['xg_size'] = dataset[0].xg.size(1)
        else:
            net_params['use_xg'] = False

        return GNN(net_params)
    else:
        raise ValueError("Unknown model {}".format(model_name))


def run_experiment_finetune(config):

    dataset_name = config['dataset']
    net = config['params']['net']
    result_PATH = config['out_dir'] + dataset_name + '_' + net + '.res'
    result_feat = str(config)


    dataset = get_dataset(dataset_name, config['feature_params'])
    model_func = get_model
    
    cross_validation_with_val_set(
                    dataset,
                    model_func,
                    config = config,
                    logger=logger,
                    result_PATH=result_PATH, result_feat=result_feat)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default="nci1_gcn")
    parser.add_argument('--gpu', type=int, default=None)
    parser.add_argument('--T', type=float, default=None)
    parser.add_argument('--epoch', type=int, default=None)
    parser.add_argument('--alpha', type=float, default=None)
    parser.add_argument('--beta', type=float, default=None)
    parser.add_argument('--loss_type', type=int, default=0)
    args = parser.parse_args()

    config_file = './config/'+args.config+'.json'

    with open(config_file) as f:
        config = json.load(f)

    if args.gpu is not None:
        config['params']['gpu'] = args.gpu
    if args.T is not None:
        config["pre_params"]["T"]=args.T
    if args.epoch is not None:
        config["pre_params"]["epoch"]=args.epoch
    if args.loss_type is not None:
        config["pre_params"]["loss_type"] = args.loss_type
    if args.alpha is not None:
        config["pre_params"]["alpha"]=args.alpha
    if args.beta is not None:
        config["pre_params"]["beta"]=args.beta

    print(config)
    run_experiment_finetune(config)

