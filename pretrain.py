import warnings 
warnings.filterwarnings('ignore')

import sys
sys.path.append('.')
sys.path.append('..')
import yaml 
import argparse
import torch

from model.models import Model
from model.asr_framework import PGASR
from lib.dataloader import get_dataloader
from lib.utils import (
    init_seed,
    get_model_params,
    load_graph, 
)

def model_supervisor(args):
    init_seed(args.seed)
    if not torch.cuda.is_available():
        args.device = 'cpu'
    
    ## load dataset
    dataloader = get_dataloader(
        data_dir=args.data_dir, 
        dataset=args.dataset, 
        batch_size=args.batch_size, 
        test_batch_size=args.test_batch_size,
    )
    graph = load_graph(args.graph_file, device=args.device)
    args.num_nodes = len(graph)
    
    
    ## init models and set optimizers
    pre_model1 = Model(args).to(args.device)
    pre_model2 = Model(args).to(args.device)
    re_model = Model(args).to(args.device)

    pre_model1_parameters = get_model_params([pre_model1])
    pre_model2_parameters = get_model_params([pre_model2])
    re_model_parameters = get_model_params([re_model])

    optimizer1 = torch.optim.Adam(
        params=pre_model1_parameters, 
        lr=args.lr_init, 
        eps=1.0e-8, 
        weight_decay=args.weight_decay1, 
        amsgrad=False
    )

    optimizer2 = torch.optim.Adam(
        params=pre_model2_parameters, 
        lr=args.lr_init, 
        eps=1.0e-8, 
        weight_decay=args.weight_decay2, 
        amsgrad=False
    )

    optimizer3 = torch.optim.Adam(
        params=re_model_parameters, 
        lr=args.lr_init, 
        eps=1.0e-8, 
        weight_decay=args.weight_decay3, 
        amsgrad=False
    )

    p_gasr = PGASR(
        models=[pre_model1, pre_model2, re_model], 
        optimizers=[optimizer1, optimizer2, optimizer3], 
        dataloader=dataloader,
        graph=graph, 
        args=args
    )

    # pre-train pipeline
    if args.pretrain == True:
        p_gasr.pre_train(mode='pre-train1')
        p_gasr.infer_policy(mode='pre-train1')

    if args.pretrain == True:
        p_gasr.pre_train(mode='pre-train2')
        p_gasr.infer_policy(mode='pre-train2')



if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_filename', default='configs/NYCBike1.yaml', 
                    type=str, help='the configuration to use')
    args = parser.parse_args()
    
    print(f'Starting experiment with configurations in {args.config_filename}...')
    configs = yaml.load(
        open(args.config_filename), 
        Loader=yaml.FullLoader
    )

    args = argparse.Namespace(**configs)
    
    model_supervisor(args)