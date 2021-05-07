import random
import pathlib
import argparse
import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

import numpy as np

from .networks import *
from .evaluation import *
from .datasets import *
from .utils import *


def train():
    logger.debug(f'loading {args.dataset} dataset...')
    dataset = DlibDataset(
        args.dataset, 
        seed=args.seed
    )

    logger.debug(f'initializing dataloader...')
    dataloader = DataLoader(
        dataset, 
        batch_size=args.batch_size,
        num_workers=2,
        pin_memory=True if args.cuda else False,
        drop_last=True
    )
    logger.debug(f'initializing and loading networks to {device}...')
    D = Discriminator(dataset.C).to(device)
    G = Generator(args.c_dim, args.z_dim, dataset.C).to(device)
    E = Encoder(dataset.C, args.c_dim).to(device)
    Pc = Predictor(args.c_dim).to(device)
    Pr = Predictor(args.c_dim).to(device)
    
    logger.debug(f'initializing optimizers...')
    optim_D = optim.Adam(
        D.parameters(),
        lr=2e-3,
        betas=(.5, .999)
    )
    optim_G = optim.Adam(
        G.parameters(),
        lr=1e-3,
        betas=(.5, .999)
    )
    optim_E = optim.Adam(
        [
            {'params': G.parameters(), 'lr': 1e-3},
            {'params': E.parameters()},
            {'params': Pc.parameters()},
            {'params': Pr.parameters()},
        ],
        lr=2e-3,
        betas=(.5, .999)
    )
    
    one = torch.ones(args.batch_size, 1).to(device)
    zero = torch.zeros(args.batch_size, 1).to(device)
    
    running_loss_D = 0.0
    running_loss_G = 0.0
    running_loss_E = 0.0
    running_loss_Pc = 0.0
    running_loss_Pr = 0.0

    for epoch in range(args.epochs):
        D.train()
        G.train()
        E.train()
        Pc.train()
        Pr.train()

        for i, (x, y) in enumerate(dataloader):
            itr = epoch*len(dataloader) + i
            x = x.to(device)
            
            # discriminator
            c = torch.empty(args.batch_size, args.c_dim).uniform_(-1, 1).to(device)
            z = torch.randn(args.batch_size, args.z_dim).to(device)
            loss_D  = F.binary_cross_entropy_with_logits(D(x), one)
            loss_D += F.binary_cross_entropy_with_logits(D(G(c, z).detach()), zero)
            optim_D.zero_grad() 
            loss_D.backward()
            optim_D.step()
            running_loss_D += loss_D.item()
            
            # generator
            c = torch.empty(args.batch_size, args.c_dim).uniform_(-1, 1).to(device)
            z = torch.randn(args.batch_size, args.z_dim).to(device)
            loss_G  = F.binary_cross_entropy_with_logits(D(G(c, z)), one)
            optim_G.zero_grad()
            loss_G.backward()
            optim_G.step()
            running_loss_G += loss_G.item()
            
            # encoder
            c = E(x).detach()
            cc, mc = consistent_like(c)
            cr, mr = restrictive_like(c)
            z = torch.randn(args.batch_size, args.z_dim).to(device)
            zc = torch.randn(args.batch_size, args.z_dim).to(device)
            zr = torch.randn(args.batch_size, args.z_dim).to(device)
            lc = mc.argmax(dim=1)
            lr = mr.argmin(dim=1)
            e = E(G(c, z))
            ec = E(G(cc, zc))
            er = E(G(cr, zr))
            pc1 = Pc(e, cc)
            pc2 = Pc(ec, c)
            pr1 = Pr(e, cr)
            pr2 = Pr(er, c)
            
            loss_E = (F.mse_loss(e, c) + F.mse_loss(ec, cc) + F.mse_loss(er, cr)) / 3
            loss_Pc = (F.cross_entropy(pc1, lc) + F.cross_entropy(pc2, lc)) / 2
            loss_Pr = (F.cross_entropy(pr1, lr) + F.cross_entropy(pr2, lr)) / 2
            loss_EP = args.beta * loss_E + args.alpha * (loss_Pc + loss_Pr)
            optim_E.zero_grad()
            loss_EP.backward()
            optim_E.step()
            running_loss_E += loss_E.item()
            running_loss_Pc += loss_Pc.item()
            running_loss_Pr += loss_Pr.item()
            
            
            if itr % args.log_interval == 0:
                if itr != 0:
                    running_loss_D /= args.log_interval
                    running_loss_G /= args.log_interval
                    running_loss_E /= args.log_interval
                    running_loss_Pc /= args.log_interval
                    running_loss_Pr /= args.log_interval

                
                logger.info(
                    f'[{epoch}/{i}] ({itr}): '
                    f'loss_D = {running_loss_D}, '
                    f'loss_G = {running_loss_G}, '
                    f'loss_E = {running_loss_E}, '
                    f'loss_Pc = {running_loss_Pc}, '
                    f'loss_Pr = {running_loss_Pr}'
                )
                
                writer.add_scalar('Loss/D', running_loss_D, itr)
                writer.add_scalar('Loss/G', running_loss_G, itr)
                writer.add_scalar('Loss/E', running_loss_E, itr)
                writer.add_scalar('Loss/Pc', running_loss_Pc, itr)                
                writer.add_scalar('Loss/Pr', running_loss_Pr, itr)
                writer.flush()
                
                running_loss_D = 0.0
                running_loss_G = 0.0
                running_loss_E = 0.0
                running_loss_Pc = 0.0
                running_loss_Pr = 0.0
        
        D.eval()
        G.eval()
        E.eval()
        Pc.eval()
        Pr.eval()
        
        c = torch.empty(8, args.c_dim).uniform_(-1, 1).to(device)
        z = torch.randn(8, args.z_dim).to(device)
        logger.info('logging samples...')
        with torch.no_grad():
            writer.add_images('x', G(c, z), itr)
                
        # evaluate
        logger.info('evaluation...')
        res = compute_metrics(
            dataset.dataset, 
            representor(E, device),
            ['factor_vae_metric']
        )
        logger.info(res)
        writer.add_scalar('Metrics/FactorVAE', res['factor_vae_metric']['eval_accuracy'], itr)
        writer.flush()
        
        # save checkpoint
        logger.info('saving checkpoint...')
        checkpoint = {
            'epoch': epoch,
            'i': i,
            'optim_D': optim_D.state_dict(),
            'optim_G': optim_G.state_dict(),
            'optim_E': optim_E.state_dict(),
            'D': D.state_dict(),
            'G': G.state_dict(),
            'E': E.state_dict(),
            'Pc': Pc.state_dict(),
            'Pr': Pr.state_dict(),
        }
        torch.save(checkpoint, str(checkpoints_dir/'checkpoint.pt'))
        

parser = argparse.ArgumentParser()
parser.add_argument('--experiment', 
                    type=str, default='debug', 
                    help='Experiment name (default: "debug").')
parser.add_argument('--dataset', 
                    type=str, default='dsprites_full', 
                    help='Dataset name (default: "dsprites_full").')
parser.add_argument('--seed', 
                    type=int, default=0,
                    help='Random seed (default: 0).')
parser.add_argument('--cuda',
                    action='store_true', default=False,
                    help='Enable CUDA.')
subparsers = parser.add_subparsers()
train_parser = subparsers.add_parser(
    'train',
    help='Train a model.'
)
train_parser.add_argument('--batch-size',  
                          type=int, default=256, 
                          help='Batch size (default: 256).')
train_parser.add_argument('--epochs',  
                          type=int, default=150, 
                          help='Number of epochs (default: 150).')
train_parser.add_argument('--c-dim',  
                          type=int, default=10, 
                          help='Code dimmension (default: 10).')
train_parser.add_argument('--z-dim',
                          type=int, default=0, 
                          help='Noise dimmension (default: 0).')
train_parser.add_argument('--alpha',  
                          type=float, default=2.0, 
                          help='Alpha parameter (default: 2.0).')
train_parser.add_argument('--beta',  
                          type=float, default=1.0, 
                          help='Beta parameter (default: 1.0).')
train_parser.add_argument('--log-interval', 
                          type=int, default=100,
                          help='Log interval (default: 100).')
train_parser.set_defaults(func=train)


if __name__ == '__main__':
    args = parser.parse_args()
    results_dir = pathlib.Path('./results/')/args.experiment/args.dataset/str(args.seed)
    results_dir.mkdir(parents=True, exist_ok=True)
    
    checkpoints_dir = results_dir/'checkpoints'
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    
    evaluations_dir = results_dir/'evaluations'
    evaluations_dir.mkdir(parents=True, exist_ok=True)
    
    logs_dir = results_dir/'logs'
    logs_dir.mkdir(parents=True, exist_ok=True)
    
    logger = logging.getLogger(args.experiment)
    fhandler = logging.FileHandler(filename=logs_dir/(args.func.__name__+'.logger'), mode='w')
    formatter = logging.Formatter('%(levelname)s: %(message)s')
    fhandler.setFormatter(formatter)
    logger.addHandler(fhandler)
    logger.setLevel(logging.DEBUG)
    logger.debug(args)
    
    logger.debug(f'initializing SummaryWriter...')
    writer = SummaryWriter(
        log_dir=str(logs_dir/'tensorboard')
    )
    
    logger.debug(f'setting random seed {args.seed}...')
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    logger.debug(f'initializing device...')
    device = torch.device("cuda" if args.cuda else "cpu")
    
    
    args.func()
    
    fhandler.close()
    writer.close()
