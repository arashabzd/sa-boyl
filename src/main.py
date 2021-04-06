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
    logger.debug(f'{args.dataset} dataset loaded.')

    logger.debug(f'initializing dataloader...')
    dataloader = DataLoader(
        dataset, 
        batch_size=args.batch_size,
        num_workers=2,
        pin_memory=True if args.cuda else False,
        drop_last=True
    )
    logger.debug(f'dataloader initialized.')
    
    logger.debug(f'initializing and loading networks to {device}...')
    discriminator = Discriminator(dataset.C).to(device)
    generator = Generator(args.c_dim, args.z_dim, dataset.C).to(device)
    encoder = Encoder(dataset.C, args.c_dim).to(device)
    projector = Projector(args.c_dim, args.c_dim*args.expansion).to(device)
    predictor = Predictor(args.c_dim*args.expansion).to(device)
    logger.debug(f'networks loaded.')
    
    logger.debug(f'initializing optimizers...')
    optim_d = optim.Adam(
        discriminator.parameters(),
        lr=2e-3,
        betas=(.5, .999)
    )
    optim_g = optim.Adam(
        [
            {'params': generator.parameters(), 'lr': 1e-3, 'betas': (.5, .999), 'weight_decay': 0.0},
            {'params': encoder.parameters(),   'lr': 2e-3, 'betas': (.5, .999), 'weight_decay': 1e-6},
            {'params': projector.parameters(), 'lr': 2e-3, 'betas': (.5, .999), 'weight_decay': 1e-6},
            {'params': predictor.parameters(), 'lr': 1e-2, 'betas': (.5, .999), 'weight_decay': 1e-6},
        ]
    )
    logger.debug(f'optimizers initialized.')
    
    one = torch.ones(args.batch_size, 1).to(device)
    zero = torch.zeros(args.batch_size, 1).to(device)
    
    g = args.initial_gap
    running_loss_d = 0.0
    running_loss_g = 0.0
    running_loss_e = 0.0
    running_loss_i = 0.0

    for epoch in range(args.epochs):
        discriminator.train()
        generator.train()
        encoder.train()
        projector.train()
        predictor.train()
        
        g -= args.gap_reduction
        if g < 0:
            g = 0

        for i, (x, y) in enumerate(dataloader):
            itr = epoch*len(dataloader) + i
            x = x.to(device)
            c, _, _ = create_factors(args.batch_size, args.c_dim)
            c = c.to(device)
            z = torch.randn(args.batch_size, args.z_dim).to(device)
            loss_d  = F.binary_cross_entropy_with_logits(discriminator(x), one)
            loss_d += F.binary_cross_entropy_with_logits(discriminator(generator(c, z).detach()), zero)
            running_loss_d += loss_d.item()
            
            optim_d.zero_grad() 
            loss_d.backward()
            optim_d.step()
            
            c1, c2, mask = create_factors(args.batch_size, args.c_dim, g)
            c1 = c1.to(device)
            c2 = c2.to(device)
            mask = mask.to(device)
            z1 = torch.randn(args.batch_size, args.z_dim).to(device)
            z2 = torch.randn(args.batch_size, args.z_dim).to(device)
            x1 = generator(c1, z1)
            x2 = generator(c2, z2)

            
            loss_g  = F.binary_cross_entropy_with_logits(discriminator(x1), one) / 2
            loss_g += F.binary_cross_entropy_with_logits(discriminator(x2), one) / 2
            running_loss_g += loss_g.item()
            
            e1 = encoder(x1)
            e2 = encoder(x2)
            j1 = projector(e1)
            j2 = projector(e2)
            p1 = predictor(j1)
            p2 = predictor(j2)
            
            loss_i  = F.mse_loss(e1, c1) / 2
            loss_i += F.mse_loss(e2, c2) / 2
            running_loss_i += loss_i
            
            mask2 = torch.repeat_interleave(mask, args.expansion, dim=1)
            loss_e  = -F.cosine_similarity(mask2*p1, mask2*j2.detach()).mean() / 2
            loss_e += -F.cosine_similarity(mask2*p2, mask2*j1.detach()).mean() / 2
            running_loss_e += loss_e.item()
            
            total_loss = loss_g + args.alpha * loss_e + args.beta * loss_i
            
            optim_g.zero_grad()
            total_loss.backward()
            optim_g.step()
            
            if itr % args.log_interval == 0:
                running_loss_d /= args.log_interval
                running_loss_g /= args.log_interval
                running_loss_e /= args.log_interval
                running_loss_i /= args.log_interval
                
                logger.info(f'[{epoch}/{i}] ({itr}): loss_d = {running_loss_d}, loss_g = {running_loss_g}, loss_e = {running_loss_e}, loss_i = {loss_i}')
                
                writer.add_scalar('Loss/D', running_loss_d, itr)
                writer.add_scalar('Loss/G', running_loss_g, itr)
                writer.add_scalar('Loss/E', running_loss_e, itr)
                writer.add_scalar('Loss/I', running_loss_i, itr)
                
                running_loss_d = 0.0
                running_loss_g = 0.0
                running_loss_e = 0.0
                running_loss_i = 0.0
                
        c1, c2, mask = create_factors(4, args.c_dim, g)
        c1 = c1.to(device)
        c2 = c2.to(device)
        z1 = torch.randn(4, args.c_dim).to(device)
        z2 = torch.randn(4, args.c_dim).to(device)

        generator.eval()
        with torch.no_grad():
            x1 = generator(c1, z1)
            x2 = generator(c2, z2)
        writer.add_images('x1', x1, itr)
        writer.add_images('x2', x1, itr)
                
        # evaluate
        res = compute_metrics(
            dataset.dataset, 
            make_representor(encoder, device),
            ['factor_vae_metric', 'dci'],
            
        )
        writer.add_scalar('Metrics/DCI', res['dci']['disentanglement'], itr)
        writer.add_scalar('Metrics/FactorVAE', res['factor_vae_metric']['eval_accuracy'], itr)
        
        # save checkpoint
        checkpoint = {
            'epoch': epoch,
            'iteration': i,
            'optim_d': optim_d.state_dict(),
            'optim_g': optim_g.state_dict(),
            'discriminator': discriminator.state_dict(),
            'generator':generator.state_dict(),
            'encoder': encoder.state_dict(),
            'projector': projector.state_dict(),
            'predictor': predictor.state_dict()
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
                          type=int, default=64, 
                          help='Batch size (default: 64).')
train_parser.add_argument('--epochs',  
                          type=int, default=20, 
                          help='Number of epochs (default: 20).')
train_parser.add_argument('--c-dim',  
                          type=int, default=5, 
                          help='Code dimmension (default: 5).')
train_parser.add_argument('--z-dim',  
                          type=int, default=5, 
                          help='Noise dimmension (default: 5).')
train_parser.add_argument('--expansion',  
                          type=int, default=5, 
                          help='Projection head expantion (default: 5).')
train_parser.add_argument('--initial-gap',  
                          type=float, default=2.0, 
                          help='Initial contrastive gap (default: 2.0).')
train_parser.add_argument('--gap-reduction',  
                          type=float, default=.1, 
                          help='Linear reduction of contrastive gap with each epoch (default: 0.1).')
train_parser.add_argument('--alpha',  
                          type=float, default=1.0, 
                          help='Alpha parameter (default: 0.1).')
train_parser.add_argument('--beta',  
                          type=float, default=1.0, 
                          help='Beta parameter (default: 0.1).')
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
    formatter = logging.Formatter('[%(asctime)s] %(levelname)s: %(message)s')
    fhandler.setFormatter(formatter)
    logger.addHandler(fhandler)
    logger.setLevel(logging.DEBUG)
    logger.debug(args)
    
    logger.debug(f'initializing SummaryWriter...')
    writer = SummaryWriter(
        log_dir=str(logs_dir/'tensorboard'), 
        filename_suffix=args.func.__name__)
    
    logger.debug(f'setting random seed {args.seed}...')
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    logger.debug(f'initializing device...')
    device = torch.device("cuda" if args.cuda else "cpu")
    logger.debug(f'device {device} initialized.')
    
    
    args.func()
    
    fhandler.close()
    writer.close()
