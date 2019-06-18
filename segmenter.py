import os
import time
import datetime
import numpy as np
import torch
from torch_geometric.data import DataLoader
from utils.common import make_deterministic, lprint
from utils.config import ConfigManager, get_optim_tag

import utils.datasets as Datasets
from utils.visdom import SegementationTmp
from networks.dgcnn import DGCNNSeg
from torch_geometric.utils import one_hot
from torch_scatter import scatter_add

def mean_iou(pred, target, num_classes, batch=None):
    r"""Computes the mean Intersection over Union score.
    Args:
        pred (LongTensor): The predictions.
        target (LongTensor): The targets.
        num_classes (int): The number of classes.
        batch (LongTensor): The assignment vector which maps each pred-target
            pair to an example.
    :rtype: :class:`Tensor`
    """
    pred = one_hot(pred, num_classes, dtype=torch.long)
    target = one_hot(target, num_classes, dtype=torch.long)

    if batch is not None:
        i = scatter_add(pred & target, batch, dim=0).to(torch.float)
        u = scatter_add(pred | target, batch, dim=0).to(torch.float)
    else:
        i = (pred & target).sum(dim=0).to(torch.float)
        u = (pred | target).sum(dim=0).to(torch.float)

    iou = i / u
    iou[torch.isnan(iou)] = 1
    iou = iou.mean(dim=-1)
    return iou

def train_epoch(net, data_loader, get_label_fn):
    net.train()
    if net.lr_scheduler:
        net.lr_scheduler.step()
    
    epoch_loss = 0
    correct = total = 0
    epoch_iou = []
    num_batch = len(data_loader)
    for data in data_loader:
        data = data.to(net.device)
        lbls = get_label_fn(data)
        net.optimizer.zero_grad()
        loss, preds = net.loss_(pts=data.pos, batch_ids=data.batch, lbls=lbls)
        loss.backward()
        net.optimizer.step()
        epoch_loss += loss
        correct += preds.eq(lbls).sum().item()
        total += lbls.numel()
        epoch_iou.append(mean_iou(lbls, preds, net.num_classes, batch=data.batch))
    epoch_loss = epoch_loss / num_batch
    epoch_acc = correct / total
    epoch_iou = torch.cat(epoch_iou, dim=0).mean().item()
    return epoch_loss, epoch_acc, epoch_iou

def test_epoch(net, data_loader, get_label_fn):
    net.eval()
    correct = total = 0
    epoch_iou = []
    for data in data_loader:
        data = data.to(net.device)
        lbls = get_label_fn(data)
        with torch.no_grad():
            preds = net.pred_(pts=data.pos, batch_ids=data.batch)   
        correct += preds.eq(lbls).sum().item()
        total += lbls.numel()
        epoch_iou.append(mean_iou(lbls, preds, net.num_classes, batch=data.batch))
    epoch_acc = correct / total
    epoch_iou = torch.cat(epoch_iou, dim=0).mean().item()
    return epoch_acc, epoch_iou

def main(config):    
    # Env setup
    device = torch.device('cuda:{}'.format(config.gpu) if torch.cuda.is_available() else 'cpu')
    map_location = lambda storage, loc: storage.cuda(device.index) if torch.cuda.is_available() else storage
    print('Use device:{}.'.format(device))
    make_deterministic(config.seed)

    # Logging
    optim_tag = get_optim_tag(config) 
    out_dir = os.path.join(config.odir, config.network, config.dataset, 'K{}_{}'.format(config.K, optim_tag), config.cat)
    print('Output folder {}'.format(out_dir))
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    log = open(os.path.join(out_dir, 'log.txt'), 'a')
    lprint(str(config), log)
    ckpt_dir = os.path.join(out_dir, 'ckpt')
    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)

    # Initialize dataset
    category = config.cat
    if category == 'All': 
        dataset_handler = Datasets.__dict__['ShapeNet']('data')
    else:
        dataset_handler = Datasets.__dict__['ShapeNetCategory']('data', category)
    get_label_fn = dataset_handler.label_parser
    #categories = dataset_handler.categories
    if config.training:
        # Initialize Visdom
        visdom = SegementationTmp(legend_tag=optim_tag, viswin=config.viswin, 
                                   visenv=config.visenv, vishost=config.vishost, visport=config.visport)
        loss_meter, acc_meter, iou_meter = visdom.get_meters()
        
        # Data loading
        train_set, val_set = dataset_handler.get_train_split(), dataset_handler.get_val_split()
        train_loader = DataLoader(train_set, batch_size=config.batch, shuffle=True, num_workers=config.worker)
        val_loader = DataLoader(val_set, batch_size=config.batch, shuffle=False)

        # Initialize network
        net = DGCNNSeg(num_classes=train_set.num_classes, K=config.K, device=device)
        net.set_optimizer_(config)

        # Load model checkpoint
        last_epoch = 0
        if config.ckpt is not None and os.path.isfile(config.ckpt):
            ckpt = torch.load(config.ckpt, map_location=map_location) 
            last_epoch = ckpt['last_epoch']
            net.resume_(ckpt['state_dict'], ckpt['optimizer'], ckpt['lr_scheduler'], training=True)

        # Training
        start_time = time.time()
        max_epoch =  config.epoch
        print('Start training from {} to {}.'.format(last_epoch+1, max_epoch))
        for epoch in range(last_epoch+1, max_epoch+1):
            loss, acc, iou = train_epoch(net, train_loader, get_label_fn)
            loss_meter.update(X=epoch, Y=loss) 
            current_ckpt ={'last_epoch': epoch,
                           'network': config.network,
                           'state_dict': net.state_dict(),
                           'optimizer' : net.optimizer.state_dict(),
                           'lr_scheduler' : net.lr_scheduler.state_dict(),
                           'loss' : loss
                          }
            torch.save(current_ckpt, os.path.join(ckpt_dir, 'ckpt_last.pth'))
            lprint('Epoch {}, train loss:{:.4f} acc: {:.4f} iou: {:.4f}'.format(epoch, loss, acc, iou), log)

            # Validation
            val_acc = -1
            if config.val and epoch % config.val == 0:
                val_acc, val_iou = test_epoch(net, val_loader, get_label_fn)
                acc_meter.update(X=epoch, Y=val_acc)
                iou_meter.update(X=epoch, Y=val_iou)
                ckpt_name = 'ckpt_{}_{:.3f}.pth'.format(epoch, val_acc)
                torch.save(current_ckpt, os.path.join(ckpt_dir, ckpt_name))
                lprint('Save ckpt_{}.pth Val acc: {:.4f} iou: {:.4f}'.format(epoch, val_acc, val_iou), log)
            visdom.save_state()
        lprint('Total training time {:.4f}s\n'.format((time.time() - start_time)), log)
        
        # Final testing
        test_set = dataset_handler.get_test_split()        
        test_loader = DataLoader(test_set, batch_size=config.batch, shuffle=False)
        test_acc, test_iou = test_epoch(net, test_loader, get_label_fn)
        lprint('Testing last ckpt Accuracy: {:.4f} IoU: {:.4f}\n\n'.format(test_acc, test_iou), log) 
    else:
        lprint('Testing {} with ckpt {}'.format(config.network, config.ckpt), log)
        
         # Data loading
        test_set = dataset_handler.get_test_split()        
        test_loader = DataLoader(test_set, batch_size=config.batch, shuffle=False)
        
         # Initialize network
        net = DGCNNSeg(num_classes=test_set.num_classes, K=config.K, device=device)
        if config.ckpt is not None and os.path.isfile(config.ckpt):
            ckpt = torch.load(config.ckpt, map_location=map_location) 
            net.resume_(ckpt['state_dict'], ckpt['optimizer'], ckpt['lr_scheduler'], training=False)
        start_time = time.time() 
        test_acc, test_iou = test_epoch(net, test_loader, get_label_fn)
        lprint('Accuracy: {:.4f} IoU: {:.4f} time {:.4f}s\n\n'.format(test_acc, test_iou, time.time() - start_time), log)  
    log.close()
    
if __name__ == '__main__':
    cm = ConfigManager()
    config = cm.parse()
    main(config)