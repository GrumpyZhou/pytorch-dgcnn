import os
import time
import datetime
import numpy as np
import torch
from torch_geometric.data import DataLoader
from utils.common import make_deterministic, lprint
from utils.config import ConfigManager, get_optim_tag

import utils.datasets as Datasets
from utils.visdom import ClassificationTmp
from networks.dgcnn import DGCNNCls

def train_epoch(net, data_loader, get_label_fn):
    net.train()
    if net.lr_scheduler:
        net.lr_scheduler.step()

    for data in data_loader:
        data = data.to(net.device)                
        net.optimizer.zero_grad()
        loss = net.loss_(pts=data.pos, batch_ids=data.batch, lbls=get_label_fn(data))
        loss.backward()
        net.optimizer.step()
    return loss  

def test_epoch(net, data_loader, get_label_fn):
    net.eval()
    correct = 0
    for data in data_loader:
        data = data.to(net.device)
        with torch.no_grad():
            pred = net.pred_(pts=data.pos, batch_ids=data.batch)
        correct += pred.eq(get_label_fn(data)).sum().item()
    return correct / len(data_loader.dataset)


def test_epoch_detailed(net, data_loader, get_label_fn, categories):
    net.eval()
    num_classes = len(categories)
    cls_correct = [0 for i in range(num_classes)]
    cls_total = [0 for i in range(num_classes)]
    for data in data_loader:
        data = data.to(net.device)
        lbls = get_label_fn(data)
        with torch.no_grad():
            pred = net.pred_(pts=data.pos, batch_ids=data.batch)
        results =  pred.eq(lbls).squeeze()
        for i, res in enumerate(results):
            cls = lbls[i].item()
            cls_correct[cls] += res.item()
            cls_total[cls] += 1
    total_acc = 100.0 * np.sum(cls_correct) / np.sum(cls_total)
    per_cls_acc = [100.0 * cls_correct[i] / cls_total[i] for i, name in enumerate(categories)]
    averaged_acc = np.mean(per_cls_acc)
    print('Total: {:.2f}% Per class avg: {:.2f}\n Per class: {}'.format(total_acc, averaged_acc, 
                                ['{} {:.1f}% '.format(v[0], v[1]) for v in zip(categories, per_cls_acc)]))
    
    return total_acc

def main(config):    
    # Env setup
    device = torch.device('cuda:{}'.format(config.gpu) if torch.cuda.is_available() else 'cpu')
    map_location = lambda storage, loc: storage.cuda(device.index) if torch.cuda.is_available() else storage
    print('Use device:{}.'.format(device))
    make_deterministic(config.seed)

    # Logging
    optim_tag = get_optim_tag(config) 
    out_dir = os.path.join(config.odir, config.network, config.dataset, optim_tag)
    #                       '{}_{}'.format(optim_tag, datetime.datetime.now().strftime("%Y%m%d%H")))
    print('Output folder {}'.format(out_dir))
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    log = open(os.path.join(out_dir, 'log.txt'), 'a')
    lprint(str(config), log)
    ckpt_dir = os.path.join(out_dir, 'ckpt')
    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)

    # Initialize dataset
    dataset_handler = Datasets.__dict__[config.dataset](config.data_dir, classification=True)
    get_label_fn = dataset_handler.label_parser #get_labels_parser(config.dataset, classification=True)
    categories = dataset_handler.categories
    num_classes = dataset_handler.num_classes
    if config.training:
        # Initialize Visdom
        visdom = ClassificationTmp(legend_tag=optim_tag, viswin=config.viswin, 
                                   visenv=config.visenv, vishost=config.vishost, visport=config.visport)
        loss_meter, acc_meter = visdom.get_meters()
        
        # Data loading
        #train_set, val_set = get_dataset(config.data_dir, config.dataset, training=True)
        train_set, val_set = dataset_handler.get_train_split(), dataset_handler.get_val_split()
        train_loader = DataLoader(train_set, batch_size=config.batch, shuffle=True, num_workers=config.worker)
        val_loader = DataLoader(val_set, batch_size=config.batch, shuffle=False)

        # Initialize network
        net = DGCNNCls(num_classes=num_classes, K=config.K, device=device)
        # net.init_weights_() # TODO
        net.set_optimizer_(config)

        # Load model checkpoint
        last_epoch = 0
        if config.ckpt is not None and os.path.isfile(config.ckpt):
            ckpt = torch.load(config.ckpt, map_location=map_location) 
            last_epoch = ckpt['last_epoch']
            net.resume_(ckpt['state_dict'], ckpt['optimizer'], ckpt['lr_scheduler'], training=True)

        #Setup visualizer
        start_time = time.time()
        max_epoch =  config.epoch
        print('Start training from {} to {}.'.format(last_epoch+1, max_epoch))
        for epoch in range(last_epoch+1, max_epoch+1):
            loss = train_epoch(net, train_loader, get_label_fn)
            loss_meter.update(X=epoch, Y=loss) 
            current_ckpt ={'last_epoch': epoch,
                           'network': config.network,
                           'state_dict': net.state_dict(),
                           'optimizer' : net.optimizer.state_dict(),
                           'lr_scheduler' : net.lr_scheduler.state_dict(),
                           'loss' : loss
                          }
            torch.save(current_ckpt, os.path.join(ckpt_dir, 'ckpt_last.pth'))
            lprint('Epoch {}, loss:{:.4f}'.format(epoch, loss), log)

            # Validation
            val_acc = -1
            if config.val and epoch % config.val == 0:
                val_acc = test_epoch(net, val_loader, get_label_fn)
                #val_acc = test_epoch_detailed(net, val_loader, get_label_fn, categories) 
                acc_meter.update(X=epoch, Y=val_acc)
                ckpt_name = 'ckpt_{}_{:.3f}.pth'.format(epoch, val_acc)
                torch.save(current_ckpt, os.path.join(ckpt_dir, ckpt_name))
                lprint('Save {} val acc:{:.4f}'.format(ckpt_name, val_acc), log)
            visdom.save_state()
        lprint('Total training time {:.4f}s\n\n'.format((time.time() - start_time)), log)
    else:
        lprint('Testing {} with ckpt {}'.format(config.network, config.ckpt), log)
         # Data loading
        #test_set = get_dataset(config.data_dir, config.dataset, training=False)
        test_set = dataset_handler.get_test_split()        
        test_loader = DataLoader(test_set, batch_size=config.batch, shuffle=False)
        
         # Initialize network
        net = DGCNNCls(num_classes=num_classes, K=config.K, device=device)
        if config.ckpt is not None and os.path.isfile(config.ckpt):
            ckpt = torch.load(config.ckpt, map_location=map_location) 
            net.resume_(ckpt['state_dict'], ckpt['optimizer'], ckpt['lr_scheduler'], training=False)
        start_time = time.time()   
        test_acc = test_epoch_detailed(net, test_loader, get_label_fn, categories)
        lprint('Accuracy: {:.4f} time {:.4f}s\n\n'.format(test_acc, time.time() - start_time), log)  
    log.close()
    
if __name__ == '__main__':
    cm = ConfigManager()
    config = cm.parse()
    main(config)