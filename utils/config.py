import argparse
from collections import namedtuple


# OptimConfig = namedtuple('OptimConfig', 
#                          ['optim', 'lr', 'lrd_factor', 'lrd_step',
#                           'weight_decay', 'momentum', 'start_epoch', 
#                           'optimizer_dict', 'training'])
def get_optim_tag(config):
    optim_tag = '{}_lr{}_wd{}'.format(config.optim, config.lr, config.weight_decay)
    if config.lrd_step > 0 and config.lrd_factor < 1:
        optim_tag = '{}_lrd{}-{}'.format(optim_tag, config.lrd_factor, config.lrd_step)
    return optim_tag

class ConfigManager:
    def __init__(self):
        self.ConfigTemplate = namedtuple('Config', 
                                         ['training', 'seed', 'gpu', 'data_dir', 'dataset', 'odir', 
                                          'network', 'K', 'ckpt', 'val', 'batch', 'worker', 'epoch',
                                          'optim', 'momentum', 'weight_decay',
                                           'lr', 'lrd_factor', 'lrd_step',
                                          'visenv', 'viswin', 'visport', 'vishost'])
    def parse(self):
        parser = argparse.ArgumentParser(description='Point Cloud Training Argument Parser')
        
        # Training setup
        parser.add_argument('--test', action='store_false', dest='training', help='network testing')
        parser.add_argument('--train', action='store_true', dest='training', help='network training')
        parser.add_argument('--seed', metavar='%d', type=int, default=7, 
                                help='program random seed(default: %(default)s)')
        parser.add_argument('--gpu', metavar='%d',type=int,  default='0', 
                                help='gpu id(cpu is used if no gpu available, default: %(default)s)') 
        parser.add_argument('--data_dir', metavar='%s', type=str, default='data/',
                                help='data root directory(default: %(default)s)' )
        parser.add_argument('--dataset', type=str, choices=['ModelNet10', 'ModelNet40', 'ShapeNet'], 
                                default='ModelNet40',
                                help='dataset name(default: %(default)s)')
        parser.add_argument('--odir', metavar='%s', type=str, default='output/', 
                                help='program outputs dir(default: %(default)s)')
        
        parser.add_argument('--network', type=str, choices=['DGCNNCls', 'DGCNNSeg'], default='DGCNNCls',
                                help='network for training(default: %(default)s)')
        parser.add_argument('--K', type=int, default=20,
                                help='number of nearest neighbour for edge conv(default: %(default)s)')

        parser.add_argument('--ckpt', metavar='%s',  type=str, default=None, 
                                help='model path to resume trainnig(default: %(default)s)')
        parser.add_argument('--val', metavar='%d', type=int, default=5, 
                                help='epoch step for validation(default: %(default)s)')
        parser.add_argument('--batch', metavar='%d', type=int, default=35, 
                                help='batch size (default: %(default)s)')
        parser.add_argument('--worker', metavar='%d', type=int, default=2, 
                                help='number of threads for data loading(default: %(default)s)')  
        parser.add_argument('--epoch', metavar='%d', type=int, default=250, 
                                help='number of epochs for training(default: %(default)s)')
        
        # Optimization setup
        parser.add_argument('--optim', type=str, default='Adam', choices=['Adam', 'SGD'],  
                                help='optimizer type (default: %(default)s)')
        parser.add_argument('--momentum', metavar='%f', type=float, default=0.9, 
                                help='momentum factor for SGD(default: %(default)s)')      
        parser.add_argument('--lr', metavar='%f', type=float, default=1e-3,
                                help='initial learning rate(default: %(default)s)') 
        parser.add_argument('--lrd_factor', metavar='%f', type=float, default=0.8,
                                help='decay factor for learning rate decay(default: %(default)s)')
        parser.add_argument('--lrd_step', metavar='%d', type=int, default=50,
                                help='decay step for learning rate decay(default: %(default)s)')
        parser.add_argument('--weight_decay', metavar='%f', type=float, default=1e-3, 
                                help='weight decay rate(default: %(default)s)')        

        # Visdom setup
        parser.add_argument('--visenv', metavar='%s', type=str, default=None, 
                                help='visdom environment name, if none visdom will not be used(default: %(default)s)')
        parser.add_argument('--viswin', metavar='%s', type=str, default=None, 
                                help='title prefix for window name of visdom data plots(default: %(default)s)')
        parser.add_argument('--visport', metavar='%d', type=int, default=9333, 
                                help='the port where the visdom server is running(default: %(default)s)')
        parser.add_argument('--vishost', metavar='%s', type=str, default='localhost', 
                            help='the hostname where the visdom server is running(default: %(default)s)')
        
        config = parser.parse_args()
        return config
    
    def get_manual_config(self, training=True, seed=7, gpu=0, data_dir='../data',
                          dataset='ModelNet40', odir='output/', val=5,
                          network='DGCNNCls', K=20, ckpt=None, batch=32, worker=2, epoch=250,
                          optim='Adam', momentum=0.9,  weight_decay=1e-3,
                          lr=1e-3, lrd_factor=0.8, lrd_step=50, 
                          visenv=None, viswin=None, visport=9333, vishost='localhost'):
        # Mainly for debug or implementation phase
        config = self.ConfigTemplate(training=training, seed=seed, gpu=gpu, data_dir=data_dir,
                                    dataset=dataset, odir=odir, val=val, network=network, K=K,
                                    ckpt=ckpt, batch=batch, worker=worker, epoch=epoch,
                                    optim=optim, momentum=momentum, weight_decay=weight_decay,
                                    lr=lr, lrd_factor=lrd_factor, lrd_step=lrd_step,
                                    visenv=visenv, viswin=viswin, visport=visport, vishost=vishost)
        
        return config

if __name__ == '__main__':
    conf = ConfigManager.parse()