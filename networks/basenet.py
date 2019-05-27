import torch
import torch.nn as nn
from collections import OrderedDict

class BaseNet(nn.Module):
    def __init__(self, ):
        super().__init__()

    @property
    def num_params(self):
        return sum([p.numel() for p in self.parameters()])
      
    def forward(self, *inputs):
        """Defines the computation performed at every call.
        Inherited from Superclass torch.nn.Module.
        Should be overridden by all subclasses.
        """
        raise NotImplementedError
            
    def init_weights_(self):
        """Define how to initialize the weights of the network.
        Should be overridden by all subclasses, since it 
        normally differs according to the network models.
        """
        raise NotImplementedError
    
    def loss_(self, *inputs):        
        """Define how to calculate the loss  for the network.
        Should be overridden by all subclasses, since different 
        applications or network models may have different types 
        of targets and the corresponding criterions to evaluate 
        the predictions.
        """
        raise NotImplementedError
     
 
    def set_optimizer_(self, config):
        if config.optim == 'Adam':
            self.optimizer = torch.optim.Adam(self.parameters(), 
                                              lr=config.lr, 
                                              weight_decay=config.weight_decay)
        elif config.name == 'SGD':
            self.optimizer = torch.optim.SGD(self.parameters(), 
                                             lr=config.lr, 
                                             weight_decay=config.weight_decay, 
                                             momentum=config.momentum, 
                                             nesterov=False)
        
        # Setup lr scheduler if defined
        if config.lrd_factor < 1 and config.lrd_step > 0:
            self.lr_scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, 
                                                                step_size=config.lrd_step, 
                                                                gamma=config.lrd_factor)
        else:
            self.lr_scheduler = None
            
    def resume_(self, state_dict, optimizer_dict=None, lrs_dict=None, training=True): 
        # Load model state
        if state_dict:
            if len(state_dict.items()) == len(self.state_dict()):
                print('Load all model parameters from state dict')
                self.load_state_dict(state_dict)
            else:
                print('Load part of model parameters from state dict')
                self.load_state_dict(state_dict, strict=False) 

        if training:
            # Load optimizer state
            if optimizer_dict:
                self.optimizer.load_state_dict(optimizer_dict)

            # Load lr scheduler state
            if self.lr_scheduler and lrs_dict:
                self.lr_scheduler.load_state_dict(lrs_dict)
    
    def save_weights_(self, sav_path):
        torch.save(self.state_dict(), sav_path)
    
    def print_(self):
        for k, v in self.state_dict().items():
            print(k, v.size())

    def xavier_init_func_(self, m):
        classname = m.__class__.__name__
        if classname == 'Conv1d':
            nn.init.xavier_uniform_(m.weight.data)
            nn.init.constant_(m.bias.data, 0.0)
        elif classname == 'Linear':
            nn.init.xavier_uniform_(m.weight.data)
            nn.init.constant_(m.bias.data, 0.0)
        elif classname == 'BatchNorm2d':
            nn.init.constant_(m.weight.data, 1.0)
            nn.init.constant_(m.bias.data, 0.0)
            