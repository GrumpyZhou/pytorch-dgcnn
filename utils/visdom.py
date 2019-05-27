import torch
import visdom
import numpy as np

            
class ClassificationTmp:
    def __init__(self, legend_tag, viswin, visenv, vishost, visport):
        self.visman = VisManager(visenv, host=vishost, port=visport)
        
        # Intialize windows with name
        loss_win_tag = '{} Loss'.format(viswin) if viswin else 'Loss'
        acc_win_tag = '{} Val'.format(viswin) if viswin else 'Val'
        win_dict = {loss_win_tag : [legend_tag], acc_win_tag : [legend_tag]}
        self.visman.set_wins(win_dict)
        
        self.loss_meter = self.visman.get_meter(loss_win_tag, legend_tag)
        self.acc_meter = self.visman.get_meter(acc_win_tag, legend_tag)
            
    def get_meters(self):       
        return self.loss_meter, self.acc_meter
    
    def save_state(self):
        self.visman.save_state()
          
class VisLineMeter:
    '''Visdom Line Data Meter'''
    def __init__(self, server, env, win, legend):
        self.server = server
        self.env = env
        self.win = win
        self.legend = legend
        self.style_opts = self.get_style_opts_()

    def get_style_opts_(self):
        layout = {'plotly': dict(title=self.win, xaxis={'title': 'epochs'})}
        style_opts=dict(mode='lines', showlegend=True, layoutopts=layout)
        #style_opts=dict(mode='marker+lines', 
        #      markersize=5,
        #      markersymbol='dot',
        #      markers={'line': {'width': 0.5}},
        #      showlegend=True, layoutopts=layout)
        return style_opts
    
    def validate_input_(self, X):
        if isinstance(X, np.ndarray):
            return X
        elif isinstance(X, int) or isinstance(X, float) or isinstance(X, np.float32):
            return np.array([X])
        elif isinstance(X, torch.Tensor):
            X = X.cpu().data.numpy()
            if X.ndim == 0:
                X = X.reshape((1))
            return X
        
    def update(self, X, Y):
        if self.server:
            #if self.server.get_window_data(win=self.win, env=self.env) == '':
            self.server.line(X=self.validate_input_(X), 
                             Y=self.validate_input_(Y), 
                             env=self.env, win=self.win, name=self.legend,
                             opts=self.style_opts, update='append')
        
    def clear(self):
        self.server.line(X=None, Y=None, env=self.env, win=self.win, name=self.legend, update='remove')
        
    def __repr__(self):
        return 'Visdom meter(env={}, win={}, legend={})'.format(self.env, self.win, self.legend)
    
class VisManager:   
    """Visdom manager
    Initialize connection to the running visdom server.
    Create windows with style to plot data.
    Maintain window creation(incl. window style, data meters), 
    window state saving and clear.
    
    """
    def __init__(self, env, host='localhost', port='8097'):
        if env is None:
            self.server = None
            print('Visdom is not set..')
        else:
            self.dummy = False
            host = 'http://{}'.format(host)
            self.server = visdom.Visdom(server=host, port=port)
            self.env = env
            assert self.server.check_connection(), 'Visdom server is not active on server {}:{}'.format(host, port)  
            print('Visdom server connected on {}:{}'.format(host, port))
        self.win_pool = {}
        
    def set_wins(self, win_dict):
        '''win_dict: {win_name : [legend_name]}'''
        for win_name in win_dict:
            self.win_pool[win_name] = {}
            for legend_name in win_dict[win_name]:
                meter = VisLineMeter(self.server, self.env, win_name, legend_name)
                self.win_pool[win_name][legend_name] = meter
                print('Initialize data meters {}'.format(str(meter)))

    def get_meter(self, win_name, legend_name):
        return self.win_pool[win_name][legend_name] 

    def save_state(self):
        if self.server:
            self.server.save(envs=[self.env])
        
    def clear_all(self):
        for win_name in self.win_pool:
            for legend_name in self.win_pool[win_name]:
                self.win_pool[win_name][legend_name].clear()
                    
    def print_(self):
        print('Visdom Manager Window Pool:\n')
        for win_name in self.win_pool:
            for legend_name in self.win_pool[win_name]:
                print(self.win_pool[win_name][legend_name])
