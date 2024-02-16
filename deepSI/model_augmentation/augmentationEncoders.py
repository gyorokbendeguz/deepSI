import deepSI
import numpy as np
import torch
from torch import nn
from deepSI import model_augmentation

###################################################################################
####################             ENCODER FUNCTIONS             ####################
###################################################################################

class default_encoder_net(nn.Module):  # a simple FC net with a residual (default approach)
    def __init__(self, nb, nu, na, ny, nx, n_nodes_per_layer=64, n_hidden_layers=2, activation=nn.Tanh):
        super(default_encoder_net, self).__init__()
        from deepSI.utils import simple_res_net
        self.nu = tuple() if nu is None else ((nu,) if isinstance(nu,int) else nu)
        self.ny = tuple() if ny is None else ((ny,) if isinstance(ny,int) else ny)
        self.net = simple_res_net(n_in=nb*np.prod(self.nu,dtype=int) + na*np.prod(self.ny,dtype=int), n_out=nx,
                                  n_nodes_per_layer=n_nodes_per_layer,  n_hidden_layers=n_hidden_layers,
                                  activation=activation)

    def forward(self, upast, ypast):
        net_in = torch.cat([upast.view(upast.shape[0],-1),ypast.view(ypast.shape[0],-1)],axis=1)
        return self.net(net_in)
  
  
class state_measure_encoder:  # for known y[k]=x[k] cases
    def __init__(self, nb, nu, na, ny, nx, **kwargs):
        self.nu = tuple() if nu is None else ((nu,) if isinstance(nu, int) else nu)
        self.ny = tuple() if ny is None else ((ny,) if isinstance(ny, int) else ny)
        
    def __call__(self, upast, ypast):
        # in:                               | out:
        #  - u_past (Nd, nb+nb_right, Nu)   |  - x0 (Nd, Nx=Ny)
        #  - y_past (Nd, na+na_right, Ny)   |
        return ypast[:,-1,:]

class dynamic_state_meas_encoder(nn.Module):  # y[k]=x[k] cases with hidden states
    def __init__(self, nb, nu, na, ny, nx, nx_h, n_nodes_per_layer=64, n_hidden_layers=2, activation=nn.Tanh):
        super(dynamic_state_meas_encoder, self).__init__()
        from deepSI.utils import simple_res_net
        self.nu = tuple() if nu is None else ((nu,) if isinstance(nu,int) else nu)
        self.ny = tuple() if ny is None else ((ny,) if isinstance(ny,int) else ny)
        self.net = simple_res_net(n_in=nb*np.prod(self.nu,dtype=int) + na*np.prod(self.ny,dtype=int), n_out=nx_h,
                                  n_nodes_per_layer=n_nodes_per_layer,  n_hidden_layers=n_hidden_layers,
                                  activation=activation)

    def forward(self, upast, ypast):
        # in:                               | out:
        #  - u_past (Nd, nb+nb_right, Nu)   |  - x0 (Nd, Nx_meas+Nx_hidden)
        #  - y_past (Nd, na+na_right, Ny)   |
        x_meas = ypast[:,-1,:]
        # ypast_mod = ypast[:,:-1,:]  #ToDo: maybe it is needed or not
        net_in = torch.cat([upast.view(upast.shape[0],-1),ypast.view(ypast.shape[0],-1)],axis=1)
        x_hidden = self.net(net_in)
        return torch.hstack((x_meas, x_hidden))

