import deepSI
import numpy as np
import torch
from torch import nn
from deepSI import model_augmentation
from deepSI.model_augmentation.utils import verifySystemType, verifyNetType, RK4_step, assign_param
from deepSI.model_augmentation.augmentationEncoders import default_encoder_net, state_measure_encoder

###################################################################################
####################         DEFAULT/GENERIC FUNCTIONS         ####################
###################################################################################

class default_state_net(nn.Module):
    def __init__(self, nu, nx, augmentation_params):
        super(default_state_net, self).__init__()
        self.MApar = augmentation_params

    def forward(self, x, u):
        # in:                | out:
        #  - x (Nd, Nx)      |  - x+ (Nd, Nx)
        #  - u (Nd, Nu + np) |
        return self.MApar.f_h(x,u)[0] # Select f(x,u) function

class default_output_net(nn.Module):
    def __init__(self, nu, nx, ny, augmentation_params):
        super(default_output_net, self).__init__()
        self.MApar = augmentation_params

    def forward(self, x, u):
        # in:                | out:
        #  - x (Nd, Nx)      |  - y (Nd, Ny)
        #  - u (Nd, Nu + np) |
        return self.MApar.f_h(x,u)[1] # Select h(x,u) function

def get_dynamic_augment_fitsys(augmentation_structure, known_system, hidden_state, neur_net, aug_kwargs={}, e_net=default_encoder_net,
                               y_lag_encoder=None, u_lag_encoder=None, enet_kwargs={}, na_right=0, nb_right=0,
                               regLambda=0.01, orthLambda=0):
    nx_encoder = known_system.Nx + hidden_state
    if e_net is None:
        return deepSI.fit_systems.SS_encoder_general_hf(feedthrough=True, nx=nx_encoder, na=y_lag_encoder, nb=u_lag_encoder,
                                                        e_net=dynamic_state_meas_encoder, hf_net=augmentation_structure,
                                                        e_net_kwargs=dict(nx_h=hidden_state, **enet_kwargs),
                                                        hf_net_kwargs=dict(known_system=known_system, wnet=neur_net, regLambda=regLambda,
                                                                           orthLambda=orthLambda, nx_h=hidden_state, **aug_kwargs),
                                                        na_right=na_right, nb_right=nb_right)
    else:
        raise NotImplementedError('Not implemented yet...')

def get_augmented_fitsys(augmentation_structure, known_system, wnet, aug_kwargs={}, e_net=default_encoder_net,
                         y_lag_encoder=None, u_lag_encoder=None, enet_kwargs={}, na_right=0, nb_right=0,
                         regLambda=0.01, orthLambda=0):
    if augmentation_structure is model_augmentation.augmentationstructures.SSE_DynamicAugmentation:
        # Learn the state of the augmented model as well
        nx_system = known_system.Nx
        nx_hidden = wnet.n_state
        nx_encoder = nx_system + nx_hidden
    else:
        nx_encoder = known_system.Nx
    #ToDo: augmentation structure verification
    '''
    elif augmentation_structure is model_augmentation.augmentationstructures.SSE_StaticAugmentation or \
        augmentation_structure is model_augmentation.augmentationstructures.SSE_AdditiveAugmentation:
        nx_encoder = known_system.Nx
    else: raise ValueError("'augmentation_structure' should be either " +
                           "'SSE_DynamicAugmentation', 'SSE_StaticAugmentation' or 'SSE_AdditiveAugmentation'")
   '''
    if y_lag_encoder is None: y_lag_encoder = nx_encoder + 1
    if u_lag_encoder is None: u_lag_encoder = nx_encoder + 1
    if e_net is None:
        y_lag_encoder = 1
        u_lag_encoder = 1
        na_right = 1
        e_net = state_measure_encoder

    return deepSI.fit_systems.SS_encoder_general_hf(feedthrough=True, nx=nx_encoder, na=y_lag_encoder, nb=u_lag_encoder,
                                                    e_net=e_net, e_net_kwargs=dict(**enet_kwargs), hf_net=augmentation_structure,
                                                    hf_net_kwargs=dict(known_system=known_system, wnet=wnet, regLambda=regLambda,
                                                                   orthLambda=orthLambda, **aug_kwargs), na_right=na_right, nb_right=nb_right
                                                    )


def get_CT_augmented_fitsys(augmentation_params, y_lag_encoder, u_lag_encoder, e_net=None, f_net=None, h_net=None):
    raise NotImplementedError('Not verified or tested or whatevered yet...')
    # Input checks:
    if type(augmentation_params) is model_augmentation.augmentationstructures.SSE_DynamicAugmentation:
        # Learn the state of the augmented model as well
        nx_system = augmentation_params.Nx
        nx_hidden = augmentation_params.Nxh
        nx_encoder = nx_system + nx_hidden
    elif type(augmentation_params) is model_augmentation.augmentationstructures.SSE_StaticAugmentation:
        nx_encoder = augmentation_params.Nx
    else: raise ValueError("'augmentation_params' should be of type " +
                           "'SSE_DynamicAugmentation' or 'SSE_StaticAugmentation'")
    if augmentation_params.sys.Ts is None: raise ValueError("Sampling time associated with the system not defined...")  # ToDo: IS this the correct way of having a samping time here?
    if e_net is None: e_net = default_encoder_net
    if f_net is None: f_net = default_CT_state_net
    else: print("Make sure that your custom net is of the form as given in 'augmentationstructures.default_CT_state_net'")
    if h_net is None: h_net = default_output_net
    else: print("Make sure that your custom net is of the form as given in 'augmentationstructures.default_output_net'")
    ### TODO: For now have my own RK4 step integrator, Later... have it integrated with deepSI (also include normalization
    ### TODO: factor and stuff)
    # calculate the normalization factor for f  TODO: For now set it to zero...
    # integratornet = integrator_RK4

    return deepSI.fit_systems.SS_encoder_general(feedthrough=True, nx=nx_encoder,
            na=y_lag_encoder, nb=u_lag_encoder, e_net=e_net, e_net_kwargs=dict(),
            f_net=f_net, f_net_kwargs=dict(augmentation_params=augmentation_params),
            h_net=h_net, h_net_kwargs=dict(augmentation_params=augmentation_params))


###################################################################################
##############         SUBSPACE ENCODER BASED AUGMENTATIONS          ##############
##############                  STATIC AUGMENTATION                  ##############
###################################################################################

class SSE_Augmentation(nn.Module):  # TODO: Make generic class
    def __init__(self):
        super(SSE_Augmentation, self).__init__()
        pass

class SSE_StaticAugmentation(nn.Module):
    def __init__(self, nx, nu, ny, known_system, wnet, initial_scaling_factor=1e-3, Dzw_is_zero=True, feedthrough=True):
        super(SSE_StaticAugmentation, self).__init__()
        # First verify if we have the correct system type and augmentationparameters
        verifySystemType(known_system)
        verifyNetType(wnet, 'static')
        # Save parameters
        self.sys = known_system
        self.net = wnet
        self.Nu = self.sys.Nu
        self.Nx = self.sys.Nx
        self.Ny = self.sys.Ny
        self.Nz = self.net.n_in
        self.Nw = self.net.n_out
        self.Bw = nn.Parameter(data=initial_scaling_factor * torch.rand(self.Nx, self.Nw))
        self.Cz = nn.Parameter(data=initial_scaling_factor * torch.rand(self.Nz, self.Nx))
        self.Dzw = None if Dzw_is_zero else nn.Parameter(data=initial_scaling_factor * torch.rand(self.Nz, self.Nw))
        self.Dzu = nn.Parameter(data=initial_scaling_factor * torch.rand(self.Nz, self.Nu))
        self.Dyw = nn.Parameter(data=initial_scaling_factor * torch.rand(self.Ny, self.Nw))

    def initialize_parameters(self, Bw = None, Cz = None, Dzw = None, Dzu = None, Dyw = None):
        self.Bw.data = assign_param(self.Bw, Bw, 'Bw')
        self.Cz.data = assign_param(self.Cz, Cz, 'Cz')
        self.Dzw.data = assign_param(self.Dzw, Dzw, 'Dzw')
        self.Dzu.data = assign_param(self.Dzu, Dzu, 'Dzu')
        self.Dyw.data = assign_param(self.Dyw, Dyw, 'Dyw')

    def compute_z(self, x, u):
        # in:           | out:
        #  - x (Nd, Nx) |  - z (Nd, Nz)
        #  - u (Nd, Nu) |
        zx = torch.einsum('ij, bj -> bi', self.Cz, x)   # (Nz, Nx)*(Nd, Nx)->(Nd, Nz)
        zu = torch.einsum('ij, bj -> bi', self.Dzu, u)  # (Nz, Nu)*(Nd, Nu)->(Nd, Nz)
        zw = torch.zeros(zu.shape) if self.Dzw is None else torch.zeros(zu.shape)  # Not sure how to implement this yet... Should be something like :torch.einsum('ij, bj -> bi', self.Dzw, w)  # (Nz, Nx)*(Nd, Nx)->(Nd, Nz)
        return zx + zu + zw

    def compute_ynet_contribution(self, w):
        # in:                | out:
        #  - w (Nd, Nw)      |  - Dyw*w (Nd, Ny)
        return torch.einsum('ij, bj -> bi', self.Dyw, w) # (Ny, Nw)*(Nd, Nw)->(Nd, Ny)

    def compute_xnet_contribution(self, w):
        # in:                | out:
        #  - w (Nd, Nw)      |  - Bw*w (Nd, Nx)
        return torch.einsum('ij, bj -> bi', self.Bw, w)  # (Nx, Nw)*(Nd, Nw)->(Nd, Nx)

    def forward(self,x, u):
        # in:           | out:
        #  - x (Nd, Nx) |  - y  (Nd, Ny)
        #  - u (Nd, Nu) |  - x+ (Nd, Nx)
        # compute network contribution
        z = self.compute_z(x, u)
        w = self.net(z)
        x_plus = self.sys.f(x,u) + self.compute_xnet_contribution(w)
        y_k    = self.sys.h(x,u) + self.compute_ynet_contribution(w)
        return y_k, x_plus


###################################################################################
##############                  DYNAMIC AUGMENTATION                 ##############
###################################################################################
class SSE_DynamicAugmentation(nn.Module):
    def __init__(self, nx, nu, ny, known_system, wnet, initial_scaling_factor=1e-3, Dzw_is_zero=True, feedthrough=True):
        super(SSE_DynamicAugmentation, self).__init__()
        # First verify if we have the correct system type and augmentationparameters
        verifySystemType(known_system)
        verifyNetType(wnet, 'dynamic')
        # Save parameters
        self.sys = known_system
        self.net = wnet
        self.Nu  = self.sys.Nu
        self.Nx  = self.sys.Nx
        self.Ny  = self.sys.Ny
        self.Nz  = self.net.n_in
        self.Nw  = self.net.n_out
        self.Nxh = self.net.n_state
        self.Bw  = nn.Parameter(data=initial_scaling_factor * torch.rand(self.Nx, self.Nw))
        self.Cz  = nn.Parameter(data=initial_scaling_factor * torch.rand(self.Nz, self.Nx))
        self.Dzw = None if Dzw_is_zero else nn.Parameter(data=initial_scaling_factor * torch.rand(self.Nz, self.Nw))
        self.Dzu = nn.Parameter(data=initial_scaling_factor * torch.rand(self.Nz, self.Nu))
        self.Dyw = nn.Parameter(data=initial_scaling_factor * torch.rand(self.Ny, self.Nw))
        assert self.Dzw is None, 'No implementation yet for non-zero Dzw'

    def initialize_parameters(self, Bw=None, Cz=None, Dzw=None, Dzu=None, Dyw=None):
        self.Bw.data = assign_param(self.Bw, Bw, 'Bw')
        self.Cz.data = assign_param(self.Cz, Cz, 'Cz')
        #self.Dzw.data = assign_param(self.Dzw, Dzw, 'Dzw')
        self.Dzu.data = assign_param(self.Dzu, Dzu, 'Dzu')
        self.Dyw.data = assign_param(self.Dyw, Dyw, 'Dyw')

    def compute_z(self, x, w, u):
        # in:            | out:
        #  - x (Nd, Nx)  |  - z (Nd, Nz)
        #  - u (Nd, Nu)  |
        zx = torch.einsum('ij, bj -> bi', self.Cz, x)   # (Nz, Nx)*(Nd, Nx)->(Nd, Nz)
        zu = torch.einsum('ij, bj -> bi', self.Dzu, u)  # (Nz, Nu)*(Nd, Nu)->(Nd, Nz)
        zw = torch.zeros(zu.shape) if self.Dzw is None else torch.zeros(zu.shape)  # Not sure how to implement this yet... Should be something like :torch.einsum('ij, bj -> bi', self.Dzw, w)  # (Nz, Nx)*(Nd, Nx)->(Nd, Nz)
        return zx + zu + zw

    def compute_ynet_contribution(self, w):
        # in:                | out:
        #  - w (Nd, Nw)      |  - Dyw*w (Nd, Ny)
        return torch.einsum('ij, bj -> bi', self.Dyw, w) # (Ny, Nw)*(Nd, Nw)->(Nd, Ny)

    def compute_xnet_contribution(self, w):
        # in:                | out:
        #  - w (Nd, Nw)      |  - Bw*w (Nd, Nx)
        return torch.einsum('ij, bj -> bi', self.Bw, w)  # (Nx, Nw)*(Nd, Nw)->(Nd, Nx)

    def forward(self,x, u):
        # in:                 | out:
        #  - x (Nd, Nx + Nxh) |  - x+ (Nd, Nx + Nxh)
        #  - u (Nd, Nu)       |  - y  (Nd, Ny)
        # split up the state from the encoder in the state of the known part
        # and the state of the unknown (to be learned) part
        if x.ndim == 1: # shape of x is (Nx+Nxh)
            assert x.shape != self.Nx+self.Nxh, 'dimension of state is not nx+nx_hidden... nx = ' +  str(self.Nx) + \
                                                ', nx_h = ' + str(self.Nxh) + ', while x.shape = ' + str(x.shape)
            x_known = x[:self.Nx]
            x_learn = x[-self.Nxh:]
        else:
            x_known = x[:, :self.Nx]
            x_learn = x[:, -self.Nxh:]
        # compute the input for the network
        z = self.compute_z(x=x_known, w=None, u=u)  # z = Cz x + Dzw w + Dzu u  --> Dzw = 0
        # calculate w from NN and update hidden state
        x_learn_plus, w = self.net(hidden_state=x_learn, u=z)  # u_net = z_model
        x_known_plus = self.sys.f(x_known, u) + self.compute_xnet_contribution(w)
        y_k          = self.sys.h(x_known, u) + self.compute_ynet_contribution(w)
        x_plus = torch.cat((x_known_plus,x_learn_plus), dim=x.ndim-1)
        return y_k, x_plus


###################################################################################
##############                  ADDITIVE AUGMENTATION                 ##############
###################################################################################
class SSE_AdditiveAugmentation(nn.Module):
    def __init__(self, nx, nu, ny, known_system, wnet, regLambda=0, orthLambda=0, feedthrough=True):
        super(SSE_AdditiveAugmentation, self).__init__()
        # First verify if we have the correct system type and augmentationparameters
        verifySystemType(known_system)
        verifyNetType(wnet, 'static')
        # Save parameters
        self.sys = known_system
        self.xnet = wnet
        self.Nu = self.sys.Nu
        self.Nx = self.sys.Nx
        self.Ny = self.sys.Ny
        if hasattr(known_system, 'parm_corr_enab'):
            self.Pcorr_enab = known_system.parm_corr_enab
        else:
            self.Pcorr_enab = False
        if self.Pcorr_enab:
            self.P_orig = known_system.P.detach()
            self.Pcorr = nn.Parameter(data=0.0*self.P_orig.data)
            self.regLambda = regLambda

    def calculate_xnet(self,x,u):
        # in:               | out:
        #  - x (Nd, Nx)     |  - x+ (Nd, Nx)
        #  - u (Nd, Nu)     |

        if u.ndim == 1:
            u = torch.unsqueeze(u, dim=1)
        xnorm = (x - self.sys.xmean) / self.sys.xstd
        unorm = (u - self.sys.umean) / self.sys.ustd

        xnet_input = torch.cat((xnorm,unorm), dim=1)
        xplus_bb = self.xnet(xnet_input)
        return  xplus_bb * self.sys.xstd + self.sys.xmean

    def forward(self, x, u):
        # in:               | out:
        #  - x (Nd, Nx)     |  - x+ (Nd, Nx)
        #  - u (Nd, Nu)     |  - y  (Nd, Ny)

        if self.Pcorr_enab:
            x_plus = self.sys.f(x, u, self.P_orig + self.Pcorr) + self.calculate_xnet(x, u)
        else:
            x_plus = self.sys.f(x, u) + self.calculate_xnet(x, u)

        y_k = self.sys.h(x,u)
        return y_k, x_plus

class SSE_AdditiveDynAugmentation(nn.Module):
    def __init__(self, nx, nx_h, nu, ny, known_system, wnet, regLambda=0, orthLambda=0, feedthrough=True):
        super(SSE_AdditiveDynAugmentation, self).__init__()
        # First verify if we have the correct system type and augmentationparameters
        verifySystemType(known_system)
        verifyNetType(wnet, 'static')  # for additive augmentation it will probably remain simple ffw net, but ???
        # Save parameters
        self.sys = known_system
        self.xnet = wnet
        self.Nu = self.sys.Nu
        self.Nx = self.sys.Nx
        self.Nx_h = nx_h
        self.Ny = self.sys.Ny
        if hasattr(known_system, 'parm_corr_enab'):
            self.Pcorr_enab = known_system.parm_corr_enab
        else:
            self.Pcorr_enab = False
        if self.Pcorr_enab:
            self.P_orig = known_system.P.detach()
            self.Pcorr = nn.Parameter(data=0.0*self.P_orig.data)
            self.regLambda = regLambda

    def calculate_xnet(self,x,u):
        # in:               | out:
        #  - x (Nd, Nx)     |  - x+ (Nd, Nx)
        #  - u (Nd, Nu)     |

        if u.ndim == 1:
            u = torch.unsqueeze(u, dim=0)
        #xnorm = (x - self.sys.xmean) / self.sys.xstd
        #unorm = (u - self.sys.umean) / self.sys.ustd

        #xnet_input = torch.cat((xnorm,unorm), dim=1)
        xnet_input = torch.cat((x, u), dim=1)
        xplus_bb = self.xnet(xnet_input)
        #ToDo: implement standardization
        return  xplus_bb #* self.sys.xstd + self.sys.xmean

    def forward(self, x, u):
        # in:                       | out:
        #  - x (Nd, Nx_m+Nx_h)      |  - x+ (Nd, Nx_m+Nx_h)
        #  - u (Nd, Nu)             |  - y  (Nd, Ny)

        if x.ndim == 1: # shape of x is (Nx+Nxh)
            assert x.shape != self.Nx+self.Nxh, 'dimension of state is not nx+nx_hidden... nx = ' +  str(self.Nx) + \
                                                ', nx_h = ' + str(self.Nxh) + ', while x.shape = ' + str(x.shape)
            x_meas = x[:self.Nx]
            x_hidden = x[-self.Nx_h:]
        else:
            x_meas = x[:, :self.Nx]
            x_hidden = x[:, -self.Nx_h:]

        x_plus_bb = self.calculate_xnet(x, u)  # black-box part with measured and hidden states

        # first-principles model part
        if self.Pcorr_enab:
            x_plus_fp = self.sys.f(x_meas, u, self.P_orig + self.Pcorr)
        else:
            x_plus_fp = self.sys.f(x_meas, u)

        x_plus = torch.hstack((x_plus_fp, torch.zeros(x.size(dim=0), self.Nx_h))) + x_plus_bb
        y_k = self.sys.h(x_meas,u)
        return y_k, x_plus


class SineAugmentation(nn.Module):
    def __init__(self, nx, nu, ny, known_system, wnet, regLambda=0, orthLambda=0, feedthrough=True):
        super(SineAugmentation, self).__init__()
        # First verify if we have the correct system type and augmentationparameters
        verifySystemType(known_system)
        verifyNetType(wnet, 'static')
        # Save parameters
        self.sys = known_system
        self.xnet = wnet
        self.Nu = known_system.Nu
        self.Nx = known_system.Nx
        self.Ny = known_system.Ny
        if hasattr(known_system, 'parm_corr_enab'):
            self.Pcorr_enab = known_system.parm_corr_enab
        else:
            self.Pcorr_enab = False
        if self.Pcorr_enab:
            self.regLambda = regLambda
            self.orthLambda = orthLambda

    def calculate_xnet(self,x,u):
        # in:               | out:
        #  - x (Nd, Nx)     |  - x+ (Nd, Nx)
        #  - u (Nd, Nu)     |

        if u.ndim == 1:
            u = torch.unsqueeze(u, dim=0)
        xnorm = (x - self.sys.xmean) / self.sys.xstd
        unorm = (u - self.sys.umean) / self.sys.ustd

        xnet_input = torch.cat((xnorm,unorm), dim=1)
        xplus_bb = self.xnet(xnet_input)
        return  xplus_bb * self.sys.xstd + self.sys.xmean

    def forward(self, x, u):
        # in:               | out:
        #  - x (Nd, Nx)     |  - x+ (Nd, Nx)
        #  - u (Nd, Nu)     |  - y  (Nd, Ny)

        '''
        if self.Pcorr_enab:
            #P_corr = self.P_weight * self.P_orig + self.P_bias
            x_plus = self.sys.f(x, u, self.Pcorr) + self.calculate_xnet(x, u)
        else:
        '''

        x_plus = self.sys.f(x, u) + self.calculate_xnet(x, u)

        y_k = self.sys.h(x,u)
        return y_k, x_plus

    def calculate_orthogonalisation(self, x, u):
        # in:               | out:
        #  - x (Nd, Nx)     |  - cost
        #  - u (Nd, Nu)     |

        # for sine test only
        Matrix = torch.hstack((x, u))
        U1, _, _ = torch.linalg.svd(Matrix, full_matrices=False)
        orthogonal_components = U1 @ U1.T @ self.calculate_xnet(x, u)
        cost = self.orthLambda * torch.linalg.vector_norm(orthogonal_components)
        return cost





