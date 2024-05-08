import warnings

import deepSI
import numpy as np
import torch
from torch import nn
from deepSI import model_augmentation


def simple_res_net_2_LFR(net):
    seq = net.net_non_lin.net
    Nu = seq[0].in_features
    Nh = seq[0].out_features
    Ny = seq[-1].out_features
    nr_layers = int(0.5*(len(seq)-1))
    Dzu = torch.cat((seq[0].weight.data ,torch.zeros(((nr_layers-1)*Nh, Nu))),dim=0)
    Dyw = torch.cat((torch.zeros((Ny, (nr_layers - 1) * Nh)), seq[-1].weight.data), dim=1)
    Dyu = net.net_lin.weight.data
    by  = net.net_lin.bias.data + seq[-1].bias.data
    bz = seq[0].bias.data
    if nr_layers == 1:
        Dzw = torch.zeros((Nh,Nh))
    else:
        Dzw = torch.zeros((Nh, nr_layers * Nh))
        for i in range(1,nr_layers):
            Dzw_row = torch.cat((torch.zeros((Nh, (i - 1) * Nh)), seq[2 * i].weight.data, torch.zeros((Nh, Nh*(nr_layers - i)))), dim=1)
            Dzw = torch.cat((Dzw, Dzw_row))
            bz  = torch.cat((bz, seq[2 * i].bias.data))
    return Dzw, Dzu, Dyw, Dyu, bz, by


# Allowed systems:
def verifySystemType(sys):
    if   type(sys).__base__ is model_augmentation.lpvsystem.lpv_model_grid: return
    elif type(sys).__base__ is model_augmentation.lpvsystem.lpv_model_aff:  return
    elif type(sys)          is model_augmentation.lpvsystem.lti_system:     return
    elif type(sys).__base__ is model_augmentation.lpvsystem.lti_system:     return
    elif type(sys)          is model_augmentation.lpvsystem.lti_affine_system: return
    elif type(sys).__base__ is model_augmentation.lpvsystem.lti_affine_system: return
    elif type(sys).__base__ is model_augmentation.lpvsystem.general_nonlinear_system: return
    else: raise ValueError("Systems must be of the types defined in 'model_augmentation.lpvsystem'")

def verifyNetType(net,nettype):
    if nettype in 'static':
        if type(net) is not deepSI.utils.contracting_REN: return
        elif type(net) is not deepSI.utils.LFR_ANN: return
        else: raise ValueError("Static network required...")
    elif nettype in 'dynamic':
        if type(net) is deepSI.utils.contracting_REN: return
        elif type(net) is deepSI.utils.LFR_ANN: return
        else: raise ValueError("Dynamic network required...")
    else: raise ValueError('Unknown net type, only dynamic or static supported')

# some generic functions
def to_torch_tensor(A): # Obsolete?
    if torch.is_tensor(A):
        return A
    else:
        return torch.tensor(A, dtype=torch.float)

def RK4_step(f, x, u, h): # Functions of the form f(x,u). See other scripts for time-varying cases
    # one step of runge-kutta integration. u is zero-order-hold
    k1 = h * f(x, u)
    k2 = h * f(x + k1 / 2, u)
    k3 = h * f(x + k2 / 2, u)
    k4 = h * f(x + k3, u)
    return x + 1 / 6 * (k1 + 2 * k2 + 2 * k3 + k4)

# Function used for parameter initialization
def assign_param(A_old, A_new, nm):
    if A_new is not None:
        assert torch.is_tensor(A_new), nm + ' must be of the Tensor type'
        assert A_new.size() == A_old.size(), nm + ' must be of size' + str(A_old.size())
        return A_new.data
    else:
        return A_old.data


# Calculating the SVD of the training data-set for orthogonalization-based regularization
def calculate_orthogonalisation(sys, train_data, x_meas=False, mini_batch_size=None):
    # in:               | out:
    #  - x (Nd, Nx)     |  - cost
    #  - u (Nd, Nu)     |

    if x_meas: # when y=x
        x = train_data.y
        u = train_data.u
    else:
        sys_data = sys.apply_experiment(train_data)
        x = sys_data.x
        u = sys_data.u

    if mini_batch_size is not None:
        batch_strt = int(torch.rand(1) * (x.shape[0] - mini_batch_size))
        batch_end = batch_strt + mini_batch_size
        x = x[batch_strt:batch_end, :]
        u = u[batch_strt:batch_end, :]
    Matrix = sys.calculate_orth_matrix(x, u)
    U1, _, _ = torch.linalg.svd(Matrix, full_matrices=False)
    return U1, torch.tensor(x, dtype=torch.float), torch.tensor(u, dtype=torch.float)


def initialize_augmentation_net(network, augm_type):
    if augm_type in 'additive':
        init_additive_augmentation_net(network)
    elif augm_type in 'multiplicative':
        init_multiplicative_augmentation_net(network)


# Function for initializing neural networks in additive structure
def init_additive_augmentation_net(network):
    if type(network) is deepSI.utils.torch_nets.simple_res_net:
        # If the network is residual neur. net. (has linear part)
        network.net_lin.weight.data.fill_(0.0)
        if network.net_non_lin is not None:  # has nonlinear part
            network.net_non_lin.net[-1].weight.data.fill_(0.0)
            network.net_non_lin.net[-1].bias.data.fill_(0.0)
        else: # if only linear part is present, then it has bias value
            network.net_lin.bias.data.fill_(0.0)
    elif type(network) is deepSI.utils.torch_nets.feed_forward_nn:
        # for simple feedforward nets
        network.net[-1].weight.data.fill_(0.0)
        network.net[-1].bias.data.fill_(0.0)
    else:
        warnings.warn("Neural network type should be either 'deepSI.utils.torch_nets.simple_res_net'"
                      "or 'deepSI.utils.torch_nets.feed_forward_nn' for accurate initialization.")

def init_multiplicative_augmentation_net(network):
    if type(network) is deepSI.utils.torch_nets.simple_res_net:
        # If the network is residual neur. net. (has linear part)
        network.net_lin.weight.data.fill_(0.0)
        for i in range(network.n_out):
            idx1 = i
            idx2 = i + network.n_in - network.n_out
            network.net_lin.weight.data[idx1, idx2].fill_(1.0)
        if network.net_non_lin is not None:  # has nonlinear part
            network.net_non_lin.net[-1].weight.data.fill_(0.0)
            network.net_non_lin.net[-1].bias.data.fill_(0.0)
        else:  # if only linear layer is present, then it has a bias value
            network.net_lin.bias.data.fill_(0.0)
    else:
        warnings.warn("Neural network type should be 'deepSI.utils.torch_nets.simple_res_net' for accurate initialization.")