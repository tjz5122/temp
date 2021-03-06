import math
import torch
import torch.nn.functional as F
from torch.optim import Optimizer
from scipy import stats
import numpy as np


use_cuda = torch.cuda.is_available()
print('Use GPU?', use_cuda)

class SSM_Optimizer(Optimizer):

    def __init__(self, params, lr=-1, momentum=0, weight_decay=0):
        # nu can take values outside of the interval [0,1], but no guarantee of convergence?
        if lr <= 0:
            raise ValueError("Invalid value for learning rate (>0): {}".format(lr))
        if momentum < 0 or momentum > 1:
            raise ValueError("Invalid value for momentum [0,1): {}".format(momentum))
        if weight_decay < 0:
            raise ValueError("Invalid value for weight_decay (>=0): {}".format(weight_decay))

        defaults = dict(lr=lr, momentum=momentum, weight_decay=weight_decay)
        super(SSM_Optimizer, self).__init__(params, defaults)


    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        self.add_weight_decay()
        self.SSM_direction_and_update()
        return loss

    def add_weight_decay(self):
        
        for group in self.param_groups:
            weight_decay = group['weight_decay']
            for p in group['params']:
                if p.grad is None:
                    continue
                if weight_decay > 0:
                    p.grad.data.add_(weight_decay, p.data)

    def SSM_direction_and_update(self, dampening = 0):

        for group in self.param_groups:
            momentum = group['momentum']
            for p in group['params']:
                if p.grad is None:
                    continue
                param_state = self.state[p]
                g_k = p.grad.data
                # get momentum buffer.
                if 'momentum_buffer' not in param_state:
                    buf = param_state['momentum_buffer'] = torch.zeros_like(p.data)
                else:
                    buf = param_state['momentum_buffer']
                buf.mul_(momentum).add_(1.0 - dampening, g_k)
            
                p.data.add_(-group['lr'], buf)
                
                    
                    
class Bucket(object):
    def __init__(self, size, ratio, dtype, device, fixed_len=-1):
        self.size = size
        self.ratio = int(ratio)
        self.fixed_len = int(fixed_len)

        self.buffer = torch.zeros(size, dtype=dtype, device=device)
        self.count = 0          # number of elements kept in queue (excluding leaked)
        self.start = 0          # count = end - start
        self.end = 0
        self.total_count = 0    # total number of elements added (including leaked)
        self.statistic = 0
 
    def reset(self):
        self.buffer.zero_()    
        self.count = 0          
        self.start = 0
        self.end = 0
        self.total_count = 0

    def double_size(self):
        self.size *= 2
        self.buffer.resize_(self.size)

    def add(self, val):
        if self.end == self.size:               # when the end index reach size
            self.double_size()                      # double the size of buffer

        self.buffer[self.end] = val             # always put new value at the end
        self.end += 1                           # and increase end index by one

        if self.fixed_len > 0:
            if self.count == self.fixed_len:
                self.start += 1
            else:
                self.count += 1
        else:
            if self.total_count % self.ratio == 0:  # if leaky_count is multiple of ratio
                self.count += 1                         # increase count in queue by one
            else:                                   # otherwise leak and keep same count
                self.start += 1                         # increase start index by one

        self.total_count += 1                   # always increase total_count by one

        # reset start index to 0 and end index to count to save space
        if self.start >= self.count:
            self.buffer[0:self.count] = self.buffer[self.start:self.end]
            self.start = 0
            self.end = self.count

    # ! Need to add safeguard to allow compute only if there are enough entries
    def mean_std(self, mode='bm'):
        mean = torch.mean(self.buffer[self.start:self.end]).item() # mean of whole

        if mode == 'bm':        # batch mean variance
            b_n = int(math.floor(math.sqrt(self.count))) #batch size
            Yks = F.avg_pool1d(self.buffer[self.start:self.end].unsqueeze(0).unsqueeze(0), kernel_size=b_n, stride=b_n).view(-1) # batch mean
            diffs = Yks - mean
            std = math.sqrt(b_n /(len(Yks)-1))*torch.norm(diffs).item() # len(Yks) is number of batch
            dof = b_n - 1
        elif mode == 'olbm':    # overlapping batch mean
            b_n = int(math.floor(math.sqrt(self.count)))
            Yks = F.avg_pool1d(self.buffer[self.start:self.end].unsqueeze(0).unsqueeze(0), kernel_size=b_n, stride=1).view(-1)
            diffs = Yks - mean
            std = math.sqrt(b_n*self.count/(len(Yks)*(len(Yks)-1)))*torch.norm(diffs).item()
            dof = self.count - b_n
        else:                   # otherwise use mode == 'iid'
            std = torch.std(self.buffer[self.start:self.end]).item()
            dof = self.count - 1

        return mean, std, dof

    def stats_test(self, sigma, tolerance, mode='bm',step_test=1,truncate=0.02):
        mean, std, dof = self.mean_std(mode=mode)

        # confidence interval
        t_sigma_dof = stats.t.ppf(1-sigma/2., dof)
        self.statistic = std * t_sigma_dof / math.sqrt(self.count)
        if step_test != 0:
            self.statistic -= truncate
            
        return self.statistic < tolerance


class SSM(SSM_Optimizer):

    def __init__(self, params, lr=-1, momentum=0, weight_decay=0, dampening = 0,
                 drop_factor=10, significance=0.05, tolerance = 0.01, var_mode='bm',
                 leak_ratio=8, minN_stats=100, testfreq=100, samplefreq = 10, trun=0.02, mode='loss_plus_smooth'):

        if lr <= 0:
            raise ValueError("Invalid value for learning rate (>0): {}".format(lr))
        if momentum < 0 or momentum > 1:
            raise ValueError("Invalid value for momentum [0,1): {}".format(momentum))
        if weight_decay < 0:
            raise ValueError("Invalid value for weight_decay (>=0): {}".format(weight_decay))
        if drop_factor < 1:
            raise ValueError("Invalid value for drop_factor (>=1): {}".format(drop_factor))
        if significance <= 0 or significance >= 1:
            raise ValueError("Invalid value for significance (0,1): {}".format(significance))
        if var_mode not in ['bm', 'olbm', 'iid']:
            raise ValueError("Invalid value for var_mode ('bm', 'olbm', or 'iid'): {}".format(var_mode))
        if leak_ratio < 1:
            raise ValueError("Invalid value for leak_ratio (int, >=1): {}".format(leak_ratio))
        # if minN_stats < 100:
        #     raise ValueError("Invalid value for minN_stats (int, >=100): {}".format(minN_stats))
        if testfreq < 1:
            raise ValueError("Invalid value for testfreq (int, >=1): {}".format(testfreq))

        super(SSM, self).__init__(params, lr=lr, momentum=momentum, weight_decay=weight_decay)
        # New Python3 way to call super()
        # super().__init__(params, lr=lr, momentum=momentum, nu=nu, weight_decay=weight_decay)

        # State initialization: leaky bucket belongs to global state.
        p = self.param_groups[0]['params'][0]
        if 'bucket' not in self.state:
            self.state['bucket'] = Bucket(1000, leak_ratio, p.dtype, p.device)

        self.state['lr'] = float(lr)
        self.state['momemtum'] = float(momentum)
        self.state['drop_factor'] = drop_factor
        self.state['significance'] = significance
        self.state['tolerance'] = tolerance
        self.state['var_mode'] = var_mode
        self.state['minN_stats'] = int(minN_stats)
        self.state['dampening'] = dampening
        self.state['samplefreq'] = int(samplefreq)
        self.state['testfreq'] = int(testfreq)
        self.state['nSteps'] = 0
        self.state['loss'] = 0
        self.state['mode'] = mode
        self.state['step_test'] = 0
        self.state['truncate'] = trun


        # statistics to monitor
        self.state['smoothing'] = 0
        self.state['stats_x1d'] = 0
        self.state['stats_ld2'] = 0
        self.state['stats_val'] = 0
        self.state['stats_test'] = 0
        self.state['statistic'] = 0
        self.state['stats_stationary'] = 0
        self.state['stats_mean'] = 0


    def step(self, closure=None):
        """
        Performs a single optimization step.
        Arguments:
            closure (callable, optional): A closure that reevaluates model and returns loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        self.add_weight_decay()
        self.SSM_direction_and_update(dampening = self.state['dampening'])
        self.state['nSteps'] += 1
        self.stats_adaptation()

        return loss

    def stats_adaptation(self):

        dk = self._gather_flat_buffer('step_buffer')
        xk1 = self._gather_flat_param() 
        gk = self._gather_flat_grad()
        
        
        if self.state['nSteps'] % self.state['samplefreq'] == 0:

            if self.state['mode'] == 'loss_plus_smooth':
                self.state['tolerance'] = 0.01
                self.state['smoothing'] = xk1.dot(gk).item() - (0.5 * self.state['lr']) * ((1 + self.state['momemtum'])/(1 - self.state['momemtum'])) * (dk.dot(dk).item())
                if self.state['step_test'] == 0:
                    self.state['stats_val'] =  self.state['loss'] + self.state['smoothing']
                else:
                    self.state['loss'] = np.log10(self.state['loss']) / np.log10(10/self.state['step_test']) 
                    self.state['stats_val'] = self.state['loss']  + self.state['smoothing']
                    
                    
            if self.state['mode'] == 'loss':
                self.state['tolerance'] = 0.005
                if self.state['step_test'] == 0:
                    self.state['stats_val'] =  self.state['loss'] 
                else:
                    self.state['loss'] = np.log10(self.state['loss']) / np.log10(10/self.state['step_test']) 
                    self.state['stats_val'] = self.state['loss']  
                
                
            if self.state['mode'] == 'sasa_plus':
                self.state['stats_x1d'] = xk1.dot(dk).item()
                self.state['stats_ld2'] = (0.5 * self.state['lr']) * (dk.dot(dk).item())
                self.state['stats_val'] = self.state['stats_x1d'] + self.state['stats_ld2']
        
            # add statistic to leaky bucket
            self.state['bucket'].add(self.state['stats_val'])
        
        # check statistics and adjust learning rate
        self.state['stats_test'] = 0
        self.state['stats_stationary'] = 0
        self.state['statistic'] = 0
        if self.state['bucket'].count > self.state['minN_stats'] and self.state['nSteps'] % self.state['testfreq'] == 0:
            stationary= self.state['bucket'].stats_test(self.state['significance'], self.state['tolerance'], self.state['var_mode'],self.state['step_test'],self.state['truncate'])
            self.state['stats_test'] = 1
            self.state['statistic'] = self.state['bucket'].statistic
            self.state['stats_stationary'] = int(stationary)
    
            # perform statistical test for stationarity
            if self.state['stats_stationary'] == 1:
                self.state['lr'] /= self.state['drop_factor']
                self.state['step_test'] += 1
                for group in self.param_groups:
                    group['lr'] = self.state['lr']
                self._zero_buffers('momentum_buffer')
                self.state['bucket'].reset()


    def _gather_flat_grad(self):
        views = []
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    view = p.data.new(p.data.numel()).zero_()
                elif p.grad.data.is_sparse:
                    view = p.grad.data.to_dense().view(-1) #take derivative of p
                else:
                    view = p.grad.data.view(-1) 
                views.append(view)
        return torch.cat(views, 0)
    
    # methods for gather flat parameters
    def _gather_flat_param(self):
        views = []
        for group in self.param_groups:
            for p in group['params']:
                view = p.data.view(-1)
                views.append(view)
        return torch.cat(views, 0)

    # method for gathering/initializing flat buffers that are the same shape as the parameters
    def _gather_flat_buffer(self, buf_name):
        views = []
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                if buf_name not in state:  # init buffer
                    view = p.data.new(p.data.numel()).zero_()
                else:
                    view = state[buf_name].data.view(-1)
                views.append(view)
        return torch.cat(views, 0) 
    
    def _zero_buffers(self, buf_name):
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                if buf_name in state:
                    state[buf_name].zero_()
        return None
    
