from turtle import forward
import torch
import numpy as np
import torch.nn as nn
from einops import rearrange, repeat
from torch.nn import init
from torch.nn import functional as F

from torch.autograd import Variable
import math
from torch.optim.lr_scheduler import LambdaLR
import logging
from torch.optim import SGD, Adam, lr_scheduler
import sys
import os

class Classifier(nn.Module):
    """LeNet classifier model for ADDA."""

    def __init__(self, input_size, hidden_dims, num_class, dropout_rate = 0.2):
        """Init LeNet encoder."""
        super(Classifier, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_dims)
        self.fc2 = nn.Linear(hidden_dims, num_class)
        self.relu = nn.ReLU(inplace=False)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, feat):
        """Forward the LeNet classifier."""
        out = self.dropout(self.relu(feat))
        out = self.fc1(out)
        out = self.relu(out)
        out = self.fc2(out)
        return out
class FrozenBatchNorm3d(nn.Module):
    def __init__(self, n):
        super(FrozenBatchNorm3d, self).__init__()
        self.register_buffer("weight", torch.ones(n))
        self.register_buffer("bias", torch.zeros(n))
        self.register_buffer("running_mean", torch.zeros(n))
        self.register_buffer("running_var", torch.ones(n))

    def forward(self, x):
        # Cast all fixed parameters to half() if necessary
        if x.dtype == torch.float16:
            self.weight = self.weight.half()
            self.bias = self.bias.half()
            self.running_mean = self.running_mean.half()
            self.running_var = self.running_var.half()

        scale = self.weight * self.running_var.rsqrt()
        bias = self.bias - self.running_mean * scale
        # print('x:',x.size())
        # print('scale:',scale.size())
        # print('bias:',bias.size())
        scale = scale.reshape(1, -1, 1, 1, 1)
        bias = bias.reshape(1, -1, 1, 1, 1)
        return x * scale + bias

    def __repr__(self):
        s = self.__class__.__name__ + "("
        s += "{})".format(self.weight.shape[0])
        return s

class FrozenBatchNorm2d(nn.Module):
    def __init__(self, n):
        super(FrozenBatchNorm2d, self).__init__()
        self.register_buffer("weight", torch.ones(n))
        self.register_buffer("bias", torch.zeros(n))
        self.register_buffer("running_mean", torch.zeros(n))
        self.register_buffer("running_var", torch.ones(n))

    def forward(self, x):
        # Cast all fixed parameters to half() if necessary
        if x.dtype == torch.float16:
            self.weight = self.weight.half()
            self.bias = self.bias.half()
            self.running_mean = self.running_mean.half()
            self.running_var = self.running_var.half()

        scale = self.weight * self.running_var.rsqrt()
        bias = self.bias - self.running_mean * scale
        # print('x:',x.size())
        # print('scale:',scale.size())
        # print('bias:',bias.size())
        scale = scale.reshape(1, -1, 1, 1)
        bias = bias.reshape(1, -1, 1, 1)
        return x * scale + bias

    def __repr__(self):
        s = self.__class__.__name__ + "("
        s += "{})".format(self.weight.shape[0])
        return s

class MetaModule(nn.Module):
    # adopted from: Adrien Ecoffet https://github.com/AdrienLE
    def params(self):
        for name, param in self.named_params(self):
            yield param

    def named_leaves(self):
        return []

    def named_submodules(self):
        return []

    def named_params(self, curr_module=None, memo=None, prefix=''):
        if memo is None:
            memo = set()

        if hasattr(curr_module, 'named_leaves'):
            for name, p in curr_module.named_leaves():
                if p is not None and p not in memo:
                    memo.add(p)
                    yield prefix + ('.' if prefix else '') + name, p
        else:
            for name, p in curr_module._parameters.items():
                if p is not None and p not in memo:
                    memo.add(p)
                    yield prefix + ('.' if prefix else '') + name, p

        for mname, module in curr_module.named_children():
            submodule_prefix = prefix + ('.' if prefix else '') + mname
            for name, p in self.named_params(module, memo, submodule_prefix):
                yield name, p

    def update_params(self, lr_inner, source_params=None,
                      solver='sgd', beta1=0.9, beta2=0.999, weight_decay=5e-4):
        if solver == 'sgd':
            for tgt, src in zip(self.named_params(self), source_params):
                name_t, param_t = tgt
                grad = src if src is not None else 0
                tmp = param_t - lr_inner * grad
                self.set_param(self, name_t, tmp)
        elif solver == 'adam':
            for tgt, gradVal in zip(self.named_params(self), source_params):
                name_t, param_t = tgt
                exp_avg, exp_avg_sq = torch.zeros_like(param_t.data), \
                                      torch.zeros_like(param_t.data)
                bias_correction1 = 1 - beta1
                bias_correction2 = 1 - beta2
                gradVal.add_(weight_decay, param_t)
                exp_avg.mul_(beta1).add_(1 - beta1, gradVal)
                exp_avg_sq.mul_(beta2).addcmul_(1 - beta2, gradVal, gradVal)
                exp_avg_sq.add_(1e-8)  # to avoid possible nan in backward
                denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(1e-8)
                step_size = lr_inner / bias_correction1
                newParam = param_t.addcdiv(-step_size, exp_avg, denom)
                self.set_param(self, name_t, newParam)

    def setParams(self, params):
        for tgt, param in zip(self.named_params(self), params):
            name_t, _ = tgt
            self.set_param(self, name_t, param)

    def set_param(self, curr_mod, name, param):
        if '.' in name:
            n = name.split('.')
            module_name = n[0]
            rest = '.'.join(n[1:])
            for name, mod in curr_mod.named_children():
                if module_name == name:
                    self.set_param(mod, rest, param)
                    break
        else:
            setattr(curr_mod, name, param)

    def setBN(self, inPart, name, param):
        if '.' in name:
            part = name.split('.')
            self.setBN(getattr(inPart, part[0]), '.'.join(part[1:]), param)
        else:
            setattr(inPart, name, param)

    def detach_params(self):
        for name, param in self.named_params(self):
            self.set_param(self, name, param.detach())

    def copyModel(self, newModel, same_var=False):
        # copy meta model to meta model
        tarName = list(map(lambda v: v, newModel.state_dict().keys()))

        # requires_grad
        partName, partW = list(map(lambda v: v[0], newModel.named_params(newModel))), list(
            map(lambda v: v[1], newModel.named_params(newModel)))  # new model's weight

        metaName, metaW = list(map(lambda v: v[0], self.named_params(self))), list(
            map(lambda v: v[1], self.named_params(self)))
        bnNames = list(set(tarName) - set(partName))

        # copy vars
        for name, param in zip(metaName, partW):
            if not same_var:
                param = to_var(param.data.clone(), requires_grad=True)
            self.set_param(self, name, param)
        # copy training mean var
        tarName = newModel.state_dict()
        for name in bnNames:
            param = to_var(tarName[name], requires_grad=False)
            self.setBN(self, name, param)

    def copyWeight(self, modelW):
        # copy state_dict to buffers
        curName = list(map(lambda v: v[0], self.named_params(self)))
        tarNames = set()
        for name in modelW.keys():
            # print(name)
            if name.startswith("module"):
                tarNames.add(".".join(name.split(".")[1:]))
            else:
                tarNames.add(name)
        bnNames = list(tarNames - set(curName))  
        for tgt in self.named_params(self):
            name_t, param_t = tgt
            # print(name_t)
            module_name_t = 'module.' + name_t
            if name_t in modelW:
                param = to_var(modelW[name_t], requires_grad=True)
                self.set_param(self, name_t, param)
            elif module_name_t in modelW:
                param = to_var(modelW['module.' + name_t], requires_grad=True)
                self.set_param(self, name_t, param)
            else:
                continue


    def copyWeight_eval(self, modelW):
        # copy state_dict to buffers
        curName = list(map(lambda v: v[0], self.named_params(self)))
        tarNames = set()
        for name in modelW.keys():
            # print(name)
            if name.startswith("module"):
                tarNames.add(".".join(name.split(".")[1:]))
            else:
                tarNames.add(name)
        bnNames = list(tarNames - set(curName))  ## in BN resMeta bnNames only contains running var/mean
        for tgt in self.named_params(self):
            name_t, param_t = tgt
            # print(name_t)
            module_name_t = 'module.' + name_t
            if name_t in modelW:
                param = to_var(modelW[name_t], requires_grad=True)
                self.set_param(self, name_t, param)
            elif module_name_t in modelW:
                param = to_var(modelW['module.' + name_t], requires_grad=True)
                self.set_param(self, name_t, param)
            else:
                continue

        for name in bnNames:
            try:
                param = to_var(modelW[name], requires_grad=False)
            except:
                param = to_var(modelW['module.' + name], requires_grad=False)
            self.setBN(self, name, param)

    
class MetaBatchNorm2d(MetaModule):
    def __init__(self, *args, **kwargs):
        super().__init__()
        ignore = nn.BatchNorm2d(*args, **kwargs)

        self.num_features = ignore.num_features
        self.eps = ignore.eps
        self.momentum = ignore.momentum
        self.affine = ignore.affine
        self.track_running_stats = ignore.track_running_stats

        if self.affine:
            self.register_buffer('weight', to_var(ignore.weight.data, requires_grad=True))
            self.register_buffer('bias', to_var(ignore.bias.data, requires_grad=True))

        if self.track_running_stats:
            self.register_buffer('running_mean', torch.zeros(self.num_features))
            self.register_buffer('running_var', torch.ones(self.num_features))
            self.register_buffer('num_batches_tracked', torch.LongTensor([0]).squeeze())
        else:
            self.register_buffer('running_mean', None)
            self.register_buffer('running_var', None)
            self.register_buffer('num_batches_tracked', None)

    def forward(self, x):
        val2 = self.weight.sum()
        res = F.batch_norm(x, self.running_mean, self.running_var, self.weight, self.bias,
                           self.training or not self.track_running_stats, self.momentum, self.eps)
        return res

    def named_leaves(self):
        return [('weight', self.weight), ('bias', self.bias)]

class MetaBatchNorm1d(MetaModule):
    def __init__(self, *args, **kwargs):
        super().__init__()
        ignore = nn.BatchNorm1d(*args, **kwargs)

        self.num_features = ignore.num_features
        self.eps = ignore.eps
        self.momentum = ignore.momentum
        self.affine = ignore.affine
        self.track_running_stats = ignore.track_running_stats

        if self.affine:
            self.register_buffer('weight', to_var(ignore.weight.data, requires_grad=True))
            self.register_buffer('bias', to_var(ignore.bias.data, requires_grad=True))

        if self.track_running_stats:
            self.register_buffer('running_mean', torch.zeros(self.num_features))
            self.register_buffer('running_var', torch.ones(self.num_features))
            self.register_buffer('num_batches_tracked', torch.LongTensor([0]).squeeze())
        else:
            self.register_buffer('running_mean', None)
            self.register_buffer('running_var', None)
            self.register_buffer('num_batches_tracked', None)

    def forward(self, x):
        return F.batch_norm(x, self.running_mean, self.running_var, self.weight, self.bias,
                            self.training or not self.track_running_stats, self.momentum, self.eps)
        ## meta test set this one to False self.training or not self.track_running_stats
    def named_leaves(self):
        return [('weight', self.weight), ('bias', self.bias)]


class MixUpBatchNorm1d(MetaBatchNorm1d):
    def __init__(self, num_features, eps=1e-5, momentum=0.1,
                 affine=True, track_running_stats=True):
        super(MixUpBatchNorm1d, self).__init__(
            num_features, eps, momentum, affine, track_running_stats)

        self.register_buffer('meta_mean1', torch.zeros(self.num_features))
        self.register_buffer('meta_var1', torch.zeros(self.num_features))
        self.register_buffer('meta_mean2', torch.zeros(self.num_features))
        self.register_buffer('meta_var2', torch.zeros(self.num_features))
        self.device_count = torch.cuda.device_count()

    def forward(self, input, MTE='', save_index=0):
        exponential_average_factor = 0.0

        if self.training and self.track_running_stats:
            if self.num_batches_tracked is not None:
                self.num_batches_tracked += 1
                if self.momentum is None:  # use cumulative moving average
                    exponential_average_factor = 1.0 / float(self.num_batches_tracked)
                else:  # use exponential moving average
                    exponential_average_factor = self.momentum

        # calculate running estimates
        if self.training:
            if MTE == 'sample':
                from torch.distributions.normal import Normal
                Distri1 = Normal(self.meta_mean1, self.meta_var1)
                Distri2 = Normal(self.meta_mean2, self.meta_var2)
                sample1 = Distri1.sample([input.size(0), ])
                sample2 = Distri2.sample([input.size(0), ])
                lam = np.random.beta(1., 1.)
                inputmix1 = lam * sample1 + (1-lam) * input
                inputmix2 = lam * sample2 + (1-lam) * input

                mean1 = inputmix1.mean(dim=0)
                var1 = inputmix1.var(dim=0, unbiased=False)
                mean2 = inputmix2.mean(dim=0)
                var2 = inputmix2.var(dim=0, unbiased=False)

                output1 = (inputmix1 - mean1[None, :]) / (torch.sqrt(var1[None, :] + self.eps))
                output2 = (inputmix2 - mean2[None, :]) / (torch.sqrt(var2[None, :] + self.eps))
                if self.affine:
                    output1 = output1 * self.weight[None, :] + self.bias[None, :]
                    output2 = output2 * self.weight[None, :] + self.bias[None, :]
                return [output1,output2]

            else:
                mean = input.mean(dim=0)
                # use biased var in train
                var = input.var(dim=0, unbiased=False)
                n = input.numel() / input.size(1)
                print('mean.size:', mean.size())
                with torch.no_grad():
                    running_mean = exponential_average_factor * mean \
                                   + (1 - exponential_average_factor) * self.running_mean
                    print('running_mean:', running_mean.size())
                    # update running_var with unbiased var
                    running_var = exponential_average_factor * var * n / (n - 1) \
                                  + (1 - exponential_average_factor) * self.running_var
                    self.running_mean.copy_(running_mean)
                    self.running_var.copy_(running_var)
                    if save_index == 0:
                        self.meta_mean1.copy_(mean)
                        self.meta_var1.copy_(var)
                    elif save_index == 1:
                        self.meta_mean2.copy_(mean)
                        self.meta_var2.copy_(var)

        else:
            mean = self.running_mean
            var = self.running_var

        input = (input - mean[None, :]) / (torch.sqrt(var[None, :] + self.eps))
        if self.affine:
            input = input * self.weight[None, :] + self.bias[None, :]

        return input
        
class MixUpBatchNorm2d(MetaBatchNorm2d):
    def __init__(self, num_features, eps=1e-5, momentum=None,
                 affine=True, track_running_stats=True):
        super(MixUpBatchNorm2d, self).__init__(
            num_features, eps, momentum, affine, track_running_stats)

        self.register_buffer('meta_mean1', torch.zeros(self.num_features))
        self.register_buffer('meta_var1', torch.zeros(self.num_features))
        self.register_buffer('meta_mean2', torch.zeros(self.num_features))
        self.register_buffer('meta_var2', torch.zeros(self.num_features))
        self.device_count = torch.cuda.device_count()

    def forward(self, input, MTE='', save_index=0):
        exponential_average_factor = 0.0

        if self.training and self.track_running_stats:
            if self.num_batches_tracked is not None:
                self.num_batches_tracked += 1
                if self.momentum is None:  # use cumulative moving average
                    exponential_average_factor = 1.0 / float(self.num_batches_tracked)
                else:  # use exponential moving average
                    exponential_average_factor = self.momentum

        # calculate running estimates
        if self.training:
            if MTE == 'sample':
                from torch.distributions.normal import Normal
                Distri1 = Normal(self.meta_mean1, self.meta_var1)
                Distri2 = Normal(self.meta_mean2, self.meta_var2)
                sample1 = Distri1.sample([input.size(0), ])
                sample2 = Distri2.sample([input.size(0), ])
                lam = np.random.beta(1., 1.)
                inputmix1 = lam * sample1 + (1-lam) * input
                inputmix2 = lam * sample2 + (1-lam) * input

                mean1 = inputmix1.mean(dim=0)
                var1 = inputmix1.var(dim=0, unbiased=False)
                mean2 = inputmix2.mean(dim=0)
                var2 = inputmix2.var(dim=0, unbiased=False)

                output1 = (inputmix1 - mean1[None, :]) / (torch.sqrt(var1[None, :] + self.eps))
                output2 = (inputmix2 - mean2[None, :]) / (torch.sqrt(var2[None, :] + self.eps))
                if self.affine:
                    output1 = output1 * self.weight[None, :] + self.bias[None, :]
                    output2 = output2 * self.weight[None, :] + self.bias[None, :]
                return [output1, output2]

            else:
                mean = input.mean(dim=0)
                # use biased var in train
                var = input.var(dim=0, unbiased=False)
                n = input.numel() / input.size(1)

                with torch.no_grad():
                    running_mean = exponential_average_factor * mean \
                                   + (1 - exponential_average_factor) * self.running_mean
                    # update running_var with unbiased var
                    running_var = exponential_average_factor * var * n / (n - 1) \
                                  + (1 - exponential_average_factor) * self.running_var
                    self.running_mean.copy_(running_mean)
                    self.running_var.copy_(running_var)
                    if save_index == 1:
                        self.meta_mean1.copy_(mean)
                        self.meta_var1.copy_(var)
                    elif save_index == 2:
                        self.meta_mean2.copy_(mean)
                        self.meta_var2.copy_(var)

        else:
            mean = self.running_mean
            var = self.running_var

        input = (input - mean[None, :]) / (torch.sqrt(var[None, :] + self.eps))
        if self.affine:
            input = input * self.weight[None, :] + self.bias[None, :]

        return input, [mean, var]


def get_linear_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, last_epoch=-1):
    """
    Create a schedule with a learning rate that decreases linearly from the initial lr set in the optimizer to 0, after
    a warmup period during which it increases linearly from 0 to the initial lr set in the optimizer.
    Args:
        optimizer ([`~torch.optim.Optimizer`]):
            The optimizer for which to schedule the learning rate.
        num_warmup_steps (`int`):
            The number of steps for the warmup phase.
        num_training_steps (`int`):
            The total number of training steps.
        last_epoch (`int`, *optional*, defaults to -1):
            The index of the last epoch when resuming training.
    Return:
        `torch.optim.lr_scheduler.LambdaLR` with the appropriate schedule.
    """

    def lr_lambda(current_step: int):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        return max(
            0.0, float(num_training_steps - current_step) / float(max(1, num_training_steps - num_warmup_steps))
        )

    return LambdaLR(optimizer, lr_lambda, last_epoch)



def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def  set_optimizer(optimizer_type, lr_scheduler_type, param, learning_rate, epoch):

    if optimizer_type == 'adam':
            optimizer = torch.optim.Adam(param, lr=learning_rate, weight_decay=0.0001)
    elif optimizer_type == 'sgd':
            optimizer = SGD(param, lr=learning_rate, momentum=0.9, dampening=0, weight_decay=0.0001, nesterov='store_true')
    elif optimizer_type == 'adamw':
            optimizer = torch.optim.AdamW(param, lr=learning_rate, weight_decay=0.0001)
    assert lr_scheduler_type in ['plateau', 'multistep', 'linear_warmup']
    if lr_scheduler_type == 'plateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, 'min',factor=0.1, patience=3,threshold=0.001, threshold_mode='abs', min_lr=1e-6,verbose=True)
    elif lr_scheduler_type == 'multistep':
        scheduler = lr_scheduler.MultiStepLR(optimizer, [10,20,40])
    elif lr_scheduler_type == 'linear_warmup':
        scheduler = get_linear_schedule_with_warmup(optimizer, 50, epoch)
    return optimizer, scheduler

def setup_logging(log_path):
    # Set up logging
    if os.path.exists(log_path) is False:
        os.mkdir(log_path)
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    log_file = log_path + '/result_log.txt'
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

class ShowProcess():
    i = 0
    max_steps = 0 
    max_arrow = 50
    infoDone = 'done'

    def __init__(self, max_steps, infoDone = 'Done'):
        self.max_steps = max_steps
        self.i = 0
        self.infoDone = infoDone

    def show_process(self, i=None):
        if i is not None:
            self.i = i
        else:
            self.i += 1
        num_arrow = int(self.i * self.max_arrow / self.max_steps) 
        num_line = self.max_arrow - num_arrow 
        percent = self.i * 100.0 / self.max_steps 
        process_bar = '[' + '>' * num_arrow + '-' * num_line + ']'\
                      + '%.2f' % percent + '%' + '\r'
        sys.stdout.write(process_bar) 
        sys.stdout.flush()
        if self.i >= self.max_steps:
            self.close()

    def close(self):
        print('')
        print(self.infoDone)
        self.i = 0