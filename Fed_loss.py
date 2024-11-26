import torch
import torch.nn.functional as f
from Utils import tensor_diag
from collections import defaultdict
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
def calculate_importance(model, dataloader):
    importance = defaultdict(list)
    # model.cuda()
    for n, p in model.named_parameters():
        for i in range(2):
            if model.state_dict()[n].dim() == 5:
                importance[n].append(p.mean([1, 2, 3, 4]).clone().detach().view(-1).fill_(0))
            elif model.state_dict()[n].dim() == 4:
                importance[n].append(p.mean([1, 2, 3]).clone().detach().view(-1).fill_(0))
            elif model.state_dict()[n].dim() == 3:
                importance[n].append(p.mean([1, 2]).clone().detach().view(-1).fill_(0))
            elif model.state_dict()[n].dim() == 2:
                importance[n].append(p.mean(dim=1).clone().detach().view(-1).fill_(0))
            else:
                importance[n].append(p.clone().detach().view(-1).fill_(0))

    model.eval()
    for _, (inputs, targets) in enumerate(dataloader):
        inputs = inputs.cuda()
        targets = targets.cuda()
        outputs = model(inputs.float())
        pre = outputs[2]
        pre = pre.mean(dim=0)

        for i in range(2):
            model.zero_grad()
            pre[i].backward(retain_graph=True)
            for n, p in model.named_parameters():
                if p.grad is not None:
                    if p.dim() == 5:#conv
                        node_imp = p.grad.mean([1,2,3,4]).view(-1)
                    elif p.dim() == 4:#linear
                        node_imp = p.grad.mean([1,2,3]).view(-1)
                    elif p.dim() == 3:#linear
                        node_imp = p.grad.mean([1,2]).view(-1)
                    elif p.dim() == 2:#linear
                        node_imp = p.grad.mean(dim=1).view(-1)
                    else:
                        node_imp = p.grad.view(-1)
                    importance[n][i] += node_imp / len(dataloader)
    return importance

def calculate_gm_importance(model, gm, dataloader):
    importance = defaultdict(list)
    CE = nn.CrossEntropyLoss()
    for n, p in gm.named_parameters():
        for i in range(2):
            if gm.state_dict()[n].dim() == 5:
                importance[n].append(p.mean([1, 2, 3, 4]).clone().detach().view(-1).fill_(0))
            elif gm.state_dict()[n].dim() == 4:
                importance[n].append(p.mean([1, 2, 3]).clone().detach().view(-1).fill_(0))
            elif gm.state_dict()[n].dim() == 3:
                importance[n].append(p.mean([1, 2]).clone().detach().view(-1).fill_(0))
            elif gm.state_dict()[n].dim() == 2:
                importance[n].append(p.mean(dim=1).clone().detach().view(-1).fill_(0))
            else:
                importance[n].append(p.clone().detach().view(-1).fill_(0))

    model.eval()
    num = 0
    loss_cls = 0.0
    for _, (inputs, targets) in enumerate(dataloader):
        inputs = inputs.cuda()
        targets = targets.cuda()
        outputs = model(inputs.float())
        model_output = outputs[5]

        output_l = gm(torch.tensor(model_output))
        output = output_l.mean(dim=0)
        num = num + 1
        for i in range(2):
            model.zero_grad()
            output[i].backward(retain_graph=True)
            for n, p in gm.named_parameters():
                if p.grad is not None:
                    if p.dim() == 5:#conv
                        node_imp = p.grad.mean([1,2,3,4]).view(-1)
                    elif p.dim() == 4:#linear
                        node_imp = p.grad.mean([1,2,3]).view(-1)
                    elif p.dim() == 3:#linear
                        node_imp = p.grad.mean([1,2]).view(-1)
                    elif p.dim() == 2:#linear
                        node_imp = p.grad.mean(dim=1).view(-1)
                    else:
                        node_imp = p.grad.view(-1)
                    importance[n][i] += node_imp / len(dataloader)
    return importance

def svd(imp):
        left_eigen_vec, eigen_val, right_eigen_vec = torch.svd_lowrank(torch.stack(imp).t(), q=1)
        return left_eigen_vec, eigen_val, right_eigen_vec.t()

def com_prototype(Models, sites):
    prototype = []
    for site in sites:
        weight = Models[site].fc.weight.detach()#c*f
        normalized = f.normalize(weight, p=2, dim=1)
        prototype.append((normalized).unsqueeze(0))#1*c*f
    prototype = torch.cat(prototype)#t*c*f
    return prototype

def com_site_prototype(Models, sites, local_site):
    prototype = []
    for site in range(5):
        if site != local_site:
            weight = Models[site].fc.weight.detach()#c*f
            normalized = f.normalize(weight, p=2, dim=1)
            prototype.append((normalized).unsqueeze(0))#1*c*f
    prototype = torch.cat(prototype)
    return prototype

def comp_GW(left, eigen, right, w):
    rec = torch.matmul(torch.matmul(left, tensor_diag(eigen, eigen.device)), right).cuda()
    w=w
    if w.dim() == 7:#task+conv-dim+class
        rec = rec.unsqueeze(2).unsqueeze(3).unsqueeze(4).unsqueeze(5).expand(w.shape)
        gw = (rec * w)
        _gw = gw.mean(dim=[1, 2, 3, 4, 5])
    elif w.dim() == 6:#task+conv-dim+class
        rec = rec.unsqueeze(2).unsqueeze(3).unsqueeze(4).expand(w.shape)
        gw = (rec * w)
        _gw = gw.mean(dim=[1, 2, 3, 4])
    elif w.dim() == 5:#task+conv-dim+class
        rec = rec.unsqueeze(2).unsqueeze(3).expand(w.shape)
        gw = (rec * w)
        _gw = gw.mean(dim=[1, 2, 3])
    elif w.dim() == 4:#task+linear-dim+class
        rec = rec.unsqueeze(2).expand(w.shape)
        gw = (rec * w)
        _gw = gw.mean(dim=[1, 2])
    else:
        _gw = (rec * w).mean(dim=1)
    return _gw

def GM_loss(Models, gm, train_loader, sites, local_site):
    regularization_terms = Cal_site_GM(Models, gm, train_loader, sites, local_site)
    importance = calculate_gm_importance(Models[local_site], gm, train_loader[local_site])
    GM = []
    grad_cossim = []
    grad_mse = []
    for n, p in gm.named_parameters():
        left_eigen_local, eigen_local, right_eigen_local = svd(importance[n])
        local_rec = torch.matmul(torch.matmul(left_eigen_local, tensor_diag(eigen_local, eigen_local.device)), right_eigen_local).cuda()
        # for site in sites:
        for site in range(5):
            if site != local_site:
                left_e_site = torch.cat(regularization_terms[site]['left_eigen_vec'][n], dim=0)
                eigen_site = torch.cat(regularization_terms[site]['eigen_val'][n], dim=0)
                right_site = torch.cat(regularization_terms[site]['right_eigen_vec'][n], dim=0)
                site_rec = torch.matmul(torch.matmul(left_e_site, tensor_diag(eigen_site, eigen_site.device)), right_site).cuda()
                site_rec = torch.squeeze(site_rec)
                
                if len(local_rec.shape) > 1:
                    _cossim = F.cosine_similarity(local_rec, site_rec, dim=1).mean()
                else:
                    _cossim = F.cosine_similarity(local_rec, site_rec, dim=0)
                grad_cossim.append(_cossim)
    grad_cossim = torch.stack(grad_cossim)
    gm_loss = (1.0 - grad_cossim).mean()    

    return gm_loss

def local_fvd_loss(Models, hypermodel, train_loader, sites, local_site):
    GW = torch.zeros([len(sites)-1, 2]).cuda()
    regularization_terms = Cal_site_Gwstar(Models, train_loader, sites, local_site)
    GWstar = torch.cat(regularization_terms['GWstar'])#t*c
    for n, p in Models[local_site].named_parameters():
        for site in range(5):
            if site != local_site:
                left_e_v = torch.cat(regularization_terms[site]['left_eigen_vec'][n],dim=0)
                eigen = torch.cat(regularization_terms[site]['eigen_val'][n],dim=0)
                right_v = torch.cat(regularization_terms[site]['right_eigen_vec'][n],dim=0)
                w = p.unsqueeze(-1).expand([len(sites)-1] + list(p.shape) + [2])  # t*p*c
                GW += comp_GW(left_e_v, eigen, right_v, w)
    prototype = com_site_prototype(Models, sites, local_site)#(t-1)*f*c
    affinity = torch.matmul(prototype, prototype.permute(0, 2, 1)).cuda()#t*c*c
    decom = hypermodel(affinity, prototype)
    nom_decom = f.normalize(decom, p=2, dim=-1)#(t-1)*c*h
    precision = torch.matmul(nom_decom, nom_decom.permute(0,2,1))
    identity = torch.eye(precision.size(-1)).expand(*precision.size()).cuda()
    cross_term = torch.matmul(torch.matmul(GW.unsqueeze(1), precision), GWstar.unsqueeze(-1))#(t*1*c) matmul (t*c*c) matmul (t*c*1)
    square_term = torch.matmul(torch.matmul(GW.unsqueeze(1), precision), GW.unsqueeze(-1))
    square_star_term = torch.matmul(torch.matmul(GWstar.unsqueeze(1), precision), GWstar.unsqueeze(-1))
    all_fvd_loss = (square_term - 2*cross_term + square_star_term).sum(dim=[1,2]) 
    all_fvd_loss += (1/((precision+1e-4*identity).det())).log()
    fvd_loss = task_fvd_loss.sum()# sum task
    return fvd_loss

def global_fvd_loss(Models, global_model, hypermodel, train_loader, sites):
    GW = torch.zeros(len(sites), 2).cuda()
    regularization_terms = Cal_global_Gwstar(Models, train_loader, sites)
    GWstar = torch.cat(regularization_terms['GWstar'])#t*c
    for n, p in global_model.named_parameters():
        for site in sites:
            left_e_v = torch.cat(regularization_terms[site]['left_eigen_vec'][n],dim=0)
            eigen = torch.cat(regularization_terms[site]['eigen_val'][n],dim=0)
            right_v = torch.cat(regularization_terms[site]['right_eigen_vec'][n],dim=0)
            w = p.unsqueeze(-1).expand([len(sites)] + list(p.shape) + [2])  # t*p*c
            GW += comp_GW(left_e_v, eigen, right_v, w)
    prototype = com_prototype(Models, sites)
    affinity = torch.matmul(prototype, prototype.permute(0, 2, 1)).cuda()#t*c*c
    decom = hypermodel(affinity, prototype)
    nom_decom = f.normalize(decom, p=2, dim=-1)
    precision = torch.matmul(nom_decom, nom_decom.permute(0,2,1)).cuda()
    identity = torch.eye(precision.size(-1)).expand(*precision.size()).cuda()
    cross_term = torch.matmul(torch.matmul(GW.unsqueeze(1), precision), GWstar.unsqueeze(-1))#(t*1*c) matmul (t*c*c) matmul (t*c*1)
    square_term = torch.matmul(torch.matmul(GW.unsqueeze(1), precision), GW.unsqueeze(-1))
    square_star_term = torch.matmul(torch.matmul(GWstar.unsqueeze(1), precision), GWstar.unsqueeze(-1))
    all_fvd_loss = (square_term - 2*cross_term + square_star_term).sum(dim=[1,2]) 
    all_fvd_loss += (1/((precision+1e-4*identity).det())).log()
    fvd_loss = all_fvd_loss.sum()# sum task
    return fvd_loss

def Cal_global_Gwstar(Models, train_loader, sites):
    regularization_terms = {}
    regularization_terms['GWstar'] = []
    for site in sites:
        regularization_terms[site] = {'left_eigen_vec':defaultdict(list),'eigen_val':defaultdict(list),'right_eigen_vec':defaultdict(list)}
        importance = calculate_importance(Models[site], train_loader[site])
        num = 0
        GWstar = torch.zeros(1, 2).cuda()
        for n, p in Models[site].named_parameters():
            left_eigen_vec, eigen_val, right_eigen_vec = svd(importance[n])
            num += np.prod(left_eigen_vec.shape) + np.prod(eigen_val.shape) + np.prod(right_eigen_vec.shape)
            regularization_terms[site]['left_eigen_vec'][n].append(left_eigen_vec.unsqueeze(0))
            regularization_terms[site]['eigen_val'][n].append(eigen_val.unsqueeze(0))
            regularization_terms[site]['right_eigen_vec'][n].append(right_eigen_vec.unsqueeze(0))
            wstar = p.clone().detach().unsqueeze(-1).expand([1]+list(p.shape)+[2])
            GWstar += comp_GW(left_eigen_vec.unsqueeze(0), eigen_val.unsqueeze(0), right_eigen_vec.unsqueeze(0), wstar)
        regularization_terms['GWstar'].append(GWstar)
    return regularization_terms

def Cal_site_Gwstar(Models, train_loader, sites, local_site):
    regularization_terms = {}

    regularization_terms['GWstar'] = []
    for site in range(5):
        if site != local_site:
            regularization_terms[site]={'left_eigen_vec':defaultdict(list),'eigen_val':defaultdict(list),'right_eigen_vec':defaultdict(list)}
            GWstar = torch.zeros(1, 2).cuda()
            num = 0
            importance = calculate_importance(Models[site], train_loader[site])
            for n, p in Models[site].named_parameters():
                left_eigen_vec, eigen_val, right_eigen_vec = svd(importance[n])
                num += np.prod(left_eigen_vec.shape) + np.prod(eigen_val.shape) + np.prod(right_eigen_vec.shape)
                regularization_terms[site]['left_eigen_vec'][n].append(left_eigen_vec.unsqueeze(0))
                regularization_terms[site]['eigen_val'][n].append(eigen_val.unsqueeze(0))
                regularization_terms[site]['right_eigen_vec'][n].append(right_eigen_vec.unsqueeze(0))
                wstar = p.clone().detach().unsqueeze(-1).expand([1]+list(p.shape)+[2])
                GWstar += comp_GW(left_eigen_vec.unsqueeze(0), eigen_val.unsqueeze(0), right_eigen_vec.unsqueeze(0), wstar)
            regularization_terms['GWstar'].append(GWstar)
    return regularization_terms

def Cal_site_GM(Models, gm, train_loader, sites, local_site):
    regularization_terms = {}
    # for site in sites:
    for site in range(5):
        if site != local_site:
            regularization_terms[site]={'left_eigen_vec':defaultdict(list),'eigen_val':defaultdict(list),'right_eigen_vec':defaultdict(list)}
            importance = calculate_gm_importance(Models[site], gm, train_loader[site])
            for n, p in gm.named_parameters():
 
                left_eigen_vec, eigen_val, right_eigen_vec = svd(importance[n])
                regularization_terms[site]['left_eigen_vec'][n].append(left_eigen_vec.unsqueeze(0))
                regularization_terms[site]['eigen_val'][n].append(eigen_val.unsqueeze(0))
                regularization_terms[site]['right_eigen_vec'][n].append(right_eigen_vec.unsqueeze(0))

    return regularization_terms
