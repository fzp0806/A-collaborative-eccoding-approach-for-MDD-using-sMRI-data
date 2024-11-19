from __future__ import print_function, division
from pathlib import Path
from this import d
import torch.distributions as dist
import torch
from torch.autograd import grad, Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import SGD, Adam, lr_scheduler
import numpy as np
import torchvision
import matplotlib.pyplot as plt
import time
import os
import copy
import shutil
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score, confusion_matrix, f1_score
import logging
import scipy
import scipy.io as scio
import csv
import seaborn as sns
import pandas as pd
import sys
from load_data import  Set_Dataloader
from collections import Counter
# from Resformer import resnet50_v1b
# from conformer import Conformer
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from Utils import *
import argparse
# from parallel import DataParallelModel, DataParallelCriterion
import torch.backends.cudnn as cudnn
# from torch.nn.parallel import gather
from sklearn.metrics import roc_curve
from Reg_loss import global_reg_loss, local_reg_loss, GM_loss
from torch.utils.data import ConcatDataset
def Meta_Training(localmodel, data, label, criterion=None):
    # Federated training process  
    localmodel.train()
    torch.cuda.empty_cache()
    
    output = localmodel(data.float())
    pre = output[2]
    out_fc = output[5]
    loss_cls = criterion(pre, label.long())
    return pre, loss_cls, out_fc

    
def Test(model, local_test, criterion,logger):
    model.eval()

    with torch.no_grad():
        classify_corrects = 0
        total_loss = 0.0
        total = 0.0
        scores = []
        local_scores = []
        labels = []
        process_bar = ShowProcess(len(local_test), 'OK')
        for i, (data, label) in enumerate(local_test):
            process_bar.show_process()
            time.sleep(0.01)
            data = data.cuda()
            label = label.cuda()
            output = model(data.float()) 
            pre = output[2]
            cls_loss = criterion(pre, label.long())
            labels.append(label.data.cpu().numpy())
            scores.append(pre.data.cpu().numpy())
            total += 1
            total_loss += cls_loss.data.cpu().numpy()
        del data, model, label
        
        labels = np.hstack(labels)
        scores = np.vstack(scores)  
        predicted = np.argmax(scores, axis=1)
        try:
            test_auc = roc_auc_score(labels, scores[:,1])
        except ValueError:
            test_auc = 0
        
        classify_corrects = (predicted == labels).sum().item()     
        test_acc = classify_corrects / len(labels)
        logger.info('test_correct:{:.3f} '.format((predicted == labels).sum().item()))
        F1_scores = f1_score(labels, predicted, average='weighted')
        
        confusion = confusion_matrix(labels, predicted)
        if len(confusion.shape) > 1:
            # True positive; Sen
            Sen = confusion[0][0] / sum(confusion[0])
            # True Negtive; Spe
            Spe = confusion[1][1] / sum(confusion[1])
        else:
            Sen = confusion[0][0] / sum(confusion[0])
            Spe = confusion[0][1] / sum(confusion[0])

    return total_loss/total, test_auc, test_acc, Sen, Spe, F1_scores


def main_worker(args):
    logger = logging.getLogger()
    # torch.cuda.set_device(args.local_rank)
    # dist.init_process_group(backend='nccl')
    device = torch.device("cuda", args.local_rank)
    # os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    # os.environ["CUDA_VISIBLE_DEVICES"] = '0,1,2,3'
    Data_site = ['S1', 'S2', 'S3', 'S5', 'S6', 'S7', 'S8', 'S9', 'S10', 'S11', 'S12', 'S13', 'S14', 'S15', 'S16', 'S17', 'S18', 'S20', 'S21', 'S22', 'S24', 'S25']
    Client_site = [0,1,2,3,4]
    # Data_site = ['AMU', 'FMMU', 'shanxi', 'Xiangya_li', 'Xiangya_pu']
    # Data_site = ['AMU', 'COBRE', 'FAMUWANG', 'PUH6I', 'UCLA', 'Xiangya']
    Num_part = len(Data_site)    


    torch.cuda.empty_cache()
    CE = nn.CrossEntropyLoss()

    if args.gpu:
        cudnn.benchmark = True

    for site in Data_site:
        site_path = args.result_path / site
        if os.path.exists(site_path) == False:
            os.mkdir(site_path)

        
    # Start training with five fold validation strategy
    for fold in range(5):
        # Model in different sites
        Models = {}
        gm_model = {}
        gm_optimizer = {}
        cl_gcn = {}
       

        local_optimizer = {}
        local_scheduler = {}
        cl_gcn_optimizer = {}
        # Dataloader in different sites

        test_loader = {}
        batch_train_data = {}
        batch_train_label = {}

        #Something need to record in iterations
        total = {}
        Best_Acc = {}
        Best_global_Acc = {}

        all_train_data = []
        all_test_data = []
        for site in Data_site:
            train_data, test_data = Set_Dataloader(fold, site, args.data_dir)
            all_train_data.append(train_data)
            all_test_data.append(test_data)
            
        # for i in range(len(Data_site)):
        train_data_1 = all_train_data[17]
        train_data_2 = ConcatDataset([all_train_data[0]+all_train_data[1]+all_train_data[2]+all_train_data[4]+all_train_data[5] +all_train_data[11]])
        train_data_3 = ConcatDataset([all_train_data[6]+all_train_data[7]+all_train_data[8]+all_train_data[9]+all_train_data[10]])
        train_data_4 = ConcatDataset([all_train_data[3]+all_train_data[12]+all_train_data[13]+all_train_data[14]+all_train_data[15]+all_train_data[16]])
        train_data_5 =ConcatDataset([all_train_data[18]+all_train_data[19]+all_train_data[20]+all_train_data[21]])
        
        print('train_files 1:', len(train_data_1))
        print('train_files 2:', len(train_data_2))
        print('train_files 3:', len(train_data_3))
        print('train_files 4:', len(train_data_4))
        print('train_files 5:', len(train_data_5))
        train_loader_1 = torch.utils.data.DataLoader(train_data_1, batch_size=args.batch_size, num_workers=args.num_workers,pin_memory=True, shuffle=True)
        train_loader_2 = torch.utils.data.DataLoader(train_data_2, batch_size=args.batch_size, num_workers=args.num_workers,pin_memory=True, shuffle=True)
        train_loader_3 = torch.utils.data.DataLoader(train_data_3, batch_size=args.batch_size, num_workers=args.num_workers,pin_memory=True, shuffle=True)
        train_loader_4 = torch.utils.data.DataLoader(train_data_4, batch_size=args.batch_size, num_workers=args.num_workers,pin_memory=True, shuffle=True)
        train_loader_5 = torch.utils.data.DataLoader(train_data_5, batch_size=args.batch_size, num_workers=args.num_workers,pin_memory=True, shuffle=True)
        train_loader=[train_loader_1, train_loader_2, train_loader_3, train_loader_4, train_loader_5]
        
        test_data_1 = all_test_data[17]
        test_data_2 = ConcatDataset([all_test_data[0]+all_test_data[1]+all_test_data[2]+all_test_data[4]+all_test_data[5] +all_test_data[11]])
        test_data_3 = ConcatDataset([all_test_data[6]+all_test_data[7]+all_test_data[8]+all_test_data[9]+all_test_data[10]])
        test_data_4 = ConcatDataset([all_test_data[3]+all_test_data[12]+all_test_data[13]+all_test_data[14]+all_test_data[15]+all_test_data[16]])
        test_data_5 =ConcatDataset([all_test_data[18]+all_test_data[19]+all_test_data[20]+all_test_data[21]])
        test_loader_1 = torch.utils.data.DataLoader(test_data_1, batch_size=1)
        test_loader_2 = torch.utils.data.DataLoader(test_data_2, batch_size=1)
        test_loader_3 = torch.utils.data.DataLoader(test_data_3, batch_size=1)
        test_loader_4 = torch.utils.data.DataLoader(test_data_4, batch_size=1)
        test_loader_5 = torch.utils.data.DataLoader(test_data_5, batch_size=1)
        test_loader=[test_loader_1, test_loader_2, test_loader_3, test_loader_4, test_loader_5]
       

        logger.info('****************** Start Local Training **********************')
        # print('****************** Start Meta Training **********************')
    

        global_model = resnet50_v1b()
        global_model = global_model.cuda()
        global_Gcn = GCN(2048,32,32,0.2)
        global_Gcn = global_Gcn.cuda()
        gcn_optimizer, _ = set_optimizer('adam', 'plateau', global_Gcn.parameters(), args.lr, args.epoch)
      
            
        for site in range(5):   
            Models[site] = resnet50_v1b()
            Models[site] = Models[site].cuda()
            local_optimizer[site], local_scheduler[site] = set_optimizer('adamw', 'plateau', Models[site].parameters(), args.lr, args.epoch)
            gm_model[site] = Classifier(2048,64,2).cuda()
            gm_optimizer[site], _ = set_optimizer('adam', 'plateau', gm_model[site].parameters(), args.lr, args.epoch)
            cl_gcn[site] = GCN(2048,32,32,0.2)
            cl_gcn[site] = cl_gcn[site].cuda()
            cl_gcn_optimizer[site], _ = set_optimizer('adam', 'plateau', cl_gcn[site].parameters(), args.lr, args.epoch)
            

            Best_Acc[site] = 0.0
            Best_global_Acc[site] = 0.0
        # Strat Meta-LocalTraining
        
        for epoch in range(args.epoch):
            epoch_start_time = time.time()
            logger.info('Meta-Training  epoch:{}/{}'.format(epoch+1, args.epoch))
            #Initial records for each epoch
            Labels = {}
            preds = {}
            total = {}

            test_Labels = {}
            test_preds = {}
          
            local_train_loss={}
            local_test_loss={}
            for site in range(5):   
                total[site] = 0.0
                Labels[site] = []
                test_Labels[site] = []
                preds[site] = []
                test_preds[site] = []
                local_train_loss[site] = 0.0
                local_test_loss[site] = 0.0
            # Load Meta-Batch Data from each site
            
            process = 'train_scaffold'
            torch.cuda.empty_cache()
           
            for site in range(5):      
                train_iter = iter(train_loader[site])    
                for iters in range(len(train_iter)): 
                    batch_train_data[site], batch_train_label[site] = next(train_iter)
                    batch_train_data[site], batch_train_label[site] = batch_train_data[site].cuda(), batch_train_label[site].cuda()
                    output, loss_cls, out_fc = Meta_Training(Models[site], batch_train_data[site], batch_train_label[site],  criterion=CE, device=device, phase=process)

                    if epoch == 0:
                        local_optimizer[site].zero_grad()
                        loss = loss_cls
                        loss.backward()
                        local_optimizer[site].step()
                    else:
                        l_reg_loss = local_reg_loss(Models, cl_gcn[site], train_loader, Client_site, site)
                        l_GM_loss = GM_loss(Models, gm_model[site], train_loader, Client_site, site)
                        
                        loss = loss_cls + l_reg_loss + 1e-5*l_GM_loss
                        loss.requires_grad_(True)
                    
                        local_optimizer[site].zero_grad()
                        cl_gcn_optimizer[site].zero_grad()
                        loss.backward()
                        cl_gcn_optimizer[site].step()
                        local_optimizer[site].step()

                        GM_pre = gm_model[site](out_fc)
                        GM_cls = CE(GM_pre, batch_train_label[site].long())
                        GM_cls.requires_grad_(True)
                        gm_optimizer[site].zero_grad()
                        GM_cls.backward()
                        gm_optimizer[site].step()
                        logger.info('processing epoch:{} Site:{} l_loss:{:.5f} l_reg_loss:{:.5f} l_GM_loss:{:.5f} GM_cls:{:.5f}'.format(epoch+1, site, loss, l_reg_loss, l_GM_loss, GM_cls))
                        
                    local_train_loss[site] += loss.item() 
                    output_np = output.data.cpu().numpy()                    
                    label_np = batch_train_label[site].data.cpu().numpy()                   
                    Labels[site].append(label_np)
                    preds[site].append(output_np)
                    
                    torch.cuda.empty_cache()    
                    total[site] += 1    
                    
                    
            process = 'federated_aggregate'
            
            for name, para in global_model.named_parameters():
                temp = torch.zeros_like(global_model.state_dict()[name])
                for site in range(5):
                    _weight = torch.as_tensor(1 / len(Client_site)).cuda()
                    temp += _weight * Models[site].state_dict()[name]
                global_model.state_dict()[name].data.copy_(temp)
            
            for site in range(5):   
                Train_labels = np.hstack(Labels[site])
                Train_total_sub = len(Train_labels)
                Train_pred = np.vstack(preds[site])
                Train_predicted = np.argmax(Train_pred, axis=1)
                Train_correct= (Train_predicted == Train_labels).sum().item()     
                Train_epoch_cls_loss = local_train_loss[site] / total[site]
                
                Train_Acc = Train_correct / Train_total_sub        

                Scores = np.vstack(preds[site])          
                try:
                    Auc = roc_auc_score(Train_labels, Scores[:,1])
                    
                except ValueError:
                    logger.info('Site:{} process:{} Epoch:{} has only One class sample'.format(site, process, epoch+1))
                
                confusion = confusion_matrix(Train_labels, Train_predicted)
                
                # True positive; Sen
                if len(confusion.shape) > 1:
                    Sen = confusion[0][0] / sum(confusion[0])
                    # True Negtive; Spe
                    Spe = confusion[1][1] / sum(confusion[1])
                else:
                    Sen = 1
                    Spe = 1
                
                logger.info('Local-Training Site:{} process:{} Epoch:{} Cls_loss:{:.5f} Acc:{:.5f} AUC: {:.5f} Sen:{:.5f} Spe:{:.5f} '.format(site, process, epoch+1, Train_epoch_cls_loss, Train_Acc, Auc, Sen, Spe))
          
            process = 'test'
            for site in range(5):   
                logger.info('Test processing epoch:{} Site:{}'.format(epoch+1, site))

                cls_loss, Auc, Acc, Sen, Spe, F1_scores = Test(Models[site], test_loader[site], CE, fold, epoch, args.result_path, site, logger)
                local_scheduler[site].step(cls_loss)
                logger.info('Test process Epoch:{}/{}  Loss_cls: {:.5f}  AUC_Test: {:.5f} ACC_Test: {:.5f}  Sen_Test: {:.5f} Spe_Test: {:.5f} F1_scores: {:.5f}'.format(epoch+1,args.epoch, cls_loss, Auc, Acc, Sen, Spe, F1_scores))
                torch.cuda.empty_cache()
                global_cls_loss, global_Auc, global_Acc, global_Sen, global_Spe, global_F1_scores = Test(global_model, test_loader[site], CE, fold, epoch, args.result_path, site, logger)
                logger.info('Test process Epoch:{}/{}  Global Loss_cls: {:.5f} Global AUC_Test: {:.5f} Global ACC_Test: {:.5f} Global Sen_Test: {:.5f} Global Spe_Test: {:.5f} Global F1_scores: {:.5f}'.format(epoch+1,args.epoch, global_cls_loss, global_Auc, global_Acc, global_Sen, global_Spe, global_F1_scores))
                # save better models
                site_path = args.result_path / ('client_'+str(site+1))
                if os.path.exists(site_path) is False:
                    os.mkdir(site_path)
                model_savepath = args.result_path / ('models')
                if os.path.exists(model_savepath) is False:
                    os.mkdir(model_savepath)
              
                if Best_Acc[site] <= Acc:
                    Best_Acc[site] = Acc

                    torch.save(Models[site], model_savepath/('federated-model_fold_' + str(fold+1) +'_Best.pth'))
                    filename = site_path / ('fold_' + str(fold+1)+'_Best_Acc_result.csv')
                    with open(filename, 'a+', newline='') as ff:
                        csv_write = csv.writer(ff)         
                        header = ['Site', 'Acc', 'Auc','Sen', 'Spe', 'F1_scores', 'Epoch']
                        csv_write.writerow(header)
                        data_row = [site, Best_Acc[site], Auc, Sen, Spe, F1_scores, epoch+1]
                        csv_write.writerow(data_row) 
                if Best_global_Acc[site] <= global_Acc:
                    Best_global_Acc[site] = global_Acc
                    filename = site_path / ('fold_' + str(fold+1)+'_Best_global_Acc_result.csv')
                    with open(filename, 'a+', newline='') as ff:
                        csv_write = csv.writer(ff)         
                        header = ['Site', 'Acc', 'Auc','Sen', 'Spe', 'F1_scores', 'Epoch']
                        csv_write.writerow(header)
                        data_row = [site, Best_global_Acc[site], global_Auc, global_Sen, global_Spe, global_F1_scores, epoch+1]
                        csv_write.writerow(data_row) 
            
if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', default=Path('/HOME/scz0abb/run/Fan/Fed-DG-3d/Data/MDD/Rest-Meta-MDD'), help='Data directory')
    parser.add_argument('--epoch', type=int, default=30, help='Max. number of epochs')
    parser.add_argument('--num_workers', type=int, default=0, help='No. of parallel threads')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size')
    parser.add_argument('--step_loss', type=int, default=20, help='Decrease learning rate after how many epochs')
    parser.add_argument('--lr', type=float, default=1e-3, help='Initial learning rate')
    parser.add_argument('--lr_mode', default='step', help='Learning rate policy, step or poly')
    parser.add_argument('--result_path', default=Path('/HOME/scz0abb/run/Fan/Fed-DG-3d/Data/Rest-Meta-MDD-Results/Fed_Reg'), help='Directory to save the results')
    parser.add_argument('--gpu', default=True, type=lambda x: (str(x).lower() == 'true'), help='Run on CPU or GPU. If TRUE, then GPU.')
    parser.add_argument('--log_path', default='/HOME/scz0abb/run/Fan/Fed-DG-3d/Data/Rest-Meta-MDD-Results/Fed_Reg')
    parser.add_argument("--local_rank", default=0)
    
    args = parser.parse_args()
    log_name = 'resnet-attn-scaffold.txt'
    setup_logging(args.log_path, log_name)  
    if not os.path.exists(args.result_path):
        os.mkdir(args.result_path)
    shutil.copy('/HOME/scz0abb/run/Fan/Fed-DG-3d/Data/main/Meta_Fed_GM_Reg.py', args.result_path / 'train.py')
    shutil.copy('/HOME/scz0abb/run/Fan/Fed-DG-3d/Data/main/Resformer.py', args.result_path / ('model.py'))
    main_worker(args) 
