import torch
from torch import nn
from models import SAINT
from utils import mixup_class
from sup_loss import SupConLoss
from augmentations import mixup_data
from data_loader import OpenMLDataLoader
from models.Graph_Label import GraphLabelPropagation
from models.PCELoss import PCELoss
from data_openml import data_prep_openml,task_dset_ids,DataSetCatCon
import argparse
from torch.utils.data import DataLoader
import torch.optim as optim
from utils import count_parameters, classification_scores, mean_sq_error
from augmentations import embed_data_mask
from augmentations import add_noise
import torch.nn.functional as F
import datetime


import os
import numpy as np
parser = argparse.ArgumentParser()

parser.add_argument('--dset_id', required=True, type=int)
parser.add_argument('--vision_dset', action = 'store_true')
parser.add_argument('--task', required=True, type=str,choices = ['binary','multiclass','regression'])
parser.add_argument('--cont_embeddings', default='MLP', type=str,choices = ['MLP','Noemb','pos_singleMLP'])
parser.add_argument('--embedding_size', default=32, type=int)
parser.add_argument('--transformer_depth', default=6, type=int)
parser.add_argument('--attention_heads', default=8, type=int)
parser.add_argument('--attention_dropout', default=0.1, type=float)
parser.add_argument('--ff_dropout', default=0.1, type=float)
parser.add_argument('--attentiontype', default='colrow', type=str,choices = ['col','colrow','row','justmlp','attn','attnmlp'])

parser.add_argument('--optimizer', default='AdamW', type=str,choices = ['AdamW','Adam','SGD'])
parser.add_argument('--scheduler', default='cosine', type=str,choices = ['cosine','linear'])

parser.add_argument('--lr', default=0.0001, type=float)
parser.add_argument('--epochs', default=100, type=int)
parser.add_argument('--batchsize', default=256, type=int)
parser.add_argument('--savemodelroot', default='./bestmodels', type=str)
parser.add_argument('--run_name', default='testrun', type=str)
parser.add_argument('--set_seed', default= 23 , type=int)
parser.add_argument('--dset_seed', default= 23 , type=int)
parser.add_argument('--active_log', action = 'store_true')

parser.add_argument('--pretrain', action = 'store_true')
parser.add_argument('--pretrain_epochs', default=50, type=int)
parser.add_argument('--pt_tasks', default=['contrastive','denoising'], type=str,nargs='*',choices = ['contrastive','contrastive_sim','denoising'])
parser.add_argument('--pt_aug', default=[], type=str,nargs='*',choices = ['mixup','cutmix'])
parser.add_argument('--pt_aug_lam', default=0.1, type=float)
parser.add_argument('--mixup_lam', default=0.3, type=float)

parser.add_argument('--train_mask_prob', default=0, type=float)
parser.add_argument('--mask_prob', default=0, type=float)

parser.add_argument('--ssl_avail_y', default= 0, type=int)
parser.add_argument('--pt_projhead_style', default='diff', type=str,choices = ['diff','same','nohead'])
parser.add_argument('--nce_temp', default=0.7, type=float)
parser.add_argument('--split', default=0.0625, type=float)
parser.add_argument('--lam0', default=0.5, type=float)
parser.add_argument('--T', default=1, type=float)
parser.add_argument('--lam1', default=10, type=float)
parser.add_argument('--lam2', default=1, type=float)
parser.add_argument('--lam3', default=10, type=float)
parser.add_argument('--final_mlp_style', default='sep', type=str,choices = ['common','sep'])

opt = parser.parse_args()
modelsave_path = os.path.join(os.getcwd(),opt.savemodelroot,opt.task,str(opt.dset_id),opt.run_name)
if opt.task == 'regression':
    opt.dtask = 'reg'
else:
    opt.dtask = 'clf'

starttime = datetime.datetime.now()
device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")
print(f"Device is {device}.")

torch.manual_seed(opt.set_seed)
os.makedirs(modelsave_path, exist_ok=True)
np.random.seed(opt.set_seed)
torch.cuda.manual_seed(23)
torch.backends.cudnn.deterministic =True
os.environ['PYTHONHASHSEED'] =str(opt.set_seed)
labeled_ratio = opt.split




if opt.active_log:
    import wandb
    if opt.pretrain:
        wandb.init(project="saint_v2_all", group =opt.run_name ,name = f'pretrain_{opt.task}_{str(opt.attentiontype)}_{str(opt.dset_id)}_{str(opt.set_seed)}')
    else:
        if opt.task=='multiclass':
            wandb.init(project="saint_v2_all_kamal", group =opt.run_name ,name = f'{opt.task}_{str(opt.attentiontype)}_{str(opt.dset_id)}_{str(opt.set_seed)}')
        else:
            wandb.init(project="saint_v2_all", group =opt.run_name ,name = f'{opt.task}_{str(opt.attentiontype)}_{str(opt.dset_id)}_{str(opt.set_seed)}')
   

print('Downloading and processing the dataset, it might take some time.')
cat_dims, cat_idxs, con_idxs, X_train, y_train, X_valid, y_valid, X_test, y_test, train_mean, train_std = data_prep_openml(opt.dset_id, opt.dset_seed,opt.task, datasplit=[.8, .1, .1])
continuous_mean_std = np.array([train_mean,train_std]).astype(np.float32) 

##### Setting some hyperparams based on inputs and dataset
_,nfeat = X_train['data'].shape
if nfeat > 100:
    opt.embedding_size = min(8,opt.embedding_size)
    opt.batchsize = min(64, opt.batchsize)
if opt.attentiontype != 'col':
    opt.transformer_depth = 1
    opt.attention_heads = min(4,opt.attention_heads)
    opt.attention_dropout = 0.8
    opt.embedding_size = min(32,opt.embedding_size)
    opt.ff_dropout = 0.8

print(nfeat,opt.batchsize)
print(opt)

if opt.active_log:
    wandb.config.update(opt)
train_ds = DataSetCatCon(X_train, y_train, cat_idxs,opt.dtask,labeled_ratio,continuous_mean_std,method='semisupervised')


trainloader = OpenMLDataLoader(train_ds, batch_size=opt.batchsize, shuffle=True,num_workers=4)


Utrainloader = OpenMLDataLoader(train_ds, batch_size=4*opt.batchsize, shuffle=True,num_workers=4)

valid_ds = DataSetCatCon(X_valid, y_valid, cat_idxs,opt.dtask, 1,continuous_mean_std,method ='testing')
validloader = DataLoader(valid_ds, batch_size=opt.batchsize, shuffle=False,num_workers=4)

test_ds = DataSetCatCon(X_test, y_test, cat_idxs,opt.dtask, 1,continuous_mean_std,method ='testing')
testloader = DataLoader(test_ds, batch_size=opt.batchsize, shuffle=False,num_workers=4)
if opt.task == 'regression':
    y_dim = 1
else:
    y_dim = len(np.unique(y_train['data'][:,0]))

cat_dims = np.append(np.array([1]),np.array(cat_dims)).astype(int) #Appending 1 for CLS token, this is later used to generate embeddings.

pseudo_labeler = GraphLabelPropagation(k=50,max_iter=20,alpha=0.99)


model = SAINT(
categories = tuple(cat_dims), 
num_continuous = len(con_idxs),                
dim = opt.embedding_size,                           
dim_out = 1,                       
depth = opt.transformer_depth,                       
heads = opt.attention_heads,                         
attn_dropout = opt.attention_dropout,             
ff_dropout = opt.ff_dropout,                  
mlp_hidden_mults = (4, 2),       
cont_embeddings = opt.cont_embeddings,
attentiontype = opt.attentiontype,
final_mlp_style = opt.final_mlp_style,
y_dim = y_dim
)
vision_dset = opt.vision_dset
print(y_dim)
if y_dim == 2 and opt.task == 'binary':
    # opt.task = 'binary'
    criterion = nn.CrossEntropyLoss().to(device)
elif y_dim > 2 and  opt.task == 'multiclass':
    # opt.task = 'multiclass'
    criterion = nn.CrossEntropyLoss().to(device)
elif opt.task == 'regression':
    criterion = nn.MSELoss().to(device)
else:
    raise'case not written yet'
Entropy = nn.CrossEntropyLoss(reduction='none').to(device)
    
    
model.to(device)
alpha = 1.0
lam = np.random.beta(alpha, alpha, size=1)
lam = torch.tensor(lam)
contrast_loss = SupConLoss()



if opt.pretrain:
    from pretraining import SAINT_pretrain
    model = SAINT_pretrain(model, cat_idxs,X_train,y_train, continuous_mean_std, opt,device)

## Choosing the optimizer

if opt.optimizer == 'SGD':
    optimizer = optim.SGD(model.parameters(), lr=opt.lr,
                          momentum=0.9, weight_decay=5e-4)
    from utils import get_scheduler
    scheduler = get_scheduler(opt, optimizer)
elif opt.optimizer == 'Adam':
    optimizer = optim.Adam(model.parameters(),lr=opt.lr,eps=1e-4)
    optimizer2 = optim.AdamW(model.parameters(),lr=opt.lr*0.1,eps=1e-4) 
elif opt.optimizer == 'AdamW':
    optimizer = optim.AdamW(model.parameters(),lr=opt.lr,weight_decay=5e-4)
    optimizer2 = optim.AdamW(model.parameters(),lr=opt.lr*0.1,weight_decay=5e-4) 

best_valid_auroc = 0
best_valid_accuracy = 0
best_test_auroc = 0
best_test_accuracy = 0
best_valid_rmse = 100000
print('Training begins now.')
x_categ, x_cont,u_categ, u_cont, y_gts, u_gts, cat_mask, con_mask, u_cat_mask, u_con_mask,utarget, uweight = None,None,None,None,None,None,None,None,None,None,None,None
u = None
epoch_start = 40
every_f_epoch = 5
threshold = 0.95
for epoch in range(opt.epochs):
    model.train()
    running_loss = 0.0
    if (epoch<epoch_start):
        for i, data in enumerate(trainloader, 0):
            optimizer.zero_grad()
            # x_categ is the the categorical data, x_cont has continuous data, y_gts has ground truth ys. cat_mask is an array of ones same shape as x_categ and an additional column(corresponding to CLS token) set to 0s. con_mask is an array of ones same shape as x_cont.
            if type(data) == torch.Tensor:
                x = data
                x = x.to(device)
                u = data
                u = u.to(device)
            elif len(data) == 10:    
                x_categ, x_cont,u_categ, u_cont, y_gts, u_gts, cat_mask, con_mask, u_cat_mask, u_con_mask = data[0].to(device), data[1].to(device),data[2].to(device),data[3].to(device),data[4].to(device),data[5].to(device), data[6].to(device),data[7].to(device),data[8].to(device),data[9].to(device)
                
               
            elif len(data) == 12:
                x_categ, x_cont,u_categ, u_cont, y_gts, u_gts, cat_mask, con_mask, u_cat_mask, u_con_mask,utarget, uweight = data[0].to(device), data[1].to(device),data[2].to(device),data[3].to(device),data[4].to(device),data[5].to(device), data[6].to(device),data[7].to(device),data[8].to(device),data[9].to(device),data[10].to(device),data[11].to(device)
                
                
            else:
                x_categ, x_cont,y_gts,cat_mask, con_mask = data[0].to(device),data[1].to(device),data[2].to(device),data[3].to(device),data[4].to(device)
                

            #x_categ, x_cont, y_gts, cat_mask, con_mask = data[0].to(device), data[1].to(device),data[2].to(device),data[3].to(device),data[4].to(device)
            _ , x_categ_enc_2, x_cont_enc_2 = embed_data_mask(x_categ, x_cont, cat_mask, con_mask,model,vision_dset)
            _ , x_categ_enc, x_cont_enc = embed_data_mask(x_categ, x_cont, cat_mask, con_mask,model,vision_dset)

            # We are converting the data to embeddings in the next step
            x_categ_enc_2, x_cont_enc_2, lambda_indices = mixup_data(x_categ_enc_2, x_cont_enc_2,y_gts,lam=lam)
            x = model.transformer(x_categ_enc, x_cont_enc)
            x_mixup = model.transformer(x_categ_enc_2.float(), x_cont_enc_2.float())
            if u_categ is not None:
                _ , u_categ_enc, u_cont_enc = embed_data_mask(u_categ, u_cont, u_cat_mask, u_con_mask,model,vision_dset) 
                u = model.transformer(u_categ_enc, u_cont_enc)
            # Projection to metric space
            z = x
            z_mixup = x_mixup
            z = (z / z.norm(dim=-1, keepdim=True)).flatten(1,2)
            z_mixup = (z_mixup / z_mixup.norm(dim=-1, keepdim=True)).flatten(1,2)
            z_proj = model.pt_mlp(z)
            z_proj_mixup = model.pt_mlp(z_mixup)
            # select only the representations corresponding to CLS token and apply mlp on it in the next step to get the predictions.
            y_reps = x[:,0,:]
            y_outs = model.mlpfory(y_reps)
            if opt.task == 'regression':
                loss = criterion(y_outs,y_gts) 
            else:
                #CE loss
                CEloss = criterion(y_outs,y_gts.squeeze())
                #Self-supervised loss
                xxx = z_proj_mixup.clone()
                z_proj_mixup[lambda_indices] = xxx # - order indices
                z_proj_mixup = torch.unsqueeze(z_proj_mixup, dim=1)
                z_proj = torch.unsqueeze(z_proj, dim=1)
                zs = torch.cat([z_proj, z_proj_mixup], dim=1)
                SSloss = 0
                #SSloss = contrast_loss(zs, y_gts)
                loss = CEloss + SSloss
            loss.backward()
            optimizer.step()
            if opt.optimizer == 'SGD':
                scheduler.step()
            running_loss += loss.item()
        # print(running_loss)
        if opt.active_log:
            wandb.log({'epoch': epoch ,'train_epoch_loss': running_loss, 
            'loss': loss.item()
            })
        if epoch%5==0:
                model.eval()
                with torch.no_grad():
                    if opt.task in ['binary','multiclass']:
                        accuracy, auroc = classification_scores(model, validloader, device, opt.task,vision_dset)
                        test_accuracy, test_auroc = classification_scores(model, testloader, device, opt.task,vision_dset)

                        print('[EPOCH %d] VALID ACCURACY: %.3f, VALID AUROC: %.3f' %
                            (epoch + 1, accuracy,auroc ))
                        print('[EPOCH %d] TEST ACCURACY: %.3f, TEST AUROC: %.3f' %
                            (epoch + 1, test_accuracy,test_auroc ))
                        if opt.active_log:
                            wandb.log({'valid_accuracy': accuracy ,'valid_auroc': auroc })     
                            wandb.log({'test_accuracy': test_accuracy ,'test_auroc': test_auroc })  
                        if opt.task =='multiclass':
                            if accuracy > best_valid_accuracy:
                                best_valid_accuracy = accuracy
                                best_test_auroc = test_auroc
                                best_test_accuracy = test_accuracy
                                torch.save(model.state_dict(),'%s/bestmodel.pth' % (modelsave_path))
                        else:
                            if accuracy > best_valid_accuracy:
                                best_valid_accuracy = accuracy
                            # if auroc > best_valid_auroc:
                            #     best_valid_auroc = auroc
                                best_test_auroc = test_auroc
                                best_test_accuracy = test_accuracy               
                                torch.save(model.state_dict(),'%s/bestmodel.pth' % (modelsave_path))

                    else:
                        valid_rmse = mean_sq_error(model, validloader, device,vision_dset)    
                        test_rmse = mean_sq_error(model, testloader, device,vision_dset)  
                        print('[EPOCH %d] VALID RMSE: %.3f' %
                            (epoch + 1, valid_rmse ))
                        print('[EPOCH %d] TEST RMSE: %.3f' %
                            (epoch + 1, test_rmse ))
                        if opt.active_log:
                            wandb.log({'valid_rmse': valid_rmse ,'test_rmse': test_rmse })     
                        if valid_rmse < best_valid_rmse:
                            best_valid_rmse = valid_rmse
                            best_test_rmse = test_rmse
                            torch.save(model.state_dict(),'%s/bestmodel.pth' % (modelsave_path))
                model.train()
    
    
    
    
    
    
    else: 
        
        if (epoch ==epoch_start): 
            #acc = trainloader.update_pseudo_labels(model, pseudo_labeler, device)
            acc2 = Utrainloader.update_pseudo_labels(model, pseudo_labeler, device)
            print('Train Epoch: {} PL-Accuracy: {:.6f}'.format(epoch, acc2 * 100))
        if (epoch > epoch_start) and (((epoch - epoch_start) % every_f_epoch) == 0):
            #acc = trainloader.update_pseudo_labels(model, pseudo_labeler, device)
            acc2 = Utrainloader.update_pseudo_labels(model, pseudo_labeler, device)
            print('Train Epoch: {} PL-Accuracy: {:.6f}'.format(epoch, acc2 * 100))
     
        dataloader_iterator = iter(Utrainloader)
        correct_idx = []    
        for i, data in enumerate(trainloader, 0):
            optimizer2.zero_grad()           
            try:
                while True:
                    data2 = next(dataloader_iterator)
            except StopIteration:
                pass
            x_categ, x_cont,u_categ, u_cont, y_gts, u_gts, cat_mask, con_mask, u_cat_mask, u_con_mask,utarget, uweight = data[0].to(device), data[1].to(device),data[2].to(device),data[3].to(device),data[4].to(device),data[5].to(device), data[6].to(device),data[7].to(device),data[8].to(device),data[9].to(device),data[10].to(device),data[11].to(device)
            
            x_categ2, x_cont2,u_categ2, u_cont2, y_gts2, u_gts2, cat_mask2, con_mask2, u_cat_mask2, u_con_mask2, utarget2, uweight2 = data2[0].to(device), data2[1].to(device),data2[2].to(device),data2[3].to(device),data2[4].to(device),data2[5].to(device), data2[6].to(device),data2[7].to(device),data2[8].to(device),data2[9].to(device),data2[10].to(device),data2[11].to(device)
            
            #x_categ, x_cont, y_gts, cat_mask, con_mask = data[0].to(device), data[1].to(device),data[2].to(device),data[3].to(device),data[4].to(device)
            _ , x_categ_enc_2, x_cont_enc_2 = embed_data_mask(x_categ, x_cont, cat_mask, con_mask,model,vision_dset)
            _ , x_categ_enc, x_cont_enc = embed_data_mask(x_categ, x_cont, cat_mask, con_mask,model,vision_dset)

            # We are converting the data to embeddings in the next step
            x_categ_enc_2, x_cont_enc_2, lambda_indices = mixup_data(x_categ_enc_2, x_cont_enc_2,y_gts,lam=lam)
            x = model.transformer(x_categ_enc, x_cont_enc)
            x_mixup = model.transformer(x_categ_enc_2.float(), x_cont_enc_2.float())
            if u_categ2 is not None:
                _ , u_categ_enc, u_cont_enc = embed_data_mask(u_categ2, u_cont2, u_cat_mask2, u_con_mask2,model,vision_dset)
                _ , u_categ_enc_2, u_cont_enc_2 = embed_data_mask(u_categ2, u_cont2, u_cat_mask2, u_con_mask2,model,vision_dset)
                u = model.transformer(u_categ_enc, u_cont_enc)
                
                
            # Projection to metric space
            z = x
            z_mixup = x_mixup
            z = (z / z.norm(dim=-1, keepdim=True)).flatten(1,2)
            uu = u
            uu = (uu / uu.norm(dim=-1, keepdim=True)).flatten(1,2)
            z_mixup = (z_mixup / z_mixup.norm(dim=-1, keepdim=True)).flatten(1,2)          
            z_proj = model.pt_mlp(z)
            u_proj = model.pt_mlp(uu)
            z_proj_mixup = model.pt_mlp(z_mixup)
            
            # select only the representations corresponding to CLS token and apply mlp on it in the next step to get the predictions.
            y_reps = x[:,0,:]
            y_outs = model.mlpfory(y_reps)
            u_reps = u[:,0,:]
            yu_outs = model.mlpfory(u_reps)
            #calculate the pslabel for projection head
            pseudo_label = torch.softmax(yu_outs.detach()/opt.T, dim=-1)
            max_probs, ulabel = torch.max(pseudo_label, dim=-1)
            mask = max_probs.ge(0.95).float()
            index = torch.nonzero(mask)
        
            correct_num = (ulabel[index] == u_gts2.squeeze()[index]).float().sum()
            if len(index)!=0:
                #print(correct_num.item()/len(index))
                correct_idx.append(correct_num.item()/len(index))
            correct_idx.append(0)
            #supervised contrastive for unlabeled data
            u_categ_enc_2, u_cont_enc_2, lambda_indices_u = mixup_data(u_categ_enc_2, u_cont_enc_2,ulabel,lam=lam)
            u_mixup = model.transformer(u_categ_enc_2.float(), u_cont_enc_2.float())
            u_mixup = (u_mixup / u_mixup.norm(dim=-1, keepdim=True)).flatten(1,2)
            u_proj_mixup = model.pt_mlp(u_mixup)
            
            
            if opt.task == 'regression':
                loss = criterion(y_outs,y_gts) 
            else:
                #CE loss
                CEloss = criterion(y_outs,y_gts.squeeze())
                #Self-supervised loss
                xxx = z_proj_mixup.clone()
                z_proj_mixup[lambda_indices] = xxx # - order indices
                z_proj_mixup = torch.unsqueeze(z_proj_mixup, dim=1)
                z_proj = torch.unsqueeze(z_proj, dim=1)
                zs = torch.cat([z_proj, z_proj_mixup], dim=1)
                SSloss = contrast_loss(zs, y_gts)
                #Pseudo-label CE loss
                utarget2 = utarget2.squeeze().long()
                pceloss = PCELoss(yu_outs, utarget2,0, device,criterion)     
                #pceloss = criterion(yu_outs,u_gts.squeeze())
                #print(CEloss,pceloss)
                #Pseudo-label Contrastive Loss
                uuu = u_proj_mixup.clone()
                u_proj_mixup[lambda_indices_u] = uuu # - order indices
                u_proj_mixup = torch.unsqueeze(u_proj_mixup, dim=1)
                u_proj = torch.unsqueeze(u_proj, dim=1)
                us = torch.cat([u_proj, u_proj_mixup], dim=1)
                pssloss = contrast_loss(us, ulabel,mask=None, weights=mask)
                loss = CEloss + SSloss + pceloss+ pssloss
                #loss = CEloss
            loss.backward()
            optimizer2.step()
            running_loss += loss.item()
        if opt.active_log:
            wandb.log({'epoch': epoch ,'train_epoch_loss': running_loss, 
            'loss': loss.item()
            })
        if (epoch > epoch_start) and (((epoch - epoch_start) % every_f_epoch) == 0):    
            acc2 = np.mean(correct_idx)/opt.batchsize
            print('Train Epoch: {} CL-PL-Accuracy: {:.6f}'.format(epoch, acc2 * 100))    
        if epoch%5==0:
                model.eval()
                with torch.no_grad():
                    if opt.task in ['binary','multiclass']:
                        accuracy, auroc = classification_scores(model, validloader, device, opt.task,vision_dset)
                        test_accuracy, test_auroc = classification_scores(model, testloader, device, opt.task,vision_dset)

                        print('[EPOCH %d] VALID ACCURACY: %.3f, VALID AUROC: %.3f' %
                            (epoch + 1, accuracy,auroc ))
                        print('[EPOCH %d] TEST ACCURACY: %.3f, TEST AUROC: %.3f' %
                            (epoch + 1, test_accuracy,test_auroc ))
                        if opt.active_log:
                            wandb.log({'valid_accuracy': accuracy ,'valid_auroc': auroc })     
                            wandb.log({'test_accuracy': test_accuracy ,'test_auroc': test_auroc })  
                        if opt.task =='multiclass':
                            if accuracy > best_valid_accuracy:
                                best_valid_accuracy = accuracy
                                best_test_auroc = test_auroc
                                best_test_accuracy = test_accuracy
                                torch.save(model.state_dict(),'%s/bestmodel.pth' % (modelsave_path))
                        else:
                            #if accuracy > best_valid_accuracy:
                                #best_valid_accuracy = accuracy
                            if auroc > best_valid_auroc:
                                best_valid_auroc = auroc
                                best_test_auroc = test_auroc
                                best_test_accuracy = test_accuracy               
                                torch.save(model.state_dict(),'%s/bestmodel.pth' % (modelsave_path))

                    else:
                        valid_rmse = mean_sq_error(model, validloader, device,vision_dset)    
                        test_rmse = mean_sq_error(model, testloader, device,vision_dset)  
                        print('[EPOCH %d] VALID RMSE: %.3f' %
                            (epoch + 1, valid_rmse ))
                        print('[EPOCH %d] TEST RMSE: %.3f' %
                            (epoch + 1, test_rmse ))
                        if opt.active_log:
                            wandb.log({'valid_rmse': valid_rmse ,'test_rmse': test_rmse })     
                        if valid_rmse < best_valid_rmse:
                            best_valid_rmse = valid_rmse
                            best_test_rmse = test_rmse
                            torch.save(model.state_dict(),'%s/bestmodel.pth' % (modelsave_path))    
            
        
        


total_parameters = count_parameters(model)
print('TOTAL NUMBER OF PARAMS: %d' %(total_parameters))
if opt.task =='binary':
    print('AUROC on best model:  %.3f' %(best_test_auroc))
elif opt.task =='multiclass':
    print('Accuracy on best model:  %.3f' %(best_test_accuracy))
else:
    print('RMSE on best model:  %.3f' %(best_test_rmse))
    
endtime = datetime.datetime.now()
#打印
print((endtime - starttime).seconds)   

  

if opt.active_log:
    if opt.task == 'regression':
        wandb.log({'total_parameters': total_parameters, 'test_rmse_bestep':best_test_rmse , 
        'cat_dims':len(cat_idxs) , 'con_dims':len(con_idxs) })        
    else:
        wandb.log({'total_parameters': total_parameters, 'test_auroc_bestep':best_test_auroc , 
        'test_accuracy_bestep':best_test_accuracy,'cat_dims':len(cat_idxs) , 'con_dims':len(con_idxs) })
