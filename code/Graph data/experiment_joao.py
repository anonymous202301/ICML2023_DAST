import torch
import torch.nn.functional as F
from torch.optim import Adam
from tu_dataset import DataLoader
import numpy as np
from torch.utils.data import RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
import logging
import time

from utils import print_weights, accuracy, AverageMeter
from tqdm import tqdm
from sup_loss import SupConLoss


logger = logging.getLogger(__name__)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
contrast_loss = SupConLoss()
local_rank = int(-1)

def experiment(train_dataset, unsup_train_dataset, test_dataset, model_func, epochs, batch_size, lr, weight_decay,
                dataset_name=None, aug_mode='uniform', aug_ratio=0.2, suffix=0, gamma_joao=0.1):
    model = model_func(unsup_train_dataset).to(device)
    print_weights(model)
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    train_dataset.set_aug_mode('sample')
    train_dataset.set_aug_ratio(aug_ratio)
    unsup_train_dataset.set_aug_mode('sample')
    unsup_train_dataset.set_aug_ratio(aug_ratio)
    aug_prob = np.ones(25) / 25
    train_dataset.set_aug_prob(aug_prob)
    unsup_train_dataset.set_aug_prob(aug_prob)
    
    #train_loader = DataLoader(train_dataset, batch_size, shuffle=True, num_workers=16)
    #unsup_train_loader = DataLoader(unsup_train_dataset, batch_size, shuffle=True, num_workers=16)
    
    train_sampler = RandomSampler
    
    labeled_trainloader = DataLoader(
        train_dataset,
        sampler=train_sampler(train_dataset),
        batch_size=batch_size,
        num_workers=16)

    unlabeled_trainloader = DataLoader(
        unsup_train_dataset,
        sampler=train_sampler(unsup_train_dataset),
        batch_size = batch_size*7,
        num_workers=16)
    
    test_loader = DataLoader(
        test_dataset,
        sampler=SequentialSampler(test_dataset),
        batch_size=batch_size,
        num_workers=16)

    no_decay = ['bias', 'bn']
    grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(
            nd in n for nd in no_decay)], 'weight_decay': weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(
            nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]  
    optimizer = Adam(grouped_parameters, lr=lr, weight_decay=weight_decay)
    eval_step = 100
    logger.info("***** Running training *****")
    labeled_iter = iter(labeled_trainloader)
    unlabeled_iter = iter(unlabeled_trainloader)
    
    test_accs = []
    end = time.time()
    
    # for epoch in tqdm(range(1, epochs+1)):
    for epoch in range(epochs):
        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        losses_x = AverageMeter()
        losses_u = AverageMeter()
        mask_probs = AverageMeter()
        Closses = AverageMeter()
        model.train()
        p_bar = tqdm(range(eval_step),
                         disable=local_rank not in [-1, 0])
        for batch_idx in range(eval_step):
            try:
                data,data1,data2 = labeled_iter.next()
            except:
                labeled_iter = iter(labeled_trainloader)
                data,data1,data2 = labeled_iter.next()

            try:
                undata, undata2,_ = unlabeled_iter.next()
            except:
                unlabeled_iter = iter(unlabeled_trainloader)
                undata, undata2,_ = unlabeled_iter.next()
    
            data_time.update(time.time() - end)    
            data, data1,data2 = data.to(device), data1.to(device),data2.to(device)
            #print(data,data1,data2)
            logit, proj = model.forward_graphcl(data)
            Lx =  F.cross_entropy(logit, data.y.long().view(-1), reduction='mean')
            undata, undata2 = undata.to(device), undata2.to(device)
            logit_u,proj_u = model.forward_graphcl(undata)
            logit_u_2,proj_u_2 = model.forward_graphcl(undata2)
            pseudo_label = torch.softmax(logit_u.detach(), dim=-1)
            max_probs, targets_u = torch.max(pseudo_label, dim=-1)
            mask = max_probs.ge(0.9).float()
            
            Lu = (F.cross_entropy(logit_u_2, targets_u,
                                  reduction='none') * mask).mean()
            
            proj_u = F.normalize(proj_u, dim=1)
            proj_u_2 = F.normalize(proj_u_2, dim=1)
            
            zu = torch.cat([proj_u.unsqueeze(1), proj_u_2.unsqueeze(1)], dim=1)
            CLu = contrast_loss(zu,targets_u)
            #CLu = 0
            loss = Lx +  Lu + CLu
            loss.backward()
            optimizer.step()
            
 
            losses.update(loss.item())
            losses_x.update(Lx.item())
            losses_u.update(Lu.item())
            Closses.update(CLu)
            model.zero_grad()
            batch_time.update(time.time() - end)
            end = time.time()
            p_bar.set_description("Train Epoch: {epoch}/{epochs:4}. Iter: {batch:4}/{iter:4}. LR: {lr:.4f}. Data: {data:.3f}s. Batch: {bt:.3f}s. Loss: {loss:.4f}.  Loss_u: {loss_u:.4f}. Loss_x: {loss_x:.4f}. Closs: {Closs:.2f}. ".format(
                    epoch=epoch + 1,
                    epochs=epochs,
                    batch=batch_idx + 1,
                    iter=eval_step,
                    lr=lr,
                    data=data_time.avg,
                    bt=batch_time.avg,
                    loss=losses.avg,
                    loss_u=losses_u.avg,
                    loss_x=losses_x.avg,
                    Closs=Closses.avg,
                    ))
            p_bar.update() 
           
        test_acc = test(test_loader, model, epoch) 
        print(test_acc)
            
            
            
            
        """        
        pretrain_loss, aug_prob = train(loader, model, optimizer, device, gamma_joao)
        print(pretrain_loss, aug_prob)
        loader.dataset.set_aug_prob(aug_prob)

        if epoch % 20 == 0:
            weight_path = './weights_joao/' + dataset_name + '_' + str(lr) + '_' + str(epoch) + '_' + str(gamma_joao) + '_' + str(suffix)  + '.pt'
            torch.save(model.state_dict(), weight_path)

    """        
            
def test(test_loader, model, epoch):
    correct = 0
    with torch.no_grad():
        for batch_idx, test_data in enumerate(test_loader):
            model.eval()
            data,data1,data2 = test_data
            data = data.to(device)
            targets = data.y.long()
            outputs,proj = model.forward_graphcl(data)
            pred = F.log_softmax(outputs, dim=-1)
            pred = pred.max(1)[1]            
            correct += pred.eq(data.y.view(-1)).sum().item()
            print(correct)
    return correct / len(test_loader.dataset)    
            
            
          
            

def num_graphs(data):
    if data.batch is not None:
        return data.num_graphs
    else:
        return data.x.size(0)


def train(loader, model, optimizer, device, gamma_joao):
    model.train()
    total_loss = 0
    for _, data1, data2 in loader:
        # print(data1, data2)
        optimizer.zero_grad()
        data1 = data1.to(device)
        data2 = data2.to(device)
        out1 = model.forward_graphcl(data1)
        out2 = model.forward_graphcl(data2)
        loss = model.loss_graphcl(out1, out2)
        loss.backward()
        total_loss += loss.item() * num_graphs(data1)
        optimizer.step()

    aug_prob = joao(loader, model, gamma_joao)
    return total_loss/len(loader.dataset), aug_prob


def joao(loader, model, gamma_joao):
    aug_prob = loader.dataset.aug_prob
    # calculate augmentation loss
    loss_aug = np.zeros(25)
    for n in range(25):
        _aug_prob = np.zeros(25)
        _aug_prob[n] = 1
        loader.dataset.set_aug_prob(_aug_prob)

        count, count_stop = 0, len(loader.dataset)//(loader.batch_size*10)+1 # for efficiency, we only use around 10% of data to estimate the loss
        with torch.no_grad():
            for _, data1, data2 in loader:
                data1 = data1.to(device)
                data2 = data2.to(device)
                out1 = model.forward_graphcl(data1)
                out2 = model.forward_graphcl(data2)
                loss = model.loss_graphcl(out1, out2)
                loss_aug[n] += loss.item() * num_graphs(data1)
                count += 1
                if count == count_stop:
                    break
        loss_aug[n] /= (count*loader.batch_size)

    # view selection, projected gradient descent, reference: https://arxiv.org/abs/1906.03563
    beta = 1
    gamma = gamma_joao

    b = aug_prob + beta * (loss_aug - gamma * (aug_prob - 1/25))
    mu_min, mu_max = b.min()-1/25, b.max()-1/25
    mu = (mu_min + mu_max) / 2

    # bisection method
    while abs(np.maximum(b-mu, 0).sum() - 1) > 1e-2:
        if np.maximum(b-mu, 0).sum() > 1:
            mu_min = mu
        else:
            mu_max = mu
        mu = (mu_min + mu_max) / 2

    aug_prob = np.maximum(b-mu, 0)
    aug_prob /= aug_prob.sum()

    return aug_prob

