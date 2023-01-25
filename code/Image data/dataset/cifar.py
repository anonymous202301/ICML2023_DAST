import logging
import math
from torch.utils.data import DataLoader
import numpy as np
import torch
from PIL import Image
from torchvision import datasets
from torchvision import transforms
from torch.utils.data.dataloader import default_collate
from torch.utils.data.sampler import SubsetRandomSampler
from .randaugment import RandAugmentMC
import copy
import torch.nn.functional as F

logger = logging.getLogger(__name__)

cifar10_mean = (0.4914, 0.4822, 0.4465)
cifar10_std = (0.2471, 0.2435, 0.2616)
cifar100_mean = (0.5071, 0.4867, 0.4408)
cifar100_std = (0.2675, 0.2565, 0.2761)
normal_mean = (0.5, 0.5, 0.5)
normal_std = (0.5, 0.5, 0.5)



class BaseDataLoader(DataLoader):
    """
    Base class for all data loaders
    """
    def __init__(self, dataset, batch_size, shuffle, validation_split= 0.0, num_workers = 1, collate_fn=default_collate,drop_last=True):
        self.validation_split = validation_split
        self.shuffle = shuffle
        self.batch_idx = 0
        self.n_samples = len(dataset)

        self.sampler, self.valid_sampler = self._split_sampler(self.validation_split)

        self.init_kwargs = {
            'dataset': dataset,
            'batch_size': batch_size,
            'shuffle': self.shuffle,
            'collate_fn': collate_fn,
            'num_workers': num_workers
        }
        super().__init__(sampler=self.sampler, **self.init_kwargs)

    def _split_sampler(self, split):
        if split == 0.0:
            return None, None

        idx_full = np.arange(self.n_samples)

        np.random.seed(0)
        np.random.shuffle(idx_full)

        if isinstance(split, int):
            assert split > 0
            assert split < self.n_samples, "validation set size is configured to be larger than entire dataset."
            len_valid = split
        else:
            len_valid = int(self.n_samples * split)

        valid_idx = idx_full[0:len_valid]
        train_idx = np.delete(idx_full, np.arange(0, len_valid))

        train_sampler = SubsetRandomSampler(train_idx)
        valid_sampler = SubsetRandomSampler(valid_idx)

        # turn off shuffle option which is mutually exclusive with sampler
        self.shuffle = False
        self.n_samples = len(train_idx)

        return train_sampler, valid_sampler

    def split_validation(self):
        if self.valid_sampler is None:
            return None
        else:
            return DataLoader(sampler=self.valid_sampler, **self.init_kwargs)

def get_cifar10(args, root):
    transform_labeled = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(size=32,
                              padding=int(32*0.125),
                              padding_mode='reflect'),
        transforms.ToTensor(),
        transforms.Normalize(mean=cifar10_mean, std=cifar10_std)
    ])
    transform_val = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=cifar10_mean, std=cifar10_std)
    ])
    base_dataset = datasets.CIFAR10(root, train=True, download=True)

    train_labeled_idxs, train_unlabeled_idxs = x_u_split(
        args, base_dataset.targets)
    
    """
    train_labeled_dataset = CIFAR10SSL(
        root, train_labeled_idxs, train=True,
        transform=TransformLabeled(mean=cifar10_mean, std=cifar10_std))
    
    train_unlabeled_dataset = CIFAR10SSL(
        root, train_unlabeled_idxs, train=True,
        transform=TransformFixMatch(mean=cifar10_mean, std=cifar10_std))
    """
    
    train_labeled_dataset = CIFAR10SSL(
        root, train_labeled_idxs, train=True,
        transform=transform_labeled)
    train_unlabeled_dataset = CIFAR10_Unlabel(
        root, train_labeled_idxs, train_unlabeled_idxs, train=True,
        transform=transform_labeled)
   
    test_dataset = datasets.CIFAR10(
        root, train=False, transform=transform_val, download=False)

    return train_labeled_dataset, train_unlabeled_dataset, test_dataset


def get_cifar100(args, root):

    transform_labeled = transforms.Compose([
        #transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(size=32,
                              padding=int(32*0.125),
                              padding_mode='reflect'),
        transforms.ToTensor(),
        transforms.Normalize(mean=cifar100_mean, std=cifar100_std)])

    transform_val = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=cifar100_mean, std=cifar100_std)])

    base_dataset = datasets.CIFAR100(
        root, train=True, download=True)

    train_labeled_idxs, train_unlabeled_idxs = x_u_split(
        args, base_dataset.targets)
    
    
    train_labeled_dataset = CIFAR100SSL(
        root, train_labeled_idxs, train=True,
        transform=transform_val)
    
    train_unlabeled_dataset = CIFAR10_Unlabel(
        root,train_labeled_idxs, train_unlabeled_idxs, train=True,
        transform=transform_labeled)
    
    """
    train_labeled_dataset = CIFAR100SSL(
        root, train_labeled_idxs, train=True,
        transform=transform_labeled)

    train_unlabeled_dataset = CIFAR100SSL(
        root, train_unlabeled_idxs, train=True,
        transform=TransformFixMatch(mean=cifar100_mean, std=cifar100_std))
    """
    test_dataset = datasets.CIFAR100(
        root, train=False, transform=transform_val, download=False)

    return train_labeled_dataset, train_unlabeled_dataset, test_dataset


def x_u_split(args, labels):
    label_per_class = args.num_labeled // args.num_classes
    labels = np.array(labels)
    labeled_idx = []
    #unlabeled data: all data (https://github.com/kekmodel/FixMatch-pytorch/issues/10)
    unlabeled_idx = np.array(range(len(labels)))
    for i in range(args.num_classes):
        idx = np.where(labels == i)[0]
        idx = np.random.choice(idx, label_per_class, False)
        labeled_idx.extend(idx)
    labeled_idx = np.array(labeled_idx)
    assert len(labeled_idx) == args.num_labeled

    if args.expand_labels or args.num_labeled < args.batch_size:
        num_expand_x = math.ceil(
            args.batch_size * args.eval_step / args.num_labeled)
        labeled_idx = np.hstack([labeled_idx for _ in range(num_expand_x)])
    np.random.shuffle(labeled_idx)
    return labeled_idx, unlabeled_idx


class TransformFixMatch(object):
    def __init__(self, mean, std):
        self.weak = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(size=32,
                                  padding=int(32*0.125),
                                  padding_mode='reflect')])
        self.strong = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(size=32,
                                  padding=int(32*0.125),
                                  padding_mode='reflect'),
            RandAugmentMC(n=2, m=10)])
        self.normalize = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)])

    def __call__(self, x):
        weak = self.weak(x)
        strong = self.strong(x)
        return self.normalize(weak), self.normalize(strong)
    
    
    
class TransformLabeled(object):
    def __init__(self, mean, std):
        self.strong = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(size=32,
                                  padding=int(32*0.125),
                                  padding_mode='reflect')])
        self.normalize = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)])

    def __call__(self, x):
        weak = x
        strong = self.strong(x)
        return self.normalize(strong)
    


class CIFAR10SSL(datasets.CIFAR10):
    def __init__(self, root, indexs, train=True,
                 transform=None, target_transform=None,
                 download=False):
        super().__init__(root, train=train,
                         transform=transform,
                         target_transform=target_transform,
                         download=download)
        if indexs is not None:
            self.data = self.data[indexs]
            self.targets = np.array(self.targets)[indexs]

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target


class CIFAR10_Unlabel(datasets.CIFAR10):
    def __init__(self, root,labeled_idxs,unlabeled_idxs, train=True,
                 transform=None, target_transform=None,
                 download=False,method: str='semisupervised'):
        super().__init__(root, train=train,
                         transform=transform,
                         target_transform=target_transform,
                         download=download)
        if unlabeled_idxs is not None:
            self.idx = np.arange(len(self.targets))
            self.labeled_idx = labeled_idxs
            self.unlabeled_idx = unlabeled_idxs
            self.all_data = self.data
            self.targets = np.array(self.targets)
            self.xdata = self.data[labeled_idxs]
            self.xtargets = self.targets[labeled_idxs]
            self.udata = self.data[unlabeled_idxs]
            self.utargets = self.targets [unlabeled_idxs]
            self._pseudo_labels = list()
            self._pseudo_labels_weights = list()
            self.method = method.lower()
            print(self.labeled_idx,len(self.labeled_idx))
            print(self.unlabeled_idx,len(self.unlabeled_idx))
    def __len__(self):
        return len(self.data)

    def get_pseudo_labels(self):
        return self._pseudo_labels

    def set_pseudo_labels(self, pseudo_labels):
        self._pseudo_labels = pseudo_labels

    def set_pseudo_labels_weights(self, pseudo_labels_weights):
        self._pseudo_labels_weights = pseudo_labels_weights

    def _pseudolabeling__getitem__(self, idx):
        idx = self.idx[idx]
        img = self.all_data[idx]
        target = self.targets[idx]
        labeled_mask = np.array([False], dtype=np.bool)
        if idx in self.labeled_idx:
            labeled_mask[0] = True
        idx = np.asarray([idx])
        return  img, target,labeled_mask, idx, 

    def _normal__getitem__(self, idx): 
        uidx = np.random.randint(0, len(self.unlabeled_idx))
        uidx = self.unlabeled_idx[uidx]
        img,target = self.all_data[uidx], self.targets[uidx]
        img = Image.fromarray(img)
        if self.transform is not None:
            img1 = self.transform(img)
            img2 = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)
            
        if len(self._pseudo_labels):
            utarget = self._pseudo_labels[uidx]
            uweight = self._pseudo_labels_weights[uidx]
            
            return img1,img2,target, utarget, uweight
        return img1,img2,target


    def __getitem__(self, idx):
        if self.method == 'pseudolabeling':
            return self._pseudolabeling__getitem__(idx)
        else:
            return self._normal__getitem__(idx)
       

    
def interleave(x, size):
    s = list(x.shape)
    return x.reshape([-1, size] + s[1:]).transpose(0, 1).reshape([-1] + s[1:])


def de_interleave(x, size):
    s = list(x.shape)
    return x.reshape([size, -1] + s[1:]).transpose(0, 1).reshape([-1] + s[1:])    
    
    
    
class UDataLoader(BaseDataLoader):
    """
    OpenML data loading demo using BaseDataLoader
    """
    def __init__(self,dataset, batch_size, shuffle=True, validation_split=0.0, num_workers=1,drop_last=True,training=True, **kwargs):
        self.num_workers = num_workers
        self.batch_size = batch_size
        self.training = training
        self.validation_split = validation_split
        self.kwargs = kwargs
        self.dataset = dataset
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)

    def get_pseudolabeling_loader(self):
        if hasattr(self, 'pseudolabeling_loader'):
            return self.pseudolabeling_loader

        dataset = copy.copy(self.dataset)
        dataset.method = 'pseudolabeling'
        self.pseudolabeling_loader = BaseDataLoader(dataset,self.batch_size,shuffle=True,validation_split=0.0, num_workers=4,drop_last=True)

        return self.pseudolabeling_loader

    def update_pseudo_labels(self, model, pseudolabeler, device=None):

        model.eval()
        data_loader = self.get_pseudolabeling_loader()

        latents = list()
        labels = list()
        labels_mask = list()
        idxs = list()

        for i, data in enumerate(data_loader):
            img, target,labeled_mask, idx = data[0].to(device), data[1],data[2],data[3]
            inputs = img.permute(0,3,1,2).float()
          
            # Use the output of the transformer to do the label propagation
            with torch.no_grad():
                logit,proj = model(inputs)
                
                z = F.normalize(proj, dim=1)
                #logits  = de_interleave(logit,1)
                #z = (x / x.norm(dim=-1, keepdim=True)).flatten(1,2)
            latents.append(z.detach().cpu())
            labels.append(target)
            labels_mask.append(labeled_mask)
            idxs.append(idx)
            """
            # Use the output of the prejection layer to do the label propagation
            with torch.no_grad():
                _ , x_categ_enc, x_cont_enc = embed_data_mask(x_categ, x_cont, cat_mask, con_mask,model,vision_dset)
                x = model.transformer(x_categ_enc, x_cont_enc)
                x = (x / x.norm(dim=-1, keepdim=True)).flatten(1,2)
                z = model.pt_mlp(x)
            latents.append(z.detach().cpu())
            labels.append(target)
            labels_mask.append(labeled_mask)
            idxs.append(idx)
            """
        latents = torch.cat(latents, dim=0).numpy()
        labels = torch.cat(labels, dim=0).squeeze().numpy()
        labels_mask = torch.cat(labels_mask, dim=0).squeeze().numpy()
        idxs = torch.cat(idxs, dim=0).numpy()
        ordered_idxs = np.arange(0, len(latents)).reshape(-1, 1)
        input_data = (latents, labels, labels_mask, ordered_idxs)
        pseudo_labels, acc, masks = pseudolabeler(*input_data)
        idxs = idxs.squeeze()
        labels_ = np.zeros(len(data_loader.dataset.data))
        labels_[idxs] = pseudo_labels
        weights = np.zeros(len(pseudo_labels))
        weights[idxs] = masks
        
        self.dataset.set_pseudo_labels(labels_)
        self.dataset.set_pseudo_labels_weights(weights)
        return acc
    
    
    
    
    
    
    

class CIFAR100SSL(datasets.CIFAR100):
    def __init__(self, root, indexs, train=True,
                 transform=None, target_transform=None,
                 download=False):
        super().__init__(root, train=train,
                         transform=transform,
                         target_transform=target_transform,
                         download=download)
        if indexs is not None:
            self.data = self.data[indexs]
            self.targets = np.array(self.targets)[indexs]

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target


DATASET_GETTERS = {'cifar10': get_cifar10,
                   'cifar100': get_cifar100}
