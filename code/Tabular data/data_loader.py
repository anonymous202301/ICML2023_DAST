import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate
from torch.utils.data.sampler import SubsetRandomSampler
#from OpenML import DataSetCatCon
import copy
from augmentations import embed_data_mask

vision_dset = True

import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')

class BaseDataLoader(DataLoader):
    """
    Base class for all data loaders
    """
    def __init__(self, dataset, batch_size, shuffle, validation_split= 0.0, num_workers = 1, collate_fn=default_collate):
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


class OpenMLDataLoader(BaseDataLoader):
    """
    OpenML data loading demo using BaseDataLoader
    """
    def __init__(self,dataset, batch_size, shuffle=True, validation_split=0.0, num_workers=1, training=True, **kwargs):
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
        self.pseudolabeling_loader = BaseDataLoader(dataset, 32, shuffle=True,  validation_split=0.0, num_workers=self.num_workers)

        return self.pseudolabeling_loader

    def update_pseudo_labels(self, model, pseudolabeler, device=None):

        model.eval()
        data_loader = self.get_pseudolabeling_loader()

        latents = list()
        labels = list()
        labels_mask = list()
        idxs = list()

        for i, data in enumerate(data_loader):
            x_categ, x_cont, target,cat_mask, con_mask,labeled_mask, idx = data[0].to(device), data[1].to(device),data[2],data[3].to(device),data[4].to(device),data[5],data[6]
            
            
            # Use the output of the transformer to do the label propagation
            with torch.no_grad():
                _ , x_categ_enc, x_cont_enc = embed_data_mask(x_categ, x_cont, cat_mask, con_mask,model,vision_dset)
                x = model.transformer(x_categ_enc, x_cont_enc)
                z = (x / x.norm(dim=-1, keepdim=True)).flatten(1,2)
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
        pseudo_labels, acc = pseudolabeler(*input_data)
        idxs = idxs.squeeze()
        labels_ = np.zeros(len(data_loader.dataset.data))
        labels_[idxs] = pseudo_labels
        weights = np.zeros(len(pseudo_labels))
        weights[idxs] = pseudolabeler.p_weights
        self.dataset.set_pseudo_labels(labels_)
        self.dataset.set_pseudo_labels_weights(weights)
        return acc
