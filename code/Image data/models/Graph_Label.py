import torch
import torch.nn
from torch import Tensor
import numpy as np
from sklearn.preprocessing import label_binarize
import math
import itertools
import functools
import faiss
from typing import List
import operator


import time
import scipy
import scipy.stats


class GraphLabelPropagation(torch.nn.Module):
    """
    adopted from: https://github.com/ahmetius/LP-DeepSSL/blob/master/lp/db_semisuper.py
    """
    def __init__(
            self,
            k=50,
            max_iter=20,
            alpha=0.99,
            threshold=0.4,
            ):
        super(GraphLabelPropagation, self).__init__()
        self.k = k
        self.max_iter = max_iter
        self.alpha = alpha
        self.threshold = threshold
    def forward(self,
            X: np.array,
            labels: np.array,
            labels_mask: np.array,
            idxs: np.array):

        labeled_idx = idxs[labels_mask]
        unlabeled_idx = idxs[~labels_mask]
        
        if not hasattr(self, 'num_classes'):
            self.num_classes = len(np.unique(labels))
            self.class_weights = np.ones(self.num_classes,)

        # kNN search for the graph
        d = X.shape[1]
        res = faiss.StandardGpuResources()
        flat_config = faiss.GpuIndexFlatConfig()
        flat_config.device = int(torch.cuda.device_count()) - 1
        
        index = faiss.GpuIndexFlatIP(res,d,flat_config)   # build the index

        faiss.normalize_L2(X)
        index.add(X)
        N = X.shape[0]
        Nidx = index.ntotal

        c = time.time()
        D, I = index.search(X, self.k + 1)
        elapsed = time.time() - c
        #print('kNN Search done in %d seconds' % elapsed)

        # Create the graph
        D = D[:, 1:] ** 3
        I = I[:, 1:]
        row_idx = np.arange(N)
        row_idx_rep = np.tile(row_idx, (self.k, 1)).T
        W = scipy.sparse.csr_matrix((D.flatten('F'), (row_idx_rep.flatten('F'), I.flatten('F'))), shape=(N, N))
        W = W + W.T

        # Normalize the graph
        W = W - scipy.sparse.diags(W.diagonal())
        S = W.sum(axis = 1)
        S[S==0] = 1
        D = np.array(1./ np.sqrt(S))
        D = scipy.sparse.diags(D.reshape(-1))
        Wn = D * W * D

        # Initiliaze the y vector for each class (eq 5 from the paper, normalized with the class size) and apply label propagation
        Z = np.zeros((N, self.num_classes))
        A = scipy.sparse.eye(Wn.shape[0]) - self.alpha * Wn
        for i in range(self.num_classes):
            cur_idx = labeled_idx[np.where(labels[labeled_idx] == i)]
            y = np.zeros((N,))
            y[cur_idx] = 1.0 / cur_idx.shape[0]
            f, _ = scipy.sparse.linalg.cg(A, y, tol=1e-6, maxiter=self.max_iter)
            Z[:,i] = f

        # Handle numberical errors
        Z[Z < 0] = 0

        # Compute the weight for each instance based on the entropy (eq 11 from the paper)
        probs_l1 = torch.nn.functional.normalize(torch.tensor(Z), 1)
        probs_l1[probs_l1 < 0] = 0
        """
        entropy = scipy.stats.entropy(probs_l1.T)
       
        ind_1_sorted = np.argsort(entropy)
        entropy_sorted = entropy[ind_1_sorted]
        num_remember = int(remember_rate * len(entropy_sorted))
        ind_1_updated = ind_1_sorted[:num_remember]
        ind_1_drop = ind_1_sorted[num_remember:]
        weights = entropy
        weights[ind_1_updated] = 1
        weights[ind_1_drop] = 0
        weights = 1 - entropy / np.log(self.num_classes)
        weights = weights / np.max(weights[weights < 10.0])
        """
        pseudo_label = torch.softmax(probs_l1.detach(), dim=-1)
        max_probs, targets_u = torch.max(pseudo_label, dim=-1)
        mask = max_probs.ge(self.threshold).float()
        p_labels = np.argmax(pseudo_label.numpy(), 1)
        weights = mask.numpy()

        # Compute the accuracy of pseudolabels for statistical purposes
        correct_idx = (p_labels == labels).astype(int)
        #correct_idx2 = (p_labels[ind_1_updated] == labels[ind_1_updated])
        acc = correct_idx.mean()
        p_labels[labeled_idx] = labels[labeled_idx]
        #weights[labeled_idx] = 1.0
        weights[np.isnan(weights)] = 0.0
        self.p_weights = weights
        self.p_labels = p_labels
        self.masks = self.p_weights.copy()
        #self.masks[self.masks<self.threshold] = 0
        # Compute the weight for each class
        for i in range(self.num_classes):
            cur_idx = np.where(np.asarray(self.p_labels) == i)[0]
            self.class_weights[i] = (float(labels.shape[0]) / self.num_classes) / cur_idx.size

        return self.p_labels, acc,self.masks