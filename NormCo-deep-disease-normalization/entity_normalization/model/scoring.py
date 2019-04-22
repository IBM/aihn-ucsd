import torch as th
import torch.nn as nn
from torch.autograd import Variable
from torch.nn.parameter import Parameter
import torch.nn.functional as F

eps = 1e-5
class MaxMarginLoss(nn.Module):
    def __init__(self, margin=1.0):
        super(MaxMarginLoss, self).__init__()
        self.margin = (margin)
        self.zero = Variable(th.zeros(1), requires_grad=True)
        self.lossfn = nn.MarginRankingLoss(margin=margin, size_average=True, reduce=True)

    def forward(self, dist):
        """
        Max margin loss

        Parameters:
        
        dist: The positive and negative distances. Positive distance should be the first column of the tensor, 
                negative distances are the rest of the columns. Each row is one batch
        """
        neg = dist[:,1:]
        pos = dist[:,0].unsqueeze(1).expand_as(neg)
        
        return self.lossfn(neg, pos, Variable(th.ones(1), requires_grad=False))

class CrossEntropyDistanceLoss(nn.Module):
    def __init__(self):
        super(CrossEntropyDistanceLoss, self).__init__()

        self.lossfn = nn.CrossEntropyLoss(size_average=True)
    
    def forward(self, dist):
        #Create the targets; always the first index
        targets = Variable(th.zeros(dist.size()[0]).long())
        return self.lossfn(dist, targets)

class EuclideanDistance(nn.Module):
    def __init__(self):
        super(EuclideanDistance, self).__init__()
        self.m = nn.Sigmoid()

    
    def forward(self, i, j):
        i_norm = self.m(i)
        j_norm = self.m(j)

        return th.sqrt(th.sum((i_norm - j_norm)**2, dim=-1))

class CosineSimilarity(nn.Module):
    def __init__(self, dim=-1):
        super(CosineSimilarity, self).__init__()
        self.m = nn.CosineSimilarity(dim=dim)
    
    def forward(self, i, j):
        i = F.normalize(i, p=2, dim=-1)

        j = F.normalize(j, p=2, dim=-1)

        return self.m(i,j)

class BilinearMap(nn.Module):
    def __init__(self, nunits):
        super(BilinearMap, self).__init__()
        self.map = Parameter(th.Tensor(nunits, nunits))
        self.nunits = nunits
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.eye_(self.map)

    def forward(self, l, r):
        ncon = l.shape[1] 
        nneg = r.shape[2]
        nunits = l.shape[3]

        first = th.mm(l.view(-1, nunits), self.map).view(-1, ncon, 1, nunits)
        return th.sum(first * r, dim=-1)
