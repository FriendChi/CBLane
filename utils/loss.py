import torch
import torch.nn as nn
import torch.nn.functional as F
import json
import numpy as np

def compute_boundary_weights(labels, alpha=1.0):
    beta = alpha * 0.5
    # 计算相邻点之间的差异 (一阶邻域)
    diff1 = torch.abs(labels[:, 1:, :] - labels[:, :-1, :])
    
    # 计算隔一个点的差异 (二阶邻域)
    diff2 = torch.abs(labels[:, 2:, :] - labels[:, :-2, :])

    # 初始化权重矩阵
    boundary_weights = torch.zeros_like(labels).float()
    
    # 为一阶邻域加上权重
    boundary_weights[:, 1:, :] += alpha * diff1
    boundary_weights[:, :-1, :] += alpha * diff1

    # 为二阶邻域加上较小的权重 (权重由 beta 控制)
    boundary_weights[:, 2:, :] += beta * diff2
    boundary_weights[:, :-2, :] += beta * diff2
    
    # 加 1.0 保证权重不为 0
    boundary_weights = boundary_weights + 1.0
    return boundary_weights

class LaneAwareCrossEntropyLoss(nn.Module):
    def __init__(self, gamma=1.0, connectivity_weight=0.001):
        super(LaneAwareCrossEntropyLoss, self).__init__()
        self.gamma = gamma  # 权重调节参数
        self.cross_entropy_loss = nn.CrossEntropyLoss(reduction='none')  # 交叉熵损失
        self.connectivity_weight = connectivity_weight  # 连通性损失的权重

    def forward(self, logits, targets):
        """
        logits: [batch_size, num_classes, num_points, num_lanes] (32, 2, 41, 4)
        targets: [batch_size, num_points, num_lanes] (32, 41, 4)
        """
        batch_size, num_classes, num_points, num_lanes = logits.shape

        # 计算基础交叉熵损失
        loss_ce = self.cross_entropy_loss(logits, targets)  # [batch_size, num_points, num_lanes]

        # 计算权重矩阵
        weights = compute_boundary_weights(targets, self.gamma)
        # 加权交叉熵损失
        weighted_loss_ce = loss_ce * weights

        # 计算连通性损失
        probs = F.softmax(logits, dim=1)  # [batch_size, 2, 41, 4]
        pred_classes = torch.argmax(probs, dim=1)  # [batch_size, 41, 4]
        
        connectivity_loss = torch.zeros_like(loss_ce)

        for lane in range(num_lanes):
            for point in range(1, num_points): 
                same_class = (pred_classes[:, point, lane] == pred_classes[:, point - 1, lane]).float()
                connectivity_loss[:, point, lane] = 1 - same_class

        # 合并损失
        total_loss = weighted_loss_ce + self.connectivity_weight * connectivity_loss

        # 返回最终损失的均值
        return total_loss.mean()

class OhemCELoss(nn.Module):
    def __init__(self, thresh, n_min, ignore_lb=255, *args, **kwargs):
        super(OhemCELoss, self).__init__()
        self.thresh = -torch.log(torch.tensor(thresh, dtype=torch.float)).cuda()
        self.n_min = n_min
        self.ignore_lb = ignore_lb
        self.criteria = nn.CrossEntropyLoss(ignore_index=ignore_lb, reduction='none')

    def forward(self, logits, labels):
        N, C, H, W = logits.size()
        loss = self.criteria(logits, labels).view(-1)
        loss, _ = torch.sort(loss, descending=True)
        if loss[self.n_min] > self.thresh:
            loss = loss[loss>self.thresh]
        else:
            loss = loss[:self.n_min]
        return torch.mean(loss)

def soft_nll(pred, target, ignore_index = -1):
    C = pred.shape[1]
    invalid_target_index = target==ignore_index

    ttarget = target.clone()
    ttarget[invalid_target_index] = C

    target_l = target - 1
    target_r = target + 1

    invalid_part_l = target_l == -1
    invalid_part_r = target_r == C

    invalid_target_l_index = torch.logical_or(invalid_target_index, invalid_part_l)
    target_l[invalid_target_l_index] = C

    invalid_target_r_index = torch.logical_or(invalid_target_index, invalid_part_r)
    target_r[invalid_target_r_index] = C

    supp_part_l = target.clone()
    supp_part_r = target.clone()
    supp_part_l[target!=0] = C
    supp_part_r[target!=C-1] = C

    target_onehot = torch.nn.functional.one_hot(ttarget, num_classes=C+1)
    target_onehot = target_onehot[...,:-1].permute(0,3,1,2)

    target_l_onehot = torch.nn.functional.one_hot(target_l, num_classes=C+1)
    target_l_onehot = target_l_onehot[...,:-1].permute(0,3,1,2)

    target_r_onehot = torch.nn.functional.one_hot(target_r, num_classes=C+1)
    target_r_onehot = target_r_onehot[...,:-1].permute(0,3,1,2)

    supp_part_l_onehot = torch.nn.functional.one_hot(supp_part_l, num_classes=C+1)
    supp_part_l_onehot = supp_part_l_onehot[...,:-1].permute(0,3,1,2)

    supp_part_r_onehot = torch.nn.functional.one_hot(supp_part_r, num_classes=C+1)
    supp_part_r_onehot = supp_part_r_onehot[...,:-1].permute(0,3,1,2)

    target_fusion = 0.9 * target_onehot + 0.05 * target_l_onehot + 0.05 * target_r_onehot + 0.05 * supp_part_l_onehot + 0.05 * supp_part_r_onehot
    # import pdb; pdb.set_trace()
    return -(target_fusion * pred).sum() / (target!=ignore_index).sum()

class SoftmaxFocalLoss(nn.Module):
    def __init__(self, gamma, ignore_lb=255, soft_loss = True, *args, **kwargs):
        super(SoftmaxFocalLoss, self).__init__()
        self.gamma = gamma
        self.ignore_lb = ignore_lb
        self.soft_loss = soft_loss
        if not self.soft_loss:
            self.nll = nn.NLLLoss(ignore_index=ignore_lb)


    def forward(self, logits, labels):
        scores = F.softmax(logits, dim=1)
        factor = torch.pow(1.-scores, self.gamma)
        log_score = F.log_softmax(logits, dim=1)
        log_score = factor * log_score
        if self.soft_loss:
            loss = soft_nll(log_score, labels, ignore_index = self.ignore_lb)
        else:
            loss = self.nll(log_score, labels)

        # import pdb; pdb.set_trace()
        return loss

class ParsingRelationLoss(nn.Module):
    def __init__(self):
        super(ParsingRelationLoss, self).__init__()
    def forward(self,logits):
        n,c,h,w = logits.shape
        loss_all = []
        for i in range(0,h-1):
            loss_all.append(logits[:,:,i,:] - logits[:,:,i+1,:])
        #loss0 : n,c,w
        loss = torch.cat(loss_all)
        return torch.nn.functional.smooth_l1_loss(loss,torch.zeros_like(loss))

class MeanLoss(nn.Module):
    def __init__(self):
        super(MeanLoss, self).__init__()
        self.l1 = nn.SmoothL1Loss(reduction = 'none')
    def forward(self, logits, label):
        n,c,h,w = logits.shape
        grid = torch.arange(c, device=logits.device).view(1,c,1,1)
        logits = (logits.softmax(1) * grid).sum(1)
        loss = self.l1(logits, label.float())[label != -1]
        return loss.mean()

class VarLoss(nn.Module):
    def __init__(self, power = 2):
        super(VarLoss, self).__init__()
        self.power = power
    def forward(self, logits, label):
        n,c,h,w = logits.shape
        grid = torch.arange(c, device=logits.device).view(1,c,1,1)
        logits = logits.softmax(1)
        mean = (logits * grid).sum(1).view(n,1,h,w)
        # n,1,h,w
        var = (mean - grid).abs().pow(self.power) * logits
        # var = ((mean - grid).abs() - 4) * logits
        # n,c,h,w
        loss = var.sum(1)[(label != -1 ) & ( (label - mean.squeeze()).abs() < 1) ]
        return loss.mean()

class EMDLoss(nn.Module):
    def __init__(self):
        super(EMDLoss, self).__init__()
    def forward(self, logits, label):
        n, c, h, w = logits.shape
        grid = torch.arange(c, device=logits.device).view(1, c, 1, 1)
        logits = logits.softmax(1)
        # n,1,h,w
        var = (label.reshape(n, 1, h, w) - grid) * (label.reshape(n, 1, h, w) - grid) * logits
        # n,c,h,w
        loss = var.sum(1)[label != -1]
        return loss.mean()

class ParsingRelationDis(nn.Module):
    def __init__(self):
        super(ParsingRelationDis, self).__init__()
        self.l1 = torch.nn.L1Loss()
        # self.l1 = torch.nn.MSELoss()
    def forward(self, x):
        n,dim,num_rows,num_cols = x.shape
        x = torch.nn.functional.softmax(x[:,:dim-1,:,:],dim=1)
        embedding = torch.Tensor(np.arange(dim-1)).float().to(x.device).view(1,-1,1,1)
        pos = torch.sum(x*embedding,dim = 1)

        diff_list1 = []
        for i in range(0,num_rows // 2):
            diff_list1.append(pos[:,i,:] - pos[:,i+1,:])

        loss = 0
        for i in range(len(diff_list1)-1):
            loss += self.l1(diff_list1[i],diff_list1[i+1])
        loss /= len(diff_list1) - 1
        return loss


def cross_entropy(pred, target, reduction='elementwise_mean'):
    res  = -target * torch.nn.functional.log_softmax(pred, dim=1)
    if reduction == 'elementwise_mean':
        return torch.mean(torch.sum(res, dim=1))
    elif reduction == 'sum':
        return torch.sum(torch.sum(res, dim=1))
    else:
        return res

class RegLoss(nn.Module):
    def __init__(self):
        super(RegLoss, self).__init__()
        self.l1 = nn.L1Loss(reduction = 'none')
    def forward(self, logits, label):
        n,c,h,w = logits.shape
        assert c == 1
        logits = logits.sigmoid()
        loss = self.l1(logits[:,0], label)[label != -1]
        # print(logits[0], label[0])
        # import pdb; pdb.set_trace()
        return loss.mean()

class TokenSegLoss(nn.Module):
    def __init__(self):
        super(TokenSegLoss, self).__init__()
        self.criterion = nn.BCELoss()
        self.max_pool = nn.MaxPool2d(4)

    def forward(self, logits, labels):
        return self.criterion(F.interpolate(logits, size=(200, 400), mode='bilinear').sigmoid(), (self.max_pool(labels[:, 0:1, :, :]) != 0).float())

def test_cross_entropy():
    pred = torch.rand(10,200,33,66)
    target = torch.randint(200,(10,33,66))
    target_one_hot = torch.nn.functional.one_hot(target, num_classes=200).permute(0,3,1,2)
    print(torch.nn.functional.cross_entropy(pred,target))
    print(cross_entropy(pred,target_one_hot))
    print(soft_nll(torch.nn.functional.log_softmax(pred, dim=1),torch.randint(-1,200,(10,33,66))))

    # assert torch.nn.functional.cross_entropy(pred,target) == cross_entropy(pred,target_one_hot)
    print('OK')



if __name__ == "__main__":
    test_cross_entropy()
