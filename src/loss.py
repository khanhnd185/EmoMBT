import torch
import torch.nn as nn
from torch.nn.functional import normalize

class SupConLoss(nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR"""
    def __init__(self, temperature=0.07, contrast_mode='all', multiclass=True,
                 base_temperature=0.07):
        super(SupConLoss, self,).__init__()
        self.multiclass = multiclass
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(self, features, labels=None, mask=None):
        """Compute loss for model. If both `labels` and `mask` are None,
        it degenerates to SimCLR unsupervised loss:
        https://arxiv.org/pdf/2002.05709.pdf

        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """
        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))

        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            if self.multiclass:
                labels = torch.argmax(labels, dim=1)
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature)
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()

        return loss


class MultiheadSupervisedContrastiveLossNoNegatives(nn.Module):
    """ Multi-Label Contrastive Loss without the term $\Sigma_{i,k}$ (incorrectly completed RPMs). """

    def __init__(self, temperature=0.07, base_temperature=0.07, num_heads=1):
        super(MultiheadSupervisedContrastiveLossNoNegatives, self).__init__()
        self.temperature = temperature
        self.base_temperature = base_temperature
        self.num_heads = num_heads

    def forward(self, features: torch.Tensor, labels: torch.Tensor):
        """
        Forward pass of the multi-head multi label supervised contrastive loss.
        Both features and incorrect_features should be tensors with logits (unnormalized).
        :param features: a Tensor with shape (batch_size, num_views, embedding_size)
        :param labels: a Tensor with multi-label class assignment with shape (batch_size, num_classes)
        :returns: loss as a single element Tensor
        """
        device = features.device
        batch_size, num_views, embedding_size = features.size()
        mask = (torch.matmul(labels, labels.T) > 0)

        features = torch.cat(torch.unbind(features, dim=1), dim=0)

        # split heads
        num_features, feature_size = features.size()
        feature_size = feature_size // self.num_heads
        features = features.view(num_features, self.num_heads, feature_size)

        # normalize each head
        features = normalize(features, dim=-1)

        # compute logits
        dot_products = torch.matmul(
            features.view(-1, feature_size),
            features.view(-1, feature_size).T
        ).view(num_features, self.num_heads, num_features, self.num_heads).permute(0, 2, 1, 3)
        dot_products = dot_products.diagonal(dim1=2, dim2=3)
        heads = dot_products.argmax(dim=-1)

        # scale logits with temperature
        dot_products = dot_products / self.temperature

        # choose dot products of appropriate heads
        h1 = heads.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, 1, num_features)
        dot_products = dot_products.unsqueeze(dim=2).repeat(1, 1, num_features, 1)
        dot_products = dot_products.permute(0, 2, 3, 1)
        dot_products = dot_products.gather(dim=2, index=h1).squeeze()

        # for numerical stability
        logits_max, _ = dot_products.max(dim=2, keepdim=True)
        logits = dot_products - logits_max.detach()

        # tile mask
        mask = mask.repeat(num_views, num_views)

        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask, device=device),
            1,
            torch.arange(batch_size * num_views, device=device).view(-1, 1),
            0
        )
        mask = mask * logits_mask

        # compute log_prob: log(exp(x1)/sum(exp(x2))) = log(exp(x1)) - log(sum(exp(x2))) = x1 - log(sum(exp(x2)))
        # logits of positive pairs lie on the diagonal
        logits_mask = logits_mask.unsqueeze(dim=1).repeat(1, num_features, 1)
        exp_logits = logits.exp() * logits_mask  # don't include anchor * anchor in the denominator
        log_prob = logits.diagonal(dim1=1, dim2=2) - exp_logits.sum(2).log()

        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos

        # When augmentation is not used (num_views == 1) and given batch contains an anchor without positive pairs, then
        # mask.sum(1)[anchor] will be 0 and therefore mean_log_prob_pos[anchor] will be nan
        # Such anchors are excluded from loss calculation
        return loss[loss.isfinite()].mean()
