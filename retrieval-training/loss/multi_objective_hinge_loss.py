import torch
import torch.nn as nn


class MultiTaskHingeLoss(nn.Module):
    def __init__(self, epsilon_high=0.8, epsilon_low=0.65, epsilon_neg=0.4, m=2, desirability_weight=0.7, relevance_weight=0.3):
        super(MultiTaskHingeLoss, self).__init__()
        self.epsilon_high = epsilon_high
        self.epsilon_low = epsilon_low
        self.epsilon_neg = epsilon_neg
        self.m = m
        self.desirability_weight = desirability_weight
        self.relevance_weight = relevance_weight

    def compute_loss(self, y_pred, y_true):
        label_high = (y_true == 2).float()
        label_low = (y_true == 1).float()
        label_neg = (y_true == 0).float()

        loss_high = torch.pow(torch.clamp(self.epsilon_high - y_pred, min=0), self.m)
        loss_low = torch.pow(torch.clamp(y_pred - self.epsilon_low, min=0), self.m)
        loss_neg = torch.pow(torch.clamp(y_pred - self.epsilon_neg, min=0), self.m)

        loss = label_high * loss_high + label_low * loss_low + label_neg * loss_neg
        return loss.mean()

    def forward(self, y_pred, desirability_label, relevance_label):
        padded_desirability_label = torch.cat([desirability_label, 
                                               torch.zeros(y_pred.size(0) - desirability_label.size(0), 
                                                           device=y_pred.device)])
        padded_relevance_label = torch.cat([relevance_label, 
                                            torch.zeros(y_pred.size(0) - relevance_label.size(0), 
                                                        device=y_pred.device)])

        desirability_loss = self.compute_loss(y_pred, padded_desirability_label)
        relevance_loss = self.compute_loss(y_pred, padded_relevance_label)

        total_loss = self.desirability_weight * desirability_loss + self.relevance_weight * relevance_loss
        return total_loss