import torch
import torch.nn.functional as F
from engine.loss.build import LOSS_ARCH_REGISTRY

__all__ = [
    'FocalLoss'
]


def sigmoid_focal_loss(
        inputs: torch.Tensor,
        targets: torch.Tensor,
        alpha: float = -1,
        gamma: float = 2,
        reduction: str = "none",
) -> torch.Tensor:
    """
    Loss used in RetinaNet for dense detection: https://arxiv.org/abs/1708.02002.
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
        alpha: (optional) Weighting factor in range (0,1) to balance
                positive vs negative examples. Default = -1 (no weighting).
        gamma: Exponent of the modulating factor (1 - p_t) to
               balance easy vs hard examples.
        reduction: 'none' | 'mean' | 'sum'
                 'none': No reduction will be applied to the output.
                 'mean': The output will be averaged.
                 'sum': The output will be summed.
    Returns:
        Loss tensor with the reduction option applied.
    """
    p = torch.sigmoid(inputs)
    ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    p_t = p * targets + (1 - p) * (1 - targets)
    loss = ce_loss * ((1 - p_t) ** gamma)

    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss

    if reduction == "mean":
        loss = loss.mean()
    elif reduction == "sum":
        loss = loss.sum()

    return loss


sigmoid_focal_loss_jit = torch.jit.script(
    sigmoid_focal_loss
)  # type: torch.jit.ScriptModule


def sigmoid_focal_loss_star(
        inputs: torch.Tensor,
        targets: torch.Tensor,
        alpha: float = -1,
        gamma: float = 1,
        reduction: str = "none",
) -> torch.Tensor:
    """
    FL* described in RetinaNet paper Appendix: https://arxiv.org/abs/1708.02002.
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
        alpha: (optional) Weighting factor in range (0,1) to balance
                positive vs negative examples. Default = -1 (no weighting).
        gamma: Gamma parameter described in FL*. Default = 1 (no weighting).
        reduction: 'none' | 'mean' | 'sum'
                 'none': No reduction will be applied to the output.
                 'mean': The output will be averaged.
                 'sum': The output will be summed.
    Returns:
        Loss tensor with the reduction option applied.
    """
    shifted_inputs = gamma * (inputs * (2 * targets - 1))
    loss = -(F.logsigmoid(shifted_inputs)) / gamma

    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss *= alpha_t

    if reduction == "mean":
        loss = loss.mean()
    elif reduction == "sum":
        loss = loss.sum()

    return loss


sigmoid_focal_loss_star_jit = torch.jit.script(
    sigmoid_focal_loss_star
)  # type: torch.jit.ScriptModule


@LOSS_ARCH_REGISTRY.register()
class FocalLoss(torch.nn.Module):
    def __init__(self,
                 lambda_weight=1.,
                 gamma=2.0,
                 alpha=0.25,
                 reduction='mean',
                 apply_sigmoid=True
                 ):
        super(FocalLoss, self).__init__()

        assert reduction in ('none', 'mean', 'sum')

        self.lambda_weight = lambda_weight
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction
        self.apply_sigmoid = apply_sigmoid
        return

    def sigmoid_focal_loss(self,
                           pred,
                           target,
                           weight=None,
                           gamma=2.0,
                           alpha=0.25,
                           reduction='mean',
                           avg_factor=None):
        """
        Shape:
            - pred: :math:`(N, C)` where `C = number of classes`, or
              :math:`(N, C, d_1, d_2, ..., d_K)`
            - target: (the onehot) same shape as the input.
        """

        # Function.apply does not accept keyword arguments, so the decorator
        # "weighted_loss" is not applicable
        loss = sigmoid_focal_loss_jit(pred, target, gamma=gamma, alpha=alpha)
        if weight is not None:
            if weight.shape != loss.shape:
                if weight.size(0) == loss.size(0):
                    # For most cases, weight is of shape (num_priors, ),
                    #  which means it does not have the second axis num_class
                    weight = weight.view(-1, 1)
                else:
                    # Sometimes, weight per anchor per class is also needed. e.g.
                    #  in FSAF. But it may be flattened of shape
                    #  (num_priors x num_class, ), while loss is still of shape
                    #  (num_priors, num_class).
                    assert weight.numel() == loss.numel()
                    weight = weight.view(loss.size(0), -1)
            assert weight.ndim == loss.ndim
        loss = self.weight_reduce_loss(loss, weight, reduction, avg_factor)
        return loss

    def weight_reduce_loss(self, loss, weight=None, reduction='mean', avg_factor=None):
        """Apply element-wise weight and reduce loss.
        Args:
            loss (Tensor): Element-wise loss.
            weight (Tensor): Element-wise weights.
            reduction (str): Same as built-in losses of PyTorch.
            avg_factor (float): Avarage factor when computing the mean of losses.
        Returns:
            Tensor: Processed loss values.
        """
        # if weight is specified, apply element-wise weight
        if weight is not None:
            loss = loss * weight

        # if avg_factor is not specified, just reduce the loss
        if avg_factor is None:
            loss = self.reduce_loss(loss, reduction)
        else:
            # if reduction is mean, then average the loss by avg_factor
            if reduction == 'mean':
                loss = loss.sum() / avg_factor
            # if reduction is 'none', then do nothing, otherwise raise an error
            elif reduction != 'none':
                raise ValueError('avg_factor can not be used with reduction="sum"')
        return loss

    def reduce_loss(self, loss, reduction):
        """Reduce loss as specified.
        Args:
            loss (Tensor): Elementwise loss tensor.
            reduction (str): Options are "none", "mean" and "sum".
        Return:
            Tensor: Reduced loss tensor.
        """
        # none: 0, elementwise_mean:1, sum: 2
        if reduction == 'none':
            return loss
        elif reduction == 'mean':
            return loss.mean()
        elif reduction == 'sum':
            return loss.sum()

    def forward(self,
                pred,
                target):

        if self.apply_sigmoid:
            loss_cls = self.sigmoid_focal_loss(
                pred,
                target,
                gamma=self.gamma,
                alpha=self.alpha,
                reduction=self.reduction)
        else:
            raise NotImplementedError

        return self.lambda_weight * loss_cls

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        format_string += ' ,lambda_weight: {}, '.format(self.lambda_weight)
        format_string += ' ,apply_sigmoid: {}, '.format(self.apply_sigmoid)
        format_string += ' ,reduction: {})'.format(self.reduction)
        return format_string
