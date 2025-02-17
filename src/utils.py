import torch.nn.functional as F
import torch
def to_one_hot(labels, nb_class):
    """
    Converts a tensor of labels with shape (n, w, h) into a one-hot encoded tensor of shape (n, nb_class, w, h).

    Args:
        labels (torch.Tensor): Tensor of labels with shape (n, w, h) and dtype torch.int8.
        nb_class (int): Number of classes.

    Returns:
        torch.Tensor: One-hot encoded tensor of shape (n, nb_class, w, h).
    """

    labels = labels.long() # (n, w, h)
    one_hot = F.one_hot(labels, num_classes=nb_class) # (n, w, h, nb_class)
    one_hot = one_hot.permute(0, 3, 1, 2) # (n, nb_class, w, h)
    return one_hot


def box_criterion(box_outputs, box_true, delta_length=20):
    return (((box_outputs - box_true)/delta_length) ** 2).mean()


def lax_box_criterion(box_outputs, box_true, delta_length=20):
    """ box : (xmin, ymin, xmax, ymax)

    Lax loss wants to make sure the predicted boxes contains the true boxes. It should be use with classic box criterion
    as a type of regularizer that optimize the inference dice"""
    delta_box = box_outputs - box_true # (b, num_classes, 4)
    condition_swapper = torch.Tensor([-1,-1,1,1.]) # we want xmin and ymin to be under true box and xmax ymax to be above.
    condition_swapper = condition_swapper.to(box_outputs.device).view(1,1,-1)
    loss = (((F.relu(condition_swapper * delta_box))/delta_length)**2).mean()
    return loss





