import torch.nn.functional as F

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

