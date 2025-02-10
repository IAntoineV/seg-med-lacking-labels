def dice(y_true, y_pred):
    if y_true.sum() == 0:
        return 0
    area_intersection  = (y_true * y_pred).sum(dim=[0,2,3])
    return 2 * area_intersection / (y_true.sum(dim=[0,2,3])+ y_pred.sum(dim=[0,2,3])+ 1e-8)
