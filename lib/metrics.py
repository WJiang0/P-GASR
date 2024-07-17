import numpy as np
import torch

def mae_torch(pred, true, mask_value=None):
    if mask_value != None:
        mask = torch.gt(true, mask_value)
        pred = torch.masked_select(pred, mask)
        true = torch.masked_select(true, mask)
        loss = torch.mean(torch.abs(true-pred))
    return loss

    # all_loss = []
    # mask_sum = 0
    # for p, t in zip(pred, true):
    #     mask = torch.gt(t, mask_value)
    #     p = torch.masked_select(p, mask)
    #     t = torch.masked_select(t, mask)
    #     loss = torch.mean(torch.abs(t-p))

    #     mask_sum+=len(p)
    #     if torch.isnan(loss):
    #         continue
    #         # all_loss.append(torch.tensor(0).cuda())
    #     else:
    #         all_loss.append(loss)

    # batch_loss = torch.stack(all_loss)
    # f_loss = torch.mean(batch_loss)
    # # f_loss = sum_loss / mask_sum

    # return f_loss
    

def mae_torch2(pred, true, pe_weight, mask_value=None):
    all_loss = []
    for p, t in zip(pred, true):
        mask = torch.gt(t, mask_value)
        p = torch.masked_select(p, mask)
        t = torch.masked_select(t, mask)
        loss = torch.mean(torch.abs(t-p))

        if torch.isnan(loss):
            all_loss.append(torch.tensor(0).cuda())
        else:
            all_loss.append(loss)

    batch_loss = torch.stack(all_loss)
    f_loss = torch.sum(pe_weight*batch_loss)

    return f_loss

def mape_torch(pred, true, mask_value=None):
    if mask_value != None:
        mask = torch.gt(true, mask_value)
        pred = torch.masked_select(pred, mask)
        true = torch.masked_select(true, mask)
    return torch.mean(torch.abs(torch.div((true - pred), true)))

def mae_np(pred, true, mask_value=None):
    if mask_value != None:
        mask = np.where(true > (mask_value), True, False)
        true = true[mask]
        pred = pred[mask]
    return np.mean(np.absolute(pred-true))

def mape_np(pred, true, mask_value=None):
    if mask_value != None:
        mask = np.where(true > (mask_value), True, False)
        true = true[mask]
        pred = pred[mask]
    return np.mean(np.absolute(np.divide((true - pred), true)))

def test_metrics(pred, true, mask1=5, mask2=5):
    # mask1 filter the very small value, mask2 filter the value lower than a defined threshold
    assert type(pred) == type(true)
    if type(pred) == np.ndarray:
        mae  = mae_np(pred, true, mask1)
        mape = mape_np(pred, true, mask2)
    elif type(pred) == torch.Tensor:
        mae  = mae_torch(pred, true, mask1).item()
        mape = mape_torch(pred, true, mask2).item()
    else:
        raise TypeError
    return mae, mape


