import math
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import pickle as pkl

from transformer.Models import get_non_pad_mask

import matplotlib.pyplot as plt


def softplus(x, beta):
    # hard thresholding at 20
    temp = beta * x
    temp[temp > 20] = 20
    return 1.0 / beta * torch.log(1 + torch.exp(temp))


def compute_event(event, non_pad_mask):
    """ Log-likelihood of events. """
    # add 1e-9 in case some events have 0 likelihood
    event += math.pow(10, -9)
    event.masked_fill_(~non_pad_mask.bool(), 1.0)

    result = torch.log(event)
    return result


def compute_integral_biased(all_lambda, time, non_pad_mask):
    """ Log-likelihood of non-events, using linear interpolation. """

    diff_time = (time[:, 1:] - time[:, :-1]) * non_pad_mask[:, 1:]
    diff_lambda = (all_lambda[:, 1:] + all_lambda[:, :-1]) * non_pad_mask[:, 1:]

    biased_integral = diff_lambda * diff_time
    result = 0.5 * biased_integral
    return result


def compute_integral_unbiased(model, data, time, non_pad_mask, type_mask):
    """ Log-likelihood of non-events, using Monte Carlo integration. """

    num_samples = 100

    diff_time = (time[:, 1:] - time[:, :-1]) * non_pad_mask[:, 1:]
    temp_time = diff_time.unsqueeze(2) * \
                torch.rand([*diff_time.size(), num_samples], device=data.device)
    temp_time /= (time[:, :-1] + 1).unsqueeze(2)

    temp_hid = model.linear(data)[:, 1:, :]
    temp_hid = torch.sum(temp_hid * type_mask[:, 1:, :], dim=2, keepdim=True)

    all_lambda = softplus(temp_hid + model.alpha * temp_time, model.beta)
    all_lambda = torch.sum(all_lambda, dim=2) / num_samples

    unbiased_integral = all_lambda * diff_time
    return unbiased_integral


def type_loss(prediction, types, loss_func):
    """ Event prediction loss, cross entropy or label smoothing. """

    # convert [1,2,3] based types to [0,1,2]; also convert padding events to -1
    truth = types[:, 1:] - 1
    prediction = prediction[:, :-1, :]

    pred_type = torch.max(prediction, dim=-1)[1]
    correct_num = torch.sum(pred_type == truth)

    # compute cross entropy loss

    loss = loss_func(prediction.transpose(1, 2), truth)

    loss = torch.sum(loss)
    return loss, correct_num


def log_likelihood(model, enc_output, event_time, event_types):
    """ Log-likelihood of sequence. """

    non_pad_mask = get_non_pad_mask(event_types).squeeze(2)

    type_mask = torch.zeros([*event_types.size(), model.num_types], device=enc_output.device)
    for i in range(model.num_types): # mapping to device (gpu)
        type_mask[:, :, i] = (event_types == i + 1).bool().to(enc_output.device)

    all_hid = model.linear(enc_output)
    all_lambda = softplus(all_hid, model.beta)
    type_lambda = torch.sum(all_lambda * type_mask, dim=2)

    # event log-likelihood
    event_ll = compute_event(type_lambda, non_pad_mask)
    event_ll = torch.sum(event_ll, dim=-1)

    # non-event log-likelihood, either numerical integration or MC integration
    # non_event_ll = compute_integral_biased(type_lambda, time, non_pad_mask)
    non_event_ll = compute_integral_unbiased(model, enc_output, event_time, non_pad_mask, type_mask)
    non_event_ll = torch.sum(non_event_ll, dim=-1)

    return event_ll, non_event_ll



def gen_Xne(opt, event_time, event_type):

        # Step 1: Left shifted.
        # find the min event time and max event time. Note that padding entry 0 is excluded
        sorted, _ = torch.sort(torch.unique(event_time))
        
        min = sorted[1] if sorted[0] == 0 else sorted[0]
        max = sorted[-1]

        event_time[event_time == 0.0] = 42 * 10 ** 6

        # Manually construct Non-Event Matrix.

        event_time = event_time - min
        dummy_time = torch.arange(1e-1, float(max-min), 1e-1).to(opt.device).repeat(event_time.shape[0],1)  # Non-Event Time Point

        #dummy_time = torch.arange(min, max, 1e-1).to(opt.device).repeat(event_time.shape[0],1)  # Non-Event Time Point
        
        dummy_type = torch.ones_like(dummy_time) * (-1)  # Set the "Type" to be -1 at Non-Event Time Point
        for i in torch.unique(dummy_time):
            if i in event_time:
                # print('duplicate time Found! {}'.format(i)) # If conflict found between event time and Non-Event Time point
                dummy_time[dummy_time == i] = i + 1e-3  # shift the Non-Event Time point

        # Combination
        new_event_time = torch.cat((event_time, dummy_time), dim=1)
        new_event_type = torch.cat((event_type, dummy_type), dim=1)

        sorted_time, indice = new_event_time.sort()
        sorted_types = torch.zeros_like(new_event_type)
        for i in range(new_event_type.shape[0]):
            sorted_types[i, :] = new_event_type[i][indice[i]]
        sorted_types = sorted_types.long()
        sorted_time[sorted_time > 10 ** 6] = 0  # set padding entry to be 0 at Time Matrix
        sorted_time.to(opt.device)
        sorted_types.to(opt.device)

        return sorted_time, sorted_types

