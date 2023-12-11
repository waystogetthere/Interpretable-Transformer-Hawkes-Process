import argparse
import numpy as np
import pickle
import time
import torch
import torch.nn as nn
import torch.optim as optim

import transformer.Constants as Constants
from transformer.Models import get_non_pad_mask
import Utils
from Utils import gen_Xne, public_kernel_display, univariate_reconstruct_sequence_intensity, public_kernel_heatmap, multivariate_synthetic_kernel_display,multivariate_reconstruct_sequence_intensity

from preprocess.Dataset import get_dataloader
from transformer.Models import Transformer
from tqdm import tqdm



def prepare_dataloader(opt):
    """ Load data and prepare dataloader. """

    def load_data(name, dict_name):
        with open(name, 'rb') as f:
            data = pickle.load(f, encoding='latin-1')
            num_types = data['dim_process']
            print(num_types, 'num_types')
            data = data[dict_name]

            return data, int(num_types)

    print('[Info] Loading train data...')
    print(opt.data)
    train_data, num_types = load_data(opt.data + 'train.pkl', 'train')
    print('[Info] Loading dev data...')
    dev_data, _ = load_data(opt.data + 'dev.pkl', 'dev')
    print('[Info] Loading test data...')
    test_data, _ = load_data(opt.data + 'test.pkl', 'test')

    # print(len(train_data), len(dev_data), len(test_data))
    # for seq_idx in range(len(train_data)):
    #     print(train_data[seq_idx])
    #     assert False

    trainloader = get_dataloader(train_data, opt.batch_size, shuffle=True)
    testloader = get_dataloader(test_data, opt.batch_size, shuffle=False)
    return trainloader, testloader, num_types


def train_epoch(model, training_data, optimizer, pred_loss_func, opt):
    """ Epoch operation in training phase. """

    model.train()

    total_event_ll = 0  # cumulative event log-likelihood
    total_time_se = 0  # cumulative time prediction squared-error
    total_event_rate = 0  # cumulative number of correct prediction
    total_num_event = 0  # number of total events
    total_num_pred = 0  # number of predictions

    total_loss = 0
    for batch in tqdm(training_data, mininterval=2,
                      desc='  - (Training)   ', leave=False):
        """ prepare data """
        event_time, time_gap, event_type = map(lambda x: x.to(opt.device), batch)

        event_time_clone = event_time.clone()
        event_time_comb, event_type_comb = gen_Xne(opt, event_time_clone, event_type)

        """ forward """
        optimizer.zero_grad()
        _, prediction, reguarlize_loss = model(event_type_comb, event_time_comb)

        """ backward """
        event_ll, non_event_ll, pred_type = prediction  # Utils.log_likelihood(model, enc_out, event_time, event_type)
        #pred_time, pred_type, event_ll, non_event_ll = prediction
        non_pad_mask = get_non_pad_mask(event_type)
        pred_type = pred_type * non_pad_mask

        pred_loss, pred_num_event = Utils.type_loss(pred_type, event_type, pred_loss_func)
        # pred_time = pred_time * non_pad_mask.squeeze(-1)

        # print(pred_time[0, :], 'pred_time')
        # print(event_time[0, :], 'event_time')
        # event likelihood
        event_loss = event_ll - non_event_ll

        # SE is usually large, scale it to stabilize training
        # print(reguarlize_loss, 'reg loss!')
        # assert False
        
        # w_v = model.encoder.layer_stack[0].slf_attn.w_vs.weight

        # v_dot_product = torch.matmul(w_v, w_v.transpose(-1,1))
        # loss = torch.sum(v_dot_product)/torch.norm(w_v)

        # assert False
        # loss_b = (model.intensity_decoder[0][0].bias - 0.2) ** 2 + (model.intensity_decoder[1][0].bias - 0.2) ** 2
        # print(loss_b)
        # assert False
        # print(reguarlize_loss, 'loss')
        # assert False
        # print(reguarlize_loss, 'hehe', event_loss)
        # assert False
        loss = (-1) * event_loss + pred_loss*0.5#+ reguarlize_loss*1e-4+ loss_b # + loss_b #+ reguarlize_loss * 1e-4 #+ #pred_loss + se
        # loss = (-1) * event_loss
        total_loss += loss
        # print('portion: {} {} {}'.format((-1) * event_loss/loss, pred_loss/loss, se/loss))

        # loss = loss+consist_loss*scale_consist_loss+ sparsity_loss*scale_sparsity_loss

        loss.backward()

        """ update parameters """
        optimizer.step()

        """ note keeping """
        total_event_ll += event_loss.item()
        total_event_rate += pred_num_event.item()
        total_num_event += event_type.ne(Constants.PAD).sum().item()
        # we do not predict the first event
        total_num_pred += event_type.ne(Constants.PAD).sum().item() - event_time.shape[0]
    # rmse = np.sqrt(total_time_se / total_num_pred)
    return total_event_ll / total_num_event, total_event_rate / total_num_pred


def eval_epoch(model, validation_data, pred_loss_func, opt):
    """ Epoch operation in evaluation phase. """

    model.eval()

    total_event_ll = 0  # cumulative event log-likelihood
    total_time_se = 0  # cumulative time prediction squared-error
    total_event_rate = 0  # cumulative number of correct prediction
    total_num_event = 0  # number of total events
    total_num_pred = 0  # number of predictions
    with torch.no_grad():
        for batch in tqdm(validation_data, mininterval=2,
                          desc='  - (Validation) ', leave=False):
            """ prepare data """
            event_time, time_gap, event_type = map(lambda x: x.to(opt.device), batch)
            event_time_clone = event_time.clone()

            """ forward """
            event_time_comb, event_type_comb = gen_Xne(opt, event_time_clone, event_type)
            # enc_out, prediction, _ = model(event_type, event_time)

            
            """ compute loss """
            _, prediction, _ = model(event_type_comb, event_time_comb)

            event_ll, non_event_ll, pred_type = prediction

            non_pad_mask = get_non_pad_mask(event_type)
            pred_type = pred_type * non_pad_mask

            pred_loss, pred_num_event = Utils.type_loss(pred_type, event_type, pred_loss_func)

            event_loss = event_ll - non_event_ll

            # event_ll, non_event_ll = Utils.log_likelihood(model, enc_out, event_time, event_type)
            # event_loss = -torch.sum(event_ll - non_event_ll)
            # _, pred_num = Utils.type_loss(prediction[0], event_type, pred_loss_func)
            # se = Utils.time_loss(prediction[1], event_time)

            """ note keeping """
            total_event_ll += event_loss.item()
            # total_time_se += se.item()
            total_event_rate += pred_num_event.item()
            total_num_event += event_type.ne(Constants.PAD).sum().item()
            total_num_pred += event_type.ne(Constants.PAD).sum().item() - event_time.shape[0]

    # rmse = np.sqrt(total_time_se / total_num_pred)
    return total_event_ll / total_num_event  , total_event_rate / total_num_pred


def train(model, training_data, validation_data, optimizer, scheduler, pred_loss_func, opt):
    """ Start training. """
    train_event_losses = []
    valid_event_losses = []  # validation log-likelihood
    valid_pred_losses = []  # validation event type prediction accuracy
    valid_rmse = []  # validation event time prediction RMSE
    for epoch_i in range(opt.epoch):
        epoch = epoch_i + 1
        print('[ Epoch', epoch, ']')

        start = time.time()
        # train_event, train_type, train_time, total_loss = train_epoch(model, training_data, optimizer, pred_loss_func, opt)
        train_event,  train_type = train_epoch(model, training_data, optimizer, pred_loss_func, opt)
        print('  - (Training)    loglikelihood: {ll: 8.5f}, '
               'accuracy: {acc:3.3f} '
              'elapse: {elapse:3.3f} min'
              .format(ll=train_event, acc=train_type, elapse=(time.time() - start) / 60))
        train_event_losses += [train_event]
        # scheduler.step()
        # continue
        eval_ = True
        if eval_:
            start = time.time()
            valid_event, valid_type= eval_epoch(model, validation_data, pred_loss_func, opt)
            print('  - (Testing)     loglikelihood: {ll: 8.5f}, '
                  'accuracy: {acc:3.3f} '
                'elapse: {elapse:3.3f} min'
                .format(ll=valid_event, acc=valid_type,elapse=(time.time() - start) / 60))

            if len(valid_event_losses) and valid_event > max(valid_event_losses):
                print('Best Model Found at epoch {} '.format(epoch_i))
                torch.save(model.state_dict(), 'saved_model/best_model.pth')
            valid_event_losses += [valid_event]
            
            # valid_pred_losses += [valid_type]
            # valid_rmse += [valid_time]
            print('  - [Info] Maximum ll: {event: 8.5f}, '
                # 'Maximum accuracy: {pred: 8.5f}, Minimum RMSE: {rmse: 8.5f}'
                .format(event=max(valid_event_losses)))

            # logging
            with open(opt.log, 'a') as f:
                f.write('{epoch}, {ll: 8.5f}\n'
                        .format(epoch=epoch, ll=valid_event))

        # scheduler.step()
    torch.save(model.state_dict(), 'saved_model/model.pth')
    import pickle as pkl
    with open('train_losses.pkl', 'wb') as f:
        pkl.dump(train_event_losses, f)
    with open('valid_losses.pkl', 'wb') as f:
        pkl.dump(valid_event_losses, f)
    
    import matplotlib.pyplot as plt

    plt.figure()
    plt.plot(train_event_losses, label='Train')
    plt.plot(valid_event_losses, label='Validation')
    plt.legend()
    plt.savefig('nll curve.png')
    plt.close()





def main():
    """ Main function. """

    parser = argparse.ArgumentParser()

    parser.add_argument('-data')

    parser.add_argument('-epoch', type=int, default=50)
    parser.add_argument('-batch_size', type=int, default=16)

    parser.add_argument('-d_model', type=int, default=16)
    parser.add_argument('-d_inner_hid', type=int, default=128)
    parser.add_argument('-d_k', type=int, default=16)
    parser.add_argument('-d_v', type=int, default=32)

    parser.add_argument('-n_head', type=int, default=1)
    parser.add_argument('-n_layers', type=int, default=1)

    parser.add_argument('-dropout', type=float, default=0.0)
    parser.add_argument('-lr', type=float, default=1e-3)
    parser.add_argument('-smooth', type=float, default=0.1)

    parser.add_argument('-log', type=str, default='log.txt')

    parser.add_argument('-train_able', type=int, default=1)
    parser.add_argument('-load_model', type=int, default=1)

    opt = parser.parse_args()


    # default device is CUDA
    # opt.device = torch.device('cuda') 
    opt.device = torch.device('cuda')
    # opt.device = torch.device('cpu') 
    
   # opt.data = "data/stackov"
    # opt.data = "data/inHomo/"
    # opt.data = "data/meme/"

    # opt.data = "data/exp multivariate/"
    # opt.data = "data/half-sin multivariate/"
    # opt.data = "data/exp multivariate/"
    
    # opt.data = "data/taxi/"
    # opt.data = "data/amazon/"
    # opt.data = "data/stackoverflow/"
    # opt.data = "data/taobao/"
    opt.data = "data/data_conttime/"
    # setup the log file
    with open(opt.log, 'w') as f:
        f.write('Epoch, Log-likelihood, Accuracy, RMSE\n')

    print('[Info] parameters: {}'.format(opt))

    """ prepare dataloader """

    trainloader, testloader, num_types = prepare_dataloader(opt)
    """ prepare model """
    model = Transformer(
        num_types=num_types,
        d_model=opt.d_model,
        d_inner=opt.d_inner_hid,
        n_layers=opt.n_layers,
        n_head=opt.n_head,
        d_k=opt.d_k,
        d_v=opt.d_v,
        dropout=opt.dropout
    )
    print(opt.device)
    model.to(opt.device)

    if opt.load_model :
        # model.load_state_dict(torch.load('saved_model/model.pth'))
        # model.load_state_dict(torch.load('saved_model/best_model_taxi.pth'))
        model.load_state_dict(torch.load('saved_model/best_model.pth'))
        # model.load_state_dict(torch.load('saved_model/cherry_model.pth'))
        print('load model!')

    # s = model.encoder.event_emb.weight
    # print(s.shape)
    # hehe = torch.matmul(s[1:,:],s[1:,:].T)
    # import matplotlib.pyplot as plt

    # plt.figure()
    # plt.imshow(hehe.detach().cpu().numpy())
    # print(torch.norm(s[1:,:],dim=-1))
    # plt.colorbar()
    # plt.savefig('lookup simiarity.png')
    # assert False
    # else:
    #     print('train from scratches!')
    """ optimizer and scheduler """
    optimizer = optim.Adam(filter(lambda x: x.requires_grad, model.parameters()),
                           opt.lr, betas=(0.9, 0.999), eps=1e-05) #, weight_decay=1e-3)
    scheduler = optim.lr_scheduler.StepLR(optimizer, 10, gamma=0.5)

    """ prediction loss function, either cross entropy or label smoothing """

    pred_loss_func = nn.CrossEntropyLoss(ignore_index=-1, reduction='none')

    """ number of parameters """
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('[Info] Number of parameters: {}'.format(num_params))
    
    """ train the model """
    if opt.train_able:
        train(model, trainloader, testloader, optimizer, scheduler, pred_loss_func, opt)
    # synthetic_kernel_display(model, opt)
    # synthetic_kernel_display(model, opt)
    # public_kernel_heatmap(model, opt)
    # public_kernel_display(model, opt)

    # reconstruct_sequence_intensity(model, trainloader, opt)
    
    multivariate_reconstruct_sequence_intensity(model, trainloader, opt)

    # multivariate_synthetic_kernel_display(model, opt)

    # Utils.cherry_plt('Sin')



if __name__ == '__main__':
    main()