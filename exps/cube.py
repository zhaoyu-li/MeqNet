#!/usr/bin/env python3
#
# Partly derived from:
#   https://github.com/locuslab/optnet/blob/master/sudoku/train.py

import argparse
import time
import os

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from tqdm.auto import tqdm

import satnet
import mixnet


def print_header(msg):
    print('===>', msg)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='cube_10000')
    parser.add_argument('--batchSz', type=int, default=40)
    parser.add_argument('--testBatchSz', type=int, default=40)
    parser.add_argument('--aux', type=int, default=300)
    parser.add_argument('--m', type=int, default=375)
    parser.add_argument('--nEpoch', type=int, default=500)
    parser.add_argument('--testPct', type=float, default=0.1)
    parser.add_argument('--lr', type=float, default=2e-3)
    parser.add_argument('--save', type=str)
    parser.add_argument('--no_cuda', action='store_true')
    parser.add_argument('--model', type=str, default='mixnet')
    parser.add_argument('--pretrained', type=str, default=None)

    args = parser.parse_args()

    # For debugging: fix the random seed
    # npr.seed(1)
    # torch.manual_seed(7)

    args.cuda = not args.no_cuda and torch.cuda.is_available()
    if args.cuda:
        print('Using', torch.cuda.get_device_name(0))
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.cuda.init()

    save = 'cube-{}-aux{}-m{}-lr{}-bsz{}'.format(args.model, args.aux, args.m, args.lr, args.batchSz)
    if args.save: save = '{}-{}'.format(args.save, save)
    save = os.path.join('logs', save)

    os.makedirs(save, exist_ok=True)

    print_header('Loading data')

    with open(os.path.join(args.data_dir, 'features.pt'), 'rb') as f:
        X = torch.load(f)
    with open(os.path.join(args.data_dir, 'labels.pt'), 'rb') as f:
        Y = torch.load(f)

    N = X.size(0)
    print("number of all data:", N)
    nTrain = int(N * (1. - args.testPct))

    print_header('Forming inputs')
    is_input = X.sum(dim=3, keepdim=True).expand_as(X).int().sign()

    if args.cuda: X, is_input, Y = X.cuda(), is_input.cuda(), Y.cuda()
    train_set = TensorDataset(X[:nTrain], is_input[:nTrain], Y[:nTrain])
    test_set = TensorDataset(X[nTrain:], is_input[nTrain:], Y[nTrain:])

    print_header('Building model')
    if args.model == 'mixnet':
        model = mixnet.MixNet(324, aux=args.aux)
    else:
        model = satnet.SATNet(324, m=args.m, aux=args.aux)

    if args.cuda: model = model.cuda()

    if args.pretrained is not None:
        print(model.load_state_dict(torch.load(args.pretrained)))

    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    cur_time = time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime(time.time()))
    log_file = os.path.join(save, f'{cur_time}.txt')

    # test(args.boardSz, 0, model, optimizer, test_logger, test_set, args.testBatchSz, unperm, log_file)
    for epoch in range(1, args.nEpoch+1):
        train(epoch, model, optimizer, train_set, args.batchSz, log_file)
        test(epoch, model, optimizer, test_set, args.testBatchSz, log_file)
        torch.save(model.state_dict(), os.path.join(save, 'it'+str(epoch)+'.pth'))

    # torch.save(model.state_dict(), os.path.join(save,'model.pt'))


def run(epoch, model, optimizer, dataset, batchSz, to_train=False, log_file=None):
    loss_final, err_final = 0, 0
    loader = DataLoader(dataset, batch_size=batchSz)
    tloader = tqdm(enumerate(loader), total=len(loader))

    for i, (data, is_input, label) in tloader:
        if to_train: optimizer.zero_grad()
        preds = model(data.flatten(1).contiguous(), is_input.flatten(1).contiguous()).reshape(data.shape)
        loss = nn.functional.binary_cross_entropy(preds, label)

        if to_train:
            loss.backward()
            optimizer.step()

        preds = preds.argmax(3).flatten(1)
        label = label.argmax(3).flatten(1)
        err = 1 - torch.sum(torch.all(preds == label, dim=1)) / preds.size(0)
        tloader.set_description(
            'Epoch {} {} Loss {:.4f} Err: {:.4f}'.format(epoch, ('Train' if to_train else 'Test '), loss.item(), err))
        loss_final += loss.item()
        err_final += err

    loss_final, err_final = loss_final/len(loader), err_final/len(loader)
    with open(log_file, 'a') as f:
        pre = 'Train' if to_train else 'Test'
        print(f'{pre} epoch:{epoch}, loss={loss_final}, err={err_final}', file=f)
    if not to_train:
        print('TESTING SET RESULTS: Average loss: {:.4f} Err: {:.4f}'.format(loss_final, err_final))

    torch.cuda.empty_cache()


def train(epoch, model, optimizer, dataset, batchSz, log_file):
    run(epoch, model, optimizer, dataset, batchSz, True, log_file)


@torch.no_grad()
def test(epoch, model, optimizer, dataset, batchSz, log_file):
    run(epoch, model, optimizer, dataset, batchSz, False, log_file)


if __name__ == '__main__':
    main()