# -*- coding: utf-8 -*-

import argparse
import time

import numpy as np
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim
import torch.utils.data as data
from sklearn.ensemble import RandomForestClassifier
from torch.utils.data import DataLoader, Dataset
import decisiontree

from RCNN import RCNN
# from RNN import myRNN, LSTMClassifier
from dataset import SSTDataset_torch, loadGloveModel
from TextCNN import TextCNN


def collate_fn(data):
    """Pad data in a batch. 
    处理batch，把batch里面的tensor补到这个batch中最长的。

    Parameters
    ----------
    data : list((tensor, int), )
        data and label in a batch

    Returns
    -------
    tuple(tensor, tensor)
    
    """
    # data: [(tensor, label), ...]
    max_len = max([i[0].shape[0] for i in data])

    labels = torch.tensor([i[1] for i in data], dtype=torch.long)  # labels in this batch
    # print('labels', labels)
    padded = torch.zeros((len(data), max_len), dtype=torch.long)  # padded tensor
    # randomizing might be better

    # print('pad', padded.size())
    for i, _ in enumerate(padded):
        padded[i][:data[i][0].shape[0]] = data[i][0]

    return padded, labels

def extraction(data_iter, model, args):
    # Evaluating the given model(Extracting features from test set)
    model.eval()
    with torch.no_grad():
        # corrects = 0
        # avg_loss = 0
        # total = 0
        extracted_features = []

        def hook(module, input, output):
            extracted_features.append(input[0].cpu().data)

        if args.model_name == 'cnn':
            handle = model.bn1.register_forward_hook(hook)
        elif args.model_name == 'rcnn':
            handle = model.fc.register_forward_hook(hook)
        set_labels = []
        # x = np.zeros((len(model.Ks)*model.kernel_num))
        for data, label in data_iter:
            # assert data.shape[0] == 16
            set_labels.append(label.cpu().numpy())
            sentences = data.to(args.device, non_blocking=True)
            labels = label.to(args.device, non_blocking=True)
            _ = model(sentences)
            # torch.max(logit, 1)[1]: index
            # corrects += (torch.max(logit, 1)[1].view(labels.size()).data == labels.data).sum().item()

        # shpe = None
        # for s in extracted_features:
        #     if s.shape != shpe:
        #         print(s.shape)
        #         shpe = s.shape
        # for s in set_labels:
        #     if s.shape != shpe:
        #         print(s.shape)
        #         shpe = s.shape
        # np.hstack(set_labels)
        # size = len(data_iter.dataset)
        handle.remove()
        model.train()
        return np.vstack(extracted_features), np.hstack(set_labels)

def main():
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    parser = argparse.ArgumentParser()
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--epoch', type=int, default=2)
    parser.add_argument('--kernel_num', type=int, default=100, help='Number of each size of kernels used in CNN')
    parser.add_argument('--label_num', type=int, default=2, help='Target label numbers')
    parser.add_argument('--log_interval', type=int, default=100)
    parser.add_argument('--wordvec_dim', type=int, default=50, help='Dimension of GloVe vectors')
    parser.add_argument('--model_name', type=str, default='cnn', help='Which model to use')
    parser.add_argument('--kernel_sizes', type=str, default='3,4,5', help='Sizes of kernels used in CNN')
    parser.add_argument('--dataset_path', type=str, default='data/dataset/', help='PATH to dataset')
    parser.add_argument('--runs', type=int, default=1, help='')
    
    args = parser.parse_args()
    # torch.manual_seed(args.seed)[]

    start = time.time()
    wordvec = loadGloveModel('../midterm/data/glove/glove.6B.'+ str(args.wordvec_dim) +'d.txt') 
    args.device = device
    args.weight = torch.tensor(wordvec.values, dtype=torch.float)  # word embedding for the embedding layer
    args.kernel_sizes = [int(k) for k in args.kernel_sizes.split(',')]

    # Datasets
    training_set = SSTDataset_torch(args.dataset_path, 'train', args.label_num, args.wordvec_dim, wordvec)
    testing_set = SSTDataset_torch(args.dataset_path, 'test', args.label_num, args.wordvec_dim, wordvec)
    validation_set = SSTDataset_torch(args.dataset_path, 'dev', args.label_num, args.wordvec_dim, wordvec)

    training_iter = DataLoader(dataset=training_set,
                                    batch_size=args.batch_size,
                                    num_workers=0, shuffle=True, collate_fn=collate_fn, pin_memory=True)
    testing_iter = DataLoader(dataset=testing_set,
                                    batch_size=args.batch_size,
                                    num_workers=0, collate_fn=collate_fn, pin_memory=True)
    validation_iter = DataLoader(dataset=validation_set,
                                    batch_size=args.batch_size,
                                    num_workers=0, collate_fn=collate_fn, pin_memory=True)
    print(time.time() - start)

    model_name = args.model_name.lower()
    print(model_name)

    # Select model
    if model_name == 'cnn':
        model = TextCNN(args).to(device)
    # elif model_name == 'lstm':
    #     model = LSTMClassifier(args).to(device)
    elif model_name == 'rcnn':
        model = RCNN(args).to(device)
    # elif model_name == 'rnn':
    #     model = myRNN(args).to(device)
    else:
        print('Unrecognized model name!')
        exit(1)
    del wordvec  # Save some memory

    criterion = nn.CrossEntropyLoss()
    # optimizer = optim.SGD(model.parameters(), lr=config.lr)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)  # Adam优化器

    step = 0
    loss_sum = 0
    # Train
    for epoch in range(1, args.epoch+1):
        for data, label in training_iter:
            sentences = data.to(device, non_blocking=True)  # Asynchronous loading
            # sentences = data.flip(dims=(-1,)).to(device, dtype=torch.long)
            labels = label.to(device, non_blocking=True)
            optimizer.zero_grad()
            logits = model(sentences)  # 训练

            loss = criterion(logits, labels)  # 损失

            loss_sum += loss.data  # 求和loss
            step += 1

            if step % args.log_interval == 0:
                print("epoch", epoch, end='  ')
                print("avg loss: %.5f" % (loss_sum/args.log_interval))
                loss_sum = 0
                step = 0

            loss.backward()
            optimizer.step()

    
    # Random Forest
    train_features, train_labels = extraction(training_iter, model, args)  # shuffled so regen labels
    dev_features, dev_labels = extraction(validation_iter, model, args)  # shuffled so regen labels
    test_features, test_labels = extraction(testing_iter, model, args)  # shuffled so regen labels

    # np.save('train_features', train_features)
    # np.save('dev_features', dev_features)
    # np.save('test_features', test_features)
    # np.save('train_labels', train_features)
    # np.save('dev_labels', dev_labels)
    # np.save('test_labels', test_labels)
    best = (0, 0)
    avg = [0, 0]
    for i in range(args.runs):
        # clf = RandomForestClassifier(n_estimators=300, max_depth=10, n_jobs=-1, criterion='gini', max_features='log2')
        # clf.fit(train_features, train_labels)
        # train_acc = clf.score(train_features, train_labels)
        # test_acc = clf.score(test_features, test_labels)

        # 10d features
        # train_features = np.load('train_feature_CNN.npy')
        # test_features = np.load('test_feature_CNN.npy')
        # train_labels = np.load('train_label_for_forest.npy')
        # test_labels = np.load('test_label_for_forest.npy')

        forest = decisiontree.RandomForest(2, 9, 50)
        print(train_features.shape)
        start = time.time()
        # forest.grow(np.vstack((train_features, dev_features)), np.hstack((train_labels, dev_labels)))
        forest.grow(train_features, train_labels)
        end = time.time() - start
        print(end)
        train_acc = forest.score(train_features, train_labels)
        test_acc = forest.score(test_features, test_labels)
        if test_acc > best[1]:
            best = (train_acc, test_acc)
        avg[0] += train_acc
        avg[1] += test_acc
        print(i, ' {:.4f}|{:.4f}'.format(train_acc, test_acc))
        for tree in forest.trees:
            print(np.count_nonzero(tree.classify(test_features) == test_labels) / len(test_labels))


    print('best {:.4f}|{:.4f}'.format(best[0], best[1]))
    print('avg {:.4f}|{:.4f}'.format(avg[0]/args.runs, avg[1]/args.runs))
    print('\n{:.4f}|{:.4f}|{:.4f}|{:.4f}'.format(best[0], best[1], avg[0]/args.runs, avg[1]/args.runs))
        


    # print('best: epoch {}, acc {:.4f}'.format(best, best_acc))

    print("Parameters:")
    delattr(args, 'weight')
    for attr, value in sorted(args.__dict__.items()):
        print("\t{}={}".format(attr.upper(), value))


if __name__ == "__main__":
    main()
