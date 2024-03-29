import argparse
import os
import random
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as Data
from pytorch_transformers import *
from torch.autograd import Variable
from torch.utils.data import Dataset
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score, accuracy_score
from sklearn.neighbors import LocalOutlierFactor
from tqdm import tqdm

from read_data import *
from mixtext import MixText
from outlier_detection_model import OutlierDetectionModel
from torch.utils.tensorboard import SummaryWriter

parser = argparse.ArgumentParser(description='PyTorch MixText')

parser.add_argument('--epochs', default=50, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--batch-size', default=4, type=int, metavar='N',
                    help='train batchsize')
parser.add_argument('--batch-size-u', default=24, type=int, metavar='N',
                    help='train batchsize')

parser.add_argument('--lrmain', '--learning-rate-bert', default=0.00001, type=float,
                    metavar='LR', help='initial learning rate for bert')
parser.add_argument('--lrlast', '--learning-rate-model', default=0.001, type=float,
                    metavar='LR', help='initial learning rate for models')

parser.add_argument('--gpu', default='0,1,2,3', type=str,
                    help='id(s) for CUDA_VISIBLE_DEVICES')

parser.add_argument('--n-labeled', type=int, default=20,
                    help='number of labeled data')

parser.add_argument('--un-labeled', default=5000, type=int,
                    help='number of unlabeled data')

parser.add_argument('--val-iteration', type=int, default=200,
                    help='number of labeled data')

parser.add_argument('--mix-option', default=True, type=bool, metavar='N',
                    help='mix option, whether to mix or not')
parser.add_argument('--mix-method', default=0, type=int, metavar='N',
                    help='mix method, set different mix method')
parser.add_argument('--separate-mix', default=False, type=bool, metavar='N',
                    help='mix separate from labeled data and unlabeled data')
parser.add_argument('--co', default=False, type=bool, metavar='N',
                    help='set a random choice between mix and unmix during training')
parser.add_argument('--train_aug', default=False, type=bool, metavar='N',
                    help='augment labeled training data')

parser.add_argument('--model', type=str, default='bert-base-uncased',
                    help='pretrained model')

parser.add_argument('--data-path', type=str, default='yahoo_answers_csv/',
                    help='path to data folders')

parser.add_argument('--mix-layers-set', nargs='+',
                    default=[0, 1, 2, 3], type=int, help='define mix layer set')

parser.add_argument('--alpha', default=0.75, type=float,
                    help='alpha for beta distribution')

parser.add_argument('--lambda-u', default=1, type=float,
                    help='weight for consistency loss term of unlabeled data')
parser.add_argument('--T', default=0.5, type=float,
                    help='temperature for sharpen function')

parser.add_argument('--temp-change', default=1000000, type=int)

parser.add_argument('--margin', default=0.7, type=float, metavar='N',
                    help='margin for hinge loss')
parser.add_argument('--lambda-u-hinge', default=0, type=float,
                    help='weight for hinge loss term of unlabeled data')
parser.add_argument('--seed', default=0, type=int)
parser.add_argument('--theta-threshold', default=0.5, type=float)
parser.add_argument('--save-path', type=str, default='./model/yh_10.pt')
parser.add_argument('--pretrain-epochs', type=int, default=5)
parser.add_argument('--lrbeta', '--learning-rate-beta', default=0.0000001, type=float,
                    metavar='LR')
parser.add_argument('--use-pretrain', default=False, action='store_true')
parser.add_argument('--use-load', default=False, action='store_true')
parser.add_argument('--pretrain-inititeration', type=int, default=100)
parser.add_argument('--pretrain-afteriteration', type=int, default=100)
parser.add_argument('--global-seed', default=1024, type=int)
parser.add_argument('--check-inititeration', type=int, default=10)
parser.add_argument('--check-afteriteration', type=int, default=10)

args = parser.parse_args()

global_seed = 1226
torch.manual_seed(global_seed)
torch.cuda.manual_seed_all(global_seed)
print("Global seed: ", global_seed)
print('Check init iteration: ', args.check_inititeration)
print('Check after iteration: ', args.check_afteriteration)

print('Threshold ', args.theta_threshold)

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
n_gpu = torch.cuda.device_count()
print("GPU num: ", n_gpu)

best_acc = 0
total_steps = 0
flag = 0
optimizer_target = 'beta'
steps = 0
best_pre_acc = 0
best_pre_f1 = 0
best_pre_model_dict = {}
best_pre_out_detection_dict = {}
beta_case_study_dict = {i:[] for i in range(50)}
loss_case_study_dict = {i:[] for i in range(50)}
alpha_case_study_dict = {i:[] for i in range(50)}
abs_case_study_dict = {i:[] for i in range(50)}
actual_normal_case_study_dict = {i:[] for i in range(50)}
print('Whether mix: ', args.mix_option)
print("Mix layers sets: ", args.mix_layers_set)

import time

starttime = time.strftime("%Y-%m-%d_%H:%M:%S")  # 时间格式可以自定义，如果需要定义到分钟记得改下冒号，否则输入logdir时候会出问题
print("Start experiment:", starttime)  # 定义实验时间
writer = SummaryWriter(
    log_dir="./summary_log/" + starttime[:13] + '_' + starttime[14:16] + '_' + starttime[17:19] + '_' + str(
        args.n_labeled),
    comment=starttime,
    flush_secs=60)


def main():
    global best_acc
    global best_pre_acc
    global best_pre_f1
    global writer
    # Read dataset and build dataloaders
    '''
    train_labeled_set, train_unlabeled_set, val_set, test_set, in_distribution_test_set, n_labels, in_distribution_n_labels = get_data(
        args.data_path, args.n_labeled, args.un_labeled, model=args.model, train_aug=True, seed=args.seed)
    '''
    train_labeled_set, train_unlabeled_set, val_set, test_set, in_distribution_test_set, n_labels, in_distribution_n_labels, out_distribution_test_set, test_set_tune, val_set_tune, train_unlabeled_set_labeled = get_data_tune(
        args.data_path, args.n_labeled, args.un_labeled, model=args.model, train_aug=True, seed=args.seed)
    labeled_trainloader = Data.DataLoader(
        dataset=train_labeled_set, batch_size=args.batch_size, shuffle=True)
    unlabeled_trainloader = Data.DataLoader(
        dataset=train_unlabeled_set, batch_size=args.batch_size_u, shuffle=True)
    unlabeled_labeled_trainloader = Data.DataLoader(
        dataset=train_unlabeled_set_labeled, batch_size=args.batch_size_u, shuffle=True)
    val_loader = Data.DataLoader(
        dataset=val_set, batch_size=512, shuffle=False)
    test_loader = Data.DataLoader(
        dataset=test_set, batch_size=512, shuffle=True)
    in_distribution_test_loader = Data.DataLoader(
        dataset=in_distribution_test_set, batch_size=512, shuffle=True)
    test_tune_loader = Data.DataLoader(
        dataset=test_set_tune, batch_size=512, shuffle=True)
    val_tune_loader = Data.DataLoader(
        dataset=val_set_tune, batch_size=512, shuffle=False)
    val_tune_case_study_loader = Data.DataLoader(
        dataset=val_set_tune, batch_size=50, shuffle=False)

    print('Global seed: ', global_seed)

    # Define the model, set the optimizer
    model = MixText(in_distribution_n_labels, args.mix_option).cuda()
    model = nn.DataParallel(model)
    # EM_model = MixText(in_distribution_n_labels, args.mix_option)
    # checkpoint = torch.load(args.save_path)
    # EM_model = nn.DataParallel(EM_model).cuda()

    labeled_train_iter = iter(labeled_trainloader)

    for _ in range(5):
        try:
            inputs_x, targets_x, inputs_x_length = next(labeled_train_iter)
        except:
            labeled_train_iter = iter(labeled_trainloader)
            inputs_x, targets_x, inputs_x_length = next(labeled_train_iter)

        print(inputs_x)
        print(targets_x)

    '''
    if isinstance(EM_model, torch.nn.DataParallel):
        EM_model = EM_model.module
    '''
    '''
    if isinstance(model, torch.nn.DataParallel):
        model = model.module
    '''
    # EM_model.eval()
    # EM_model.load_state_dict(checkpoint)
    # print('Use load type is ', args.use_load)
    '''
    if args.use_load:
        model.load_state_dict(checkpoint)
    '''
    # model = nn.DataParallel(model).cuda()
    outlier_detection_model = OutlierDetectionModel().cuda()
    outlier_detection_model = nn.DataParallel(outlier_detection_model)
    optimizer_beta = AdamW(
        [
            {"params": outlier_detection_model.module.mlp.parameters(), "lr": args.lrlast}
        ])
    optimizer = AdamW(
        [
            {"params": model.module.bert.parameters(), "lr": args.lrmain},
            {"params": model.module.linear.parameters(), "lr": args.lrlast},
        ])

    num_warmup_steps = math.floor(50)
    num_total_steps = args.val_iteration

    scheduler = None
    # WarmupConstantSchedule(optimizer, warmup_steps=num_warmup_steps)

    train_criterion = SemiLoss()
    criterion = nn.CrossEntropyLoss()
    sigmoid_criterion = nn.BCELoss()

    print('Use pretrain type is ', args.use_pretrain)

    if args.use_pretrain:
        pre_train(labeled_trainloader, unlabeled_trainloader, model, outlier_detection_model, model, optimizer,
                  optimizer_beta, scheduler, criterion, sigmoid_criterion, 0, val_tune_loader, test_tune_loader,
                  in_distribution_test_loader)

        model.load_state_dict(best_pre_model_dict)
        outlier_detection_model.load_state_dict(best_pre_out_detection_dict)
        print(best_pre_out_detection_dict)
        print(outlier_detection_model.state_dict())

        # get_beta(val_tune_loader, model, outlier_detection_model, epoch)

    print('After pretrain, the metrics are below')
    in_distribution_test_loss, in_distribution_test_acc = validate(
        in_distribution_test_loader, model, outlier_detection_model, criterion, 0, mode='Test Stats ')
    out_test_err, out_test_err1, out_test_outlier_detection_err, out_test_outlier_detection_err1, out_test_acc = test_out_of_distribution_set(
        test_tune_loader, model, outlier_detection_model)
    all_acc, all_acc1, all_f1, all_f11 = test_all_set(
        test_tune_loader, model, outlier_detection_model, 0, mode='test')
    print('in acc ', in_distribution_test_acc)
    print('out test outlier detection err ', out_test_outlier_detection_err)
    print('out test outlier detection err1 ', out_test_outlier_detection_err1)
    print('out test acc ', out_test_acc)
    print('all acc ', all_acc1)
    print('all f1 ', all_f11)
    print('Best acc ', best_pre_acc)
    print('Best f1 ', best_pre_f1)

    test_accs = []
    in_distribution_test_accs = []
    out_distribution_test_errs = []
    out_distribution_test_errs1 = []
    out_distribution_test_outlier_detection_errs = []
    out_distribution_test_outlier_detection_errs1 = []
    out_test_accs = []
    all_test_accs = []
    all_test_accs1 = []
    all_test_f1s = []
    all_test_f1s1 = []

    counter = 0  # 记录没有增长的轮次数量

    optimizer = AdamW(
        [
            {"params": model.module.bert.parameters(), "lr": args.lrmain},
            {"params": model.module.linear.parameters(), "lr": args.lrlast},
        ])
    optimizer_beta = AdamW(
        [
            {"params": outlier_detection_model.module.mlp.parameters(), "lr": args.lrbeta}
        ])

    # Start training
    for epoch in tqdm(range(args.epochs)):

        if counter >= 100:
            break

        train(labeled_trainloader, unlabeled_trainloader, unlabeled_labeled_trainloader, model, outlier_detection_model,
              optimizer, optimizer_beta,
              scheduler, train_criterion, epoch, in_distribution_n_labels, args.train_aug)

        # scheduler.step()

        # _, train_acc = validate(labeled_trainloader,
        #                        model,  criterion, epoch, mode='Train Stats')
        # print("epoch {}, train acc {}".format(epoch, train_acc))
        '''
        val_loss, val_acc = validate(
            val_loader, model, criterion, epoch, mode='Valid Stats')
        '''

        val_acc, val_acc1, val_f1, val_f11 = test_all_set(val_tune_loader, model, outlier_detection_model, epoch)
        writer.add_scalar('val_acc', val_acc1, epoch)
        writer.add_scalar('val_f1', val_f11, epoch)

        validate_case_study(val_tune_case_study_loader, model, outlier_detection_model, epoch)

        '''
        print("epoch {}, val acc {}, val_loss {}".format(
            epoch, val_acc, val_loss))
        '''

        print("epoch {}, val acc {}".format(epoch, val_acc1))

        if val_acc1 >= best_acc:
            counter = 0
            best_acc = val_acc1
            test_loss, test_acc = validate(
                test_loader, model, outlier_detection_model, criterion, epoch, mode='Test Stats ')
            in_distribution_test_loss, in_distribution_test_acc = validate(
                in_distribution_test_loader, model, outlier_detection_model, criterion, epoch, mode='Test Stats ')
            test_accs.append(test_acc)
            in_distribution_test_accs.append(in_distribution_test_acc)
            out_test_err, out_test_err1, out_test_outlier_detection_err, out_test_outlier_detection_err1, out_test_acc = test_out_of_distribution_set(
                test_tune_loader, model, outlier_detection_model)
            out_distribution_test_errs.append(out_test_err)
            out_distribution_test_errs1.append(out_test_err1)
            out_distribution_test_outlier_detection_errs.append(out_test_outlier_detection_err)
            out_distribution_test_outlier_detection_errs1.append(out_test_outlier_detection_err1)
            all_acc, all_acc1, all_f1, all_f11 = test_all_set(
                test_tune_loader, model, outlier_detection_model, epoch, mode='test')
            all_test_accs.append(all_acc)
            all_test_f1s.append(all_f1)
            all_test_accs1.append(all_acc1)
            all_test_f1s1.append(all_f11)
            out_test_accs.append(out_test_acc)
            print("epoch {}, test acc {},test loss {}".format(
                epoch, test_acc, test_loss))
            print("epoch {}, test acc {},test loss {}".format(
                epoch, in_distribution_test_acc, in_distribution_test_loss))
            print("epoch {}, out distribution test err {}".format(epoch, out_test_err))
            print("epoch {}, out distribution test err1 {}".format(epoch, out_test_err1))
            print("epoch {}, out test acc {}".format(epoch, out_test_acc))
            print("epoch {}, all test acc {}".format(epoch, all_acc))
            print("epoch {}, all f1 {}".format(epoch, all_f1))
            print("epoch {}, all test acc1 {}".format(epoch, all_acc1))
            print("epoch {}, all f11 {}".format(epoch, all_f11))


        else:
            counter += 1

        print('Epoch: ', epoch)

        print('Best acc:')
        print(best_acc)

        print('Test acc:')
        print(test_accs)

        print('In_distribution test acc:')
        print(in_distribution_test_accs)

        print('Out_distribution test err:')
        print(out_distribution_test_errs)

        print('Out_distribution test err1:')
        print(out_distribution_test_errs1)

        print('Out_distribution test outlier detection err:')
        print(out_distribution_test_outlier_detection_errs)

        print('Out_distribution test outlier detection err1:')
        print(out_distribution_test_outlier_detection_errs1)

        print('Out_distribution test outlier detection acc:')
        print(out_test_accs)

        print('All test acc:')
        print(all_test_accs)

        print('All test f1:')
        print(all_test_f1s)

        print('All test acc1:')
        print(all_test_accs1)

        print('All test f11:')
        print(all_test_f1s1)

    print("Finished training!")
    print('Best acc:')
    print(best_acc)

    print('Test acc:')
    print(test_accs)

    print('In_distribution test acc:')
    print(in_distribution_test_accs)

    print('Out_distribution test err:')
    print(out_distribution_test_errs)

    print('Out_distribution test err1:')
    print(out_distribution_test_errs1)

    print('Out_distribution test outlier detection acc:')
    print(out_test_accs)

    print('Out_distribution test outlier detection err:')
    print(out_distribution_test_outlier_detection_errs)

    print('Out_distribution test outlier detection err1:')
    print(out_distribution_test_outlier_detection_errs1)

    print('All test acc:')
    print(all_test_accs)

    print('All test f1:')
    print(all_test_f1s)

    print('All test acc1:')
    print(all_test_accs1)

    print('All test f11:')
    print(all_test_f1s1)

    writer.add_scalar('global seed', global_seed, 0)

    file_loss = open(f'./file/loss_ag_{args.n_labeled}.txt', 'w')
    file_alpha = open(f'./file/alpha_ag_{args.n_labeled}.txt', 'w')
    file_beta = open(f'./file/beta_ag_{args.n_labeled}.txt', 'w')
    file_abs = open(f'./file/abs_ag_{args.n_labeled}.txt', 'w')
    file_actual_normal = open(f'./file/actual_normal_ag_{args.n_labeled}.txt', 'w')

    loss_str = str(loss_case_study_dict)
    alpha_str = str(alpha_case_study_dict)
    beta_str = str(beta_case_study_dict)
    abs_str = str(abs_case_study_dict)
    actual_normal_str = str(actual_normal_case_study_dict)

    file_loss.write(loss_str)
    file_alpha.write(alpha_str)
    file_beta.write(beta_str)
    file_abs.write(abs_str)
    file_actual_normal.write(actual_normal_str)

    file_loss.close()
    file_alpha.close()
    file_beta.close()
    file_abs.close()
    file_actual_normal.close()

    writer.close()

    print('Global_seed: ', global_seed)


def pre_train(labeled_trainloader, unlabeled_trainloader, model, outlier_detection_model, EM_model, optimizer,
              optimizer_beta,
              scheduler, criterion, sigmoid_criterion, epoch, testloader, test_tune_loader,
              in_distribution_test_loader):
    global best_pre_acc
    global best_pre_f1
    global best_pre_model_dict
    global best_pre_out_detection_dict
    model.train()
    outlier_detection_model.train()

    labeled_train_iter = iter(labeled_trainloader)
    unlabeled_train_iter = iter(unlabeled_trainloader)

    best_test_acc = 0
    best_test_f1 = 0
    best_test_in_acc = 0
    best_test_out_acc = 0
    best_test_out_err = 0
    best_test_out_err1 = 0

    global writer

    for batch_id in tqdm(range(args.pretrain_inititeration)):
        model.train()
        outlier_detection_model.train()

        try:
            inputs_x, targets_x, inputs_x_length = next(labeled_train_iter)
        except:
            labeled_train_iter = iter(labeled_trainloader)
            inputs_x, targets_x, inputs_x_length = next(labeled_train_iter)

        try:
            (inputs_u, inputs_u2, inputs_ori), (length_u,
                                                length_u2, length_ori) = next(unlabeled_train_iter)

        except:
            unlabeled_train_iter = iter(unlabeled_trainloader)
            (inputs_u, inputs_u2, inputs_ori), (length_u,
                                                length_u2, length_ori) = next(unlabeled_train_iter)

        inputs_x, targets_x = inputs_x.cuda(), targets_x.cuda(non_blocking=True)
        inputs_u = inputs_u.cuda()
        outputs_x, embedding_x = model(inputs_x)
        beta_x = outlier_detection_model(embedding_x)

        outputs_u, embedding_u = model(inputs_u)
        beta_u = outlier_detection_model(embedding_u)

        pred_u = torch.zeros([outputs_u.size(0), 1]).cuda()
        pred_x = torch.ones([outputs_x.size(0), 1]).cuda()
        bce_loss_x = sigmoid_criterion(beta_x, pred_x)
        bce_loss_u = sigmoid_criterion(beta_u, pred_u)
        ce_loss_x = criterion(outputs_x, targets_x.to(torch.int64))

        loss = ce_loss_x + bce_loss_u + bce_loss_x
        optimizer_beta.zero_grad()
        optimizer.zero_grad()
        loss.backward()
        optimizer_beta.step()
        optimizer.step()

        writer.add_scalar('loss_init', loss.item(), batch_id)
        writer.add_scalar('bce_loss_x_init', bce_loss_x.item(), batch_id)
        writer.add_scalar('bce_loss_u_init', bce_loss_u.item(), batch_id)
        writer.add_scalar('ce_loss_x_init', ce_loss_x.item(), batch_id)
        # writer.add_scalar('ce_loss_unlabeled', ce_loss.item(), epoch)

        if batch_id % args.check_inititeration == 0:
            val_acc, val_acc1, val_f1, val_f11 = test_all_set(testloader, model, outlier_detection_model, epoch,
                                                              mode='test')
            if val_acc1 >= best_pre_acc:
                best_pre_model_dict = model.state_dict()
                best_pre_out_detection_dict = outlier_detection_model.state_dict()
                best_pre_acc = val_acc1
                best_pre_f1 = val_f11
                in_distribution_test_loss, in_distribution_test_acc = validate(
                    in_distribution_test_loader, model, outlier_detection_model, criterion, 0, mode='Test Stats ')
                out_test_err, out_test_err1, out_test_outlier_detection_err, out_test_outlier_detection_err1, out_test_acc = test_out_of_distribution_set(
                    test_tune_loader, model, outlier_detection_model)
                all_acc, all_acc1, all_f1, all_f11 = test_all_set(
                    test_tune_loader, model, outlier_detection_model, 0, mode='test')
                best_test_acc = all_acc1
                best_test_f1 = all_f11
                best_test_in_acc = in_distribution_test_acc
                best_test_out_acc = out_test_acc
                best_test_out_err = out_test_outlier_detection_err
                best_test_out_err1 = out_test_outlier_detection_err1
            writer.add_scalar('val_acc_pretrain_init', val_acc1, batch_id)
            writer.add_scalar('val_f1_pretrain_init', val_f11, batch_id)

    print('Best pre test acc ', best_test_acc)
    print('Best pre test f1 ', best_test_f1)
    print('Best pre test in acc ', best_test_in_acc)
    print('Best pre test out acc ', best_test_out_acc)
    print('Best pre test out err ', best_test_out_err)
    print('Best pre test out err1 ', best_test_out_err1)

    for batch_id in tqdm(range(args.pretrain_afteriteration)):
        model.train()
        outlier_detection_model.train()
        try:
            inputs_x, targets_x, inputs_x_length = next(labeled_train_iter)
        except:
            labeled_train_iter = iter(labeled_trainloader)
            inputs_x, targets_x, inputs_x_length = next(labeled_train_iter)

        try:
            (inputs_u, inputs_u2, inputs_ori), (length_u,
                                                length_u2, length_ori) = next(unlabeled_train_iter)

        except:
            unlabeled_train_iter = iter(unlabeled_trainloader)
            (inputs_u, inputs_u2, inputs_ori), (length_u,
                                                length_u2, length_ori) = next(unlabeled_train_iter)

        inputs_x, targets_x = inputs_x.cuda(), targets_x.cuda(non_blocking=True)
        inputs_u = inputs_u.cuda()
        outputs_x, embedding_x = model(inputs_x)
        beta_x = outlier_detection_model(embedding_x)

        pred_x = torch.ones([outputs_x.size(0), 1]).cuda()
        bce_loss_x = sigmoid_criterion(beta_x, pred_x)
        ce_loss_x = criterion(outputs_x, targets_x.to(torch.int64))

        outputs_u, embedding_u = model(inputs_u)
        beta_u = outlier_detection_model(embedding_u)

        pred_u = torch.zeros([outputs_u.size(0), 1]).cuda()
        bce_loss_u = sigmoid_criterion(beta_u, pred_u)
        _, predicted = torch.max(outputs_u.data, 1)
        ce_loss_u = criterion(outputs_u, predicted.to(torch.int64))

        loss = ce_loss_x + ce_loss_u + bce_loss_u + bce_loss_x
        optimizer_beta.zero_grad()
        optimizer.zero_grad()
        loss.backward()
        optimizer_beta.step()
        optimizer.step()

        writer.add_scalar('loss_after', loss.item(), batch_id)
        writer.add_scalar('bce_loss_x_after', bce_loss_x.item(), batch_id)
        writer.add_scalar('bce_loss_u_after', bce_loss_u.item(), batch_id)
        writer.add_scalar('ce_loss_x_after', ce_loss_x.item(), batch_id)
        writer.add_scalar('ce_loss_u_after', ce_loss_u.item(), batch_id)

        if batch_id % args.check_afteriteration == 0:
            val_acc, val_acc1, val_f1, val_f11 = test_all_set(testloader, model, outlier_detection_model, epoch,
                                                              mode='test')
            if val_acc1 >= best_pre_acc:
                best_pre_model_dict = model.state_dict()
                best_pre_out_detection_dict = outlier_detection_model.state_dict()
                best_pre_acc = val_acc1
                best_pre_f1 = val_f11
                in_distribution_test_loss, in_distribution_test_acc = validate(
                    in_distribution_test_loader, model, outlier_detection_model, criterion, 0, mode='Test Stats ')
                out_test_err, out_test_err1, out_test_outlier_detection_err, out_test_outlier_detection_err1, out_test_acc = test_out_of_distribution_set(
                    test_tune_loader, model, outlier_detection_model)
                all_acc, all_acc1, all_f1, all_f11 = test_all_set(
                    test_tune_loader, model, outlier_detection_model, 0, mode='test')
                best_test_acc = all_acc1
                best_test_f1 = all_f11
                best_test_in_acc = in_distribution_test_acc
                best_test_out_acc = out_test_acc
                best_test_out_err = out_test_outlier_detection_err
                best_test_out_err1 = out_test_outlier_detection_err1
            writer.add_scalar('val_acc_pretrain_after', val_acc1, batch_id)
            writer.add_scalar('val_f1_pretrain_after', val_f11, batch_id)

    print('Best pre test acc ', best_test_acc)
    print('Best pre test f1 ', best_test_f1)
    print('Best pre test in acc ', best_test_in_acc)
    print('Best pre test out acc ', best_test_out_acc)
    print('Best pre test out err ', best_test_out_err)
    print('Best pre test out err1 ', best_test_out_err1)


def train(labeled_trainloader, unlabeled_trainloader, unlabeled_labeled_trainloader, model, outlier_detection_model,
          optimizer, optimizer_beta,
          scheduler, criterion, epoch, n_labels,
          train_aug=False):
    labeled_train_iter = iter(labeled_trainloader)
    unlabeled_train_iter = iter(unlabeled_trainloader)
    unlabeled_train_labeled_iter = iter(unlabeled_labeled_trainloader)

    model.train()
    outlier_detection_model.train()

    global total_steps
    global flag
    global optimizer_target
    global writer
    global steps
    if flag == 0 and total_steps > args.temp_change:
        print('Change T!')
        args.T = 0.9
        flag = 1

    for batch_idx in range(args.val_iteration):

        total_steps += 1
        train_aug = False

        if not train_aug:
            try:
                inputs_x, targets_x, inputs_x_length = next(labeled_train_iter)
            except:
                labeled_train_iter = iter(labeled_trainloader)
                inputs_x, targets_x, inputs_x_length = next(labeled_train_iter)
        else:
            try:
                (inputs_x, inputs_x_aug), (targets_x, _), (inputs_x_length,
                                                           inputs_x_length_aug) = next(labeled_train_iter)
            except:
                labeled_train_iter = iter(labeled_trainloader)
                (inputs_x, inputs_x_aug), (targets_x, _), (inputs_x_length,
                                                           inputs_x_length_aug) = next(labeled_train_iter)

        train_aug = True
        if train_aug:
            try:
                (inputs_u, inputs_u2, inputs_ori), (length_u,
                                                    length_u2, length_ori) = next(unlabeled_train_iter)
                inputs_u_labeled, targets_u_labeled, inputs_u_length_labeled = next(unlabeled_train_labeled_iter)
            except:
                unlabeled_train_iter = iter(unlabeled_trainloader)
                (inputs_u, inputs_u2, inputs_ori), (length_u,
                                                    length_u2, length_ori) = next(unlabeled_train_iter)
                unlabeled_train_labeled_iter = iter(unlabeled_labeled_trainloader)
                inputs_u_labeled, targets_u_labeled, inputs_u_length_labeled = next(unlabeled_train_labeled_iter)

            batch_size = inputs_x.size(0)
            batch_size_2 = inputs_ori.size(0)
            targets_x = torch.zeros(batch_size, n_labels).scatter_(
                1, targets_x.view(-1, 1), 1)

            inputs_x, targets_x = inputs_x.cuda(), targets_x.cuda(non_blocking=True)
            inputs_u = inputs_u.cuda()
            inputs_u2 = inputs_u2.cuda()
            inputs_ori = inputs_ori.cuda()

            mask = []

            with torch.no_grad():
                # Predict labels for unlabeled data.
                outputs_u, _ = model(inputs_u)
                outputs_u2, _ = model(inputs_u2)
                outputs_ori, _ = model(inputs_ori)

                # Based on translation qualities, choose different weights here.
                # For AG News: German: 1, Russian: 0, ori: 1
                # For DBPedia: German: 1, Russian: 1, ori: 1
                # For IMDB: German: 0, Russian: 0, ori: 1
                # For Yahoo Answers: German: 1, Russian: 0, ori: 1 / German: 0, Russian: 0, ori: 1
                p = (1 * torch.softmax(outputs_u, dim=1) + 0 * torch.softmax(outputs_u2,
                                                                             dim=1) + 1 * torch.softmax(outputs_ori,
                                                                                                        dim=1)) / (1)
                # Do a sharpen here.
                '''
                pt = p ** (1 / args.T)
                targets_u = pt / pt.sum(dim=1, keepdim=True)

                '''
                targets_u = p
                targets_u = targets_u.detach()



        else:
            try:
                (inputs_ori, length_ori) = next(unlabeled_train_iter)
            except:
                unlabeled_train_iter = iter(unlabeled_trainloader)
                (inputs_ori, length_ori) = next(unlabeled_train_iter)

            batch_size = inputs_x.size(0)
            batch_size_2 = inputs_ori.size(0)
            targets_x = torch.zeros(batch_size, n_labels).scatter_(
                1, targets_x.long().view(-1, 1), 1)

            inputs_x, targets_x = inputs_x.cuda(), targets_x.cuda(non_blocking=True)
            inputs_ori = inputs_ori.cuda()

            mask = []

            with torch.no_grad():
                # Predict labels for unlabeled data.
                outputs_ori, _ = model(inputs_ori)

                # Based on translation qualities, choose different weights here.
                # For AG News: German: 1, Russian: 0, ori: 1
                # For DBPedia: German: 1, Russian: 1, ori: 1
                # For IMDB: German: 0, Russian: 0, ori: 1
                # For Yahoo Answers: German: 1, Russian: 0, ori: 1 / German: 0, Russian: 0, ori: 1
                p = 1 * torch.softmax(outputs_ori, dim=1) / (1)
                # Do a sharpen here.
                '''
                pt = p ** (1 / args.T)
                targets_u = pt / pt.sum(dim=1, keepdim=True)
                '''
                targets_u = p
                targets_u = targets_u.detach()

        inputs_u_labeled, targets_u_labeled = inputs_u_labeled.cuda(), targets_u_labeled.cuda()
        mixed = 1
        if args.co:
            mix_ = np.random.choice([0, 1], 1)[0]
        else:
            mix_ = 1

        if mix_ == 1:
            l = np.random.beta(args.alpha, args.alpha)
            if args.separate_mix:
                l = l
            else:
                l = max(l, 1 - l)
        else:
            l = 1

        mix_layer = np.random.choice(args.mix_layers_set, 1)[0]
        mix_layer = mix_layer - 1

        if not train_aug:
            all_inputs = torch.cat(
                [inputs_x, inputs_ori, inputs_ori], dim=0)

            all_lengths = torch.cat(
                [inputs_x_length, length_ori, length_ori], dim=0)

            all_targets = torch.cat(
                [targets_x, targets_u, targets_u], dim=0)

        else:
            all_inputs = torch.cat(
                [inputs_x, inputs_x, inputs_u, inputs_u2, inputs_ori], dim=0)
            all_lengths = torch.cat(
                [inputs_x_length, inputs_x_length, length_u, length_u2, length_ori], dim=0)
            all_targets = torch.cat(
                [targets_x, targets_x, targets_u, targets_u, targets_u], dim=0)

        if args.separate_mix:
            idx1 = torch.randperm(batch_size)
            idx2 = torch.randperm(all_inputs.size(0) - batch_size) + batch_size
            idx = torch.cat([idx1, idx2], dim=0)

        else:
            idx1 = torch.randperm(all_inputs.size(0) - batch_size_2)
            idx2 = torch.arange(batch_size_2) + \
                   all_inputs.size(0) - batch_size_2
            idx = torch.cat([idx1, idx2], dim=0)

        input_a, input_b = all_inputs, all_inputs[idx]
        target_a, target_b = all_targets, all_targets[idx]
        length_a, length_b = all_lengths, all_lengths[idx]

        if args.mix_method == 0:
            # Mix sentences' hidden representations
            logits, sentence_embedding = model(input_a, input_b, l, mix_layer)
            mixed_target = l * target_a + (1 - l) * target_b

        elif args.mix_method == 1:
            # Concat snippet of two training sentences, the snippets are selected based on l
            # For example: "I lova you so much" and "He likes NLP" could be mixed as "He likes NLP so much".
            # The corresponding labels are mixed with coefficient as well
            mixed_input = []
            if l != 1:
                for i in range(input_a.size(0)):
                    length1 = math.floor(int(length_a[i]) * l)
                    idx1 = torch.randperm(int(length_a[i]) - length1 + 1)[0]
                    length2 = math.ceil(int(length_b[i]) * (1 - l))
                    if length1 + length2 > 256:
                        length2 = 256 - length1 - 1
                    idx2 = torch.randperm(int(length_b[i]) - length2 + 1)[0]
                    try:
                        mixed_input.append(
                            torch.cat((input_a[i][idx1: idx1 + length1], torch.tensor([102]).cuda(),
                                       input_b[i][idx2:idx2 + length2],
                                       torch.tensor([0] * (256 - 1 - length1 - length2)).cuda()), dim=0).unsqueeze(0))
                    except:
                        print(256 - 1 - length1 - length2,
                              idx2, length2, idx1, length1)

                mixed_input = torch.cat(mixed_input, dim=0)

            else:
                mixed_input = input_a

            logits, _ = model(mixed_input)
            mixed_target = l * target_a + (1 - l) * target_b

        elif args.mix_method == 2:
            # Concat two training sentences
            # The corresponding labels are averaged
            if l == 1:
                mixed_input = []
                for i in range(input_a.size(0)):
                    mixed_input.append(
                        torch.cat((input_a[i][:length_a[i]], torch.tensor([102]).cuda(), input_b[i][:length_b[i]],
                                   torch.tensor([0] * (512 - 1 - int(length_a[i]) - int(length_b[i]))).cuda()),
                                  dim=0).unsqueeze(0))

                mixed_input = torch.cat(mixed_input, dim=0)
                logits, sentence_embedding = model(mixed_input, sent_size=512)

                # mixed_target = torch.clamp(target_a + target_b, max = 1)
                mixed = 0
                mixed_target = (target_a + target_b) / 2
            else:
                mixed_input = input_a
                mixed_target = target_a
                logits, _ = model(mixed_input, sent_size=256)
                mixed = 1

        Lx, Lu, w, Lu2, w2, Lu_list = criterion(logits[:batch_size], mixed_target[:batch_size],
                                                logits[batch_size:-batch_size_2],
                                                mixed_target[batch_size:-batch_size_2], logits[-batch_size_2:],
                                                epoch + batch_idx / args.val_iteration, batch_size, mixed)

        if mix_ == 1:
            # loss = Lx + w * Lu
            loss = Lx
        else:
            # loss = Lx + w * Lu + w2 * Lu2
            # loss = Lx + w * Lu
            loss = Lx

        inputs_u = inputs_u.cuda()

        outputs_u, embedding_u = model(inputs_u)
        _, predicted = torch.max(outputs_u.data, 1)
        new_criterion = nn.CrossEntropyLoss(reduction='none')
        loss_list = new_criterion(outputs_u, predicted.to(torch.int64))
        loss_list = loss_list.unsqueeze(1)

        beta = outlier_detection_model(embedding_u)  # [20, 1]
        loss_max = torch.max(loss_list)
        abs_value = torch.abs(beta - loss_list / loss_max)
        lamda = beta.clone()  # [20, 1]
        alpha = abs_value.clone()
        lamda[torch.where(beta > 0.5)] = 0
        lamda[torch.where(beta <= 0.5)] = 1
        alpha[torch.where(abs_value > args.theta_threshold)] = 0
        alpha[torch.where(abs_value <= args.theta_threshold)] = 1
        addition_loss = calculate_pow(lamda) * calculate_pow(alpha) * beta * loss_list
        loss = loss + torch.sum(addition_loss) / beta.size(0)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        writer.add_scalar('loss by step', loss.item(), steps)
        writer.add_scalar('Lx by step', Lx.item(), steps)
        writer.add_scalar('Lu by step', Lu.item(), steps)
        writer.add_scalar('addition_loss_sum by step', torch.sum(addition_loss) / beta.size(0), steps)
        steps += 1

        if args.mix_method == 0:
            # Mix sentences' hidden representations
            logits, sentence_embedding = model(input_a, input_b, l, mix_layer)
            mixed_target = l * target_a + (1 - l) * target_b

        elif args.mix_method == 1:
            # Concat snippet of two training sentences, the snippets are selected based on l
            # For example: "I lova you so much" and "He likes NLP" could be mixed as "He likes NLP so much".
            # The corresponding labels are mixed with coefficient as well
            mixed_input = []
            if l != 1:
                for i in range(input_a.size(0)):
                    length1 = math.floor(int(length_a[i]) * l)
                    idx1 = torch.randperm(int(length_a[i]) - length1 + 1)[0]
                    length2 = math.ceil(int(length_b[i]) * (1 - l))
                    if length1 + length2 > 256:
                        length2 = 256 - length1 - 1
                    idx2 = torch.randperm(int(length_b[i]) - length2 + 1)[0]
                    try:
                        mixed_input.append(
                            torch.cat((input_a[i][idx1: idx1 + length1], torch.tensor([102]).cuda(),
                                       input_b[i][idx2:idx2 + length2],
                                       torch.tensor([0] * (256 - 1 - length1 - length2)).cuda()), dim=0).unsqueeze(0))
                    except:
                        print(256 - 1 - length1 - length2,
                              idx2, length2, idx1, length1)

                mixed_input = torch.cat(mixed_input, dim=0)

            else:
                mixed_input = input_a

            logits, _ = model(mixed_input)
            mixed_target = l * target_a + (1 - l) * target_b

        elif args.mix_method == 2:
            # Concat two training sentences
            # The corresponding labels are averaged
            if l == 1:
                mixed_input = []
                for i in range(input_a.size(0)):
                    mixed_input.append(
                        torch.cat((input_a[i][:length_a[i]], torch.tensor([102]).cuda(), input_b[i][:length_b[i]],
                                   torch.tensor([0] * (512 - 1 - int(length_a[i]) - int(length_b[i]))).cuda()),
                                  dim=0).unsqueeze(0))

                mixed_input = torch.cat(mixed_input, dim=0)
                logits, sentence_embedding = model(mixed_input, sent_size=512)

                # mixed_target = torch.clamp(target_a + target_b, max = 1)
                mixed = 0
                mixed_target = (target_a + target_b) / 2
            else:
                mixed_input = input_a
                mixed_target = target_a
                logits, _ = model(mixed_input, sent_size=256)
                mixed = 1

        Lx, Lu, w, Lu2, w2, Lu_list = criterion(logits[:batch_size], mixed_target[:batch_size],
                                                logits[batch_size:-batch_size_2],
                                                mixed_target[batch_size:-batch_size_2], logits[-batch_size_2:],
                                                epoch + batch_idx / args.val_iteration, batch_size, mixed)

        outputs_u, embedding_u = model(inputs_u)
        _, predicted = torch.max(outputs_u.data, 1)
        new_criterion = nn.CrossEntropyLoss(reduction='none')
        loss_list = new_criterion(outputs_u, predicted.to(torch.int64))
        loss_list = loss_list.unsqueeze(1)

        if mix_ == 1:
            # loss = Lx + w * Lu
            loss = Lx
        else:
            # loss = Lx + w * Lu + w2 * Lu2
            # loss = Lx + w * Lu
            loss = Lx

        beta = outlier_detection_model(embedding_u)  # [20, 1]
        loss_max = torch.max(loss_list)
        abs_value = torch.abs(beta - loss_list / loss_max)
        lamda = beta.clone()  # [20, 1]
        alpha = abs_value.clone()
        lamda[torch.where(beta > 0.5)] = 0
        lamda[torch.where(beta <= 0.5)] = 1
        alpha[torch.where(abs_value > args.theta_threshold)] = 0
        alpha[torch.where(abs_value <= args.theta_threshold)] = 1
        addition_loss = calculate_pow(lamda) * calculate_pow(alpha) * beta * loss_list
        loss = - loss + torch.sum(addition_loss) / beta.size(0)

        optimizer_beta.zero_grad()
        loss.backward()
        optimizer_beta.step()

        # max_grad_norm = 1.0
        # torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        # scheduler.step()

        if batch_idx % 1000000 == 0:
            print("epoch {}, step {}, loss {}, Lx {}, loss_list {}".format(
                epoch, batch_idx, loss.item(), Lx.item(), loss_list))
            print("the mix_method is {}, mixed is {}, mix_ is {}".format(args.mix_method, mixed, mix_))
            writer.add_scalar('loss', loss.item(), epoch)
            writer.add_scalar('Lx', Lx.item(), epoch)
            writer.add_scalar('Lu', Lu.item(), epoch)
            writer.add_scalar('Lu2', Lu.item(), epoch)

        writer.add_scalar('loss by step', loss.item(), steps)
        writer.add_scalar('Lx by step', Lx.item(), steps)
        writer.add_scalar('Lu by step', Lu.item(), steps)
        writer.add_scalar('addition_loss_sum by step', torch.sum(addition_loss) / beta.size(0), steps)
        steps += 1


def validate(valloader, model, outlier_detection_model, criterion, epoch, mode):
    model.eval()
    outlier_detection_model.eval()
    with torch.no_grad():
        loss_total = 0
        total_sample = 0
        acc_total = 0
        correct = 0

        for batch_idx, (inputs, targets, length) in enumerate(valloader):
            inputs, targets = inputs.cuda(), targets.cuda(non_blocking=True)
            outputs, _ = model(inputs)
            # loss = criterion(outputs, targets.to(torch.int64))

            _, predicted = torch.max(outputs.data, 1)

            if batch_idx == 0 and mode == 'val':
                if mode != 'Pre':
                    print("Sample some true labeles and predicted labels")
                    print(predicted[:20])
                    print(targets[:20])

            correct += (np.array(predicted.cpu()) ==
                        np.array(targets.cpu())).sum()
            # loss_total += loss.item() * inputs.shape[0]
            total_sample += inputs.shape[0]

        acc_total = correct / total_sample
        loss_total = loss_total / total_sample

    return loss_total, acc_total


def test_out_of_distribution_set(testloader, model, outlier_detection_model):
    model.eval()
    outlier_detection_model.eval()

    with torch.no_grad():
        correct = 0
        total_sample = 0
        wrong = 0
        wrong1 = 0
        wrong_out_detection = 0
        wrong_out_detection1 = 0
        total_in = 0
        total_out = 0

        for batch_idx, (inputs, targets, targets_tune, length) in enumerate(testloader):
            inputs, targets = inputs.cuda(), targets.cuda(non_blocking=True)
            targets_tune = targets_tune.cuda()  # 这里面都是0和1
            outputs, embedding = model(inputs)
            # clf = LocalOutlierFactor()
            # predicted = clf.fit_predict(embedding.cpu())
            predicted_out_detection = outlier_detection_model(embedding)
            predicted_out_detection = predicted_out_detection.view(1, -1)
            # 这里做了修改，predicted现在预测结果是-1和1，-1代表out，1代表in
            # 而targets_tune中是0和1，因此需要把0给调成-1然后计算accuracy
            targets_tune[torch.where(targets_tune.eq(0))] = -1
            '''
            correct += (np.array(predicted) ==
                        np.array(targets_tune.cpu())).sum()
            '''

            '''
            for idx, data in enumerate(np.array(predicted)):
                if data == 1 and np.array(targets_tune.cpu())[idx] == -1:
                    wrong += 1
            for idx, data in enumerate(np.array(predicted)):
                if data == -1 and np.array(targets_tune.cpu())[idx] == 1:
                    wrong1 += 1
            '''
            # 当预测值是1但是目标值为-1时，错误的个数为不带1的，反之为带1的
            predicted_array = np.array(predicted_out_detection.cpu().detach().numpy().squeeze())
            predicted_out_detection_array = predicted_out_detection.cpu().detach().numpy().squeeze()
            targets_tune_array = np.array(targets_tune.cpu())
            wrong += np.sum(predicted_array[np.where(targets_tune_array == -1)] == 1)
            wrong1 += np.sum(predicted_array[np.where(targets_tune_array == 1)] == -1)
            wrong_out_detection += np.sum(predicted_out_detection_array[np.where(targets_tune_array == -1)] > 0.5)
            wrong_out_detection1 += np.sum(predicted_out_detection_array[np.where(targets_tune_array == 1)] <= 0.5)

            predicted_out_detection_array[np.where(predicted_out_detection_array > 0.5)] = 1
            predicted_out_detection_array[np.where(predicted_out_detection_array <= 0.5)] = -1
            correct += (predicted_out_detection_array == targets_tune_array).sum()

            total_sample += inputs.shape[0]
            total_in += np.sum(targets_tune_array != -1)
            total_out += np.sum(targets_tune_array == -1)
        err_total = wrong / total_sample
        err_total1 = wrong1 / total_sample
        err_total_out_detection = wrong_out_detection / total_out
        err_total_out_detection1 = wrong_out_detection1 / total_in
        acc_total = correct / total_sample

    return err_total, err_total1, err_total_out_detection, err_total_out_detection1, acc_total


def test_all_set(testloader, model, outlier_detection_model, epoch, mode='val'):
    model.eval()
    outlier_detection_model.eval()

    global writer

    with torch.no_grad():
        loss_total = 0
        total_sample = 0
        acc_total = 0
        correct = 0
        correct1 = 0
        pred_list = []
        target_list = []
        pred_list_out_detection = []
        target_list_out_detection = []
        for batch_idx, (inputs, targets, targets_tune, length) in enumerate(testloader):
            inputs, targets = inputs.cuda(), targets.cuda(non_blocking=True)
            targets_tune = targets_tune.cuda()
            # 注意这里增加了置信度获取，到时候测试是否正确
            outputs, embedding = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            # predicted是对每一个样例生成一个in的预测

            predicted1 = torch.clone(predicted)

            # clf = LocalOutlierFactor()
            # predicted_out = clf.fit_predict(embedding.cpu())
            predicted_out_detection = outlier_detection_model(embedding)  # [batch_size, 1]
            if batch_idx == 0 and mode == 'val':
                print("Sample some true labels and predicted labels to see the beta from val or test dataset")
                for i in range(20):
                    writer.add_scalar(f'predict_out_detection {i}', predicted_out_detection[i].item(), epoch)
                print(predicted_out_detection[:20])
                print(targets[:20])
            predicted_out_detection = predicted_out_detection.view(1, -1).squeeze()

            # 下面这是为了把预测结果和真实结果中属于out的部分都标成-1
            '''
            for idx, _ in enumerate(predicted_out):
                if predicted_out[idx] != 1:  # predicted_out
                    predicted[idx] = -1
            '''
            # predicted[torch.where(predicted_out_detection.lt(0.5))] = -1
            predicted1[torch.where(predicted_out_detection.le(0.5))] = -1

            '''
            for idx, _ in enumerate(targets):
                if targets_tune[idx] == 0:
                    targets[idx] = -1
            '''
            # targets已经修改成-1代表out了，这里应该不需要通过循环调整targets的值为-1了
            '''
            if batch_idx == 0:
                print("Sample some true labels and predicted labels")
                print(predicted[:20])
                print(targets[:20])
            '''
            correct += (np.array(predicted1.cpu()) ==
                        np.array(targets.cpu())).sum()
            correct1 += (np.array(predicted1.cpu()) ==
                         np.array(targets.cpu())).sum()
            total_sample += inputs.shape[0]
            pred_list.extend(np.array(predicted1.cpu()))
            target_list.extend(np.array(targets.cpu()))
            pred_list_out_detection.extend(np.array(predicted1.cpu()))
            target_list_out_detection.extend(np.array(targets.cpu()))

        acc_total = correct / total_sample
        f1 = f1_score(target_list, pred_list, average='macro')
        f1_out_detection = f1_score(target_list_out_detection, pred_list_out_detection, average='macro')

        recall = recall_score(target_list, pred_list, average='macro')
        # precis = precision_score(target_list, pred_list, average='macro')
        accuracy = accuracy_score(target_list, pred_list)
        accuracy_out_detection = accuracy_score(target_list_out_detection, pred_list_out_detection)

    return accuracy, accuracy_out_detection, f1, f1_out_detection


def validate_case_study(valloader, model, outlier_detection_model, epoch):
    model.eval()
    outlier_detection_model.eval()

    global writer

    with torch.no_grad():

        for batch_idx, (inputs, targets, targets_tune, length) in enumerate(valloader):
            inputs, targets = inputs.cuda(), targets.cuda(non_blocking=True)
            targets_tune = targets_tune.cuda()
            # 注意这里增加了置信度获取，到时候测试是否正确
            outputs, embedding = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            # predicted是对每一个样例生成一个in的预测

            predicted1 = torch.clone(predicted)

            # clf = LocalOutlierFactor()
            # predicted_out = clf.fit_predict(embedding.cpu())
            predicted_out_detection = outlier_detection_model(embedding)  # [batch_size, 1]
            predicted_out_detection = predicted_out_detection.view(1, -1).squeeze()

            # predicted[torch.where(predicted_out_detection.lt(0.5))] = -1
            predicted1[torch.where(predicted_out_detection.le(0.5))] = -1

            predicted1[torch.where(predicted1.ne(-1))] = 1  # 把预测结果的不等于-1的调成1，也就是预测属于IN的
            predicted1[torch.where(predicted1.eq(-1))] = 0  # 把预测结果等于-1的调成0，也就是预测属于OUT的

            actual_normal_information_list = torch.ones(predicted1.size())
            actual_normal_information_list[torch.where(targets_tune == predicted1)] = 1
            actual_normal_information_list[torch.where(targets_tune != predicted1)] = 0

            new_criterion = nn.CrossEntropyLoss(reduction='none')
            loss_list = new_criterion(outputs, predicted.to(torch.int64))
            # TODO: 这里算验证集的loss应该用实际的预测结果作为target还是用数据自带的？目前决定是用预测结果和训练过程维持一致
            loss_list = loss_list.unsqueeze(1)

            beta = outlier_detection_model(embedding)
            loss_max = torch.max(loss_list)
            abs_value = torch.abs(beta - loss_list / loss_max)
            lamda = beta.clone()
            alpha = abs_value.clone()
            lamda[torch.where(beta > 0.5)] = 0
            lamda[torch.where(beta <= 0.5)] = 1
            alpha[torch.where(abs_value > args.theta_threshold)] = 0
            alpha[torch.where(abs_value <= args.theta_threshold)] = 1  # [batch_size, 1]

            for i in range(50):
                loss_case_study_dict[i].append(loss_list[i].item())
                beta_case_study_dict[i].append(beta[i].item())
                alpha_case_study_dict[i].append(alpha[i].item())
                abs_case_study_dict[i].append(abs_value[i].item())
                actual_normal_case_study_dict[i].append(actual_normal_information_list[i].item())
            break


def get_beta(testloader, model, outlier_detection_model, epoch):
    model.eval()
    outlier_detection_model.eval()

    for batch_idx, (inputs, targets, targets_tune, length) in enumerate(testloader):
        inputs, targets = inputs.cuda(), targets.cuda(non_blocking=True)
        targets_tune = targets_tune.cuda()
        # 注意这里增加了置信度获取，到时候测试是否正确
        outputs, embedding = model(inputs)
        _, predicted = torch.max(outputs.data, 1)
        predicted_out_detection = outlier_detection_model(embedding)  # [batch_size, 1]
        if batch_idx == 0:
            print("Sample some true labels and predicted labels to see the beta from val or test dataset")
            for i in range(20):
                writer.add_scalar(f'predict_out_detection_pretrain {i}', predicted_out_detection[i].item(), epoch)
            print(predicted_out_detection[:20])
            print(targets[:20])


def linear_rampup(current, rampup_length=args.epochs):
    if rampup_length == 0:
        return 1.0
    else:
        current = np.clip(current / rampup_length, 0.0, 1.0)
        return float(current)


def calculate_pow(x):
    return torch.pow(-1, x)


def normalize_entropy_Q(probility, lam):  # [batch_size, num_labels]
    # 将概率中的所有0全部换成1，算出来的结果是一样的，至少我这么觉得
    probility = torch.clip(probility, min=1e-20)
    entropy = -probility * torch.log2(probility)
    entropy = torch.sum(entropy, dim=1)
    entropy = entropy / math.log2(probility.size(1))
    entropy = torch.pow(entropy, lam)
    return entropy, 1 - entropy


class SemiLoss(object):
    def __call__(self, outputs_x, targets_x, outputs_u, targets_u, outputs_u_2, epoch, batch_size, mixed=1):

        if args.mix_method == 0 or args.mix_method == 1:

            Lx = - \
                torch.mean(torch.sum(F.log_softmax(
                    outputs_x, dim=1) * targets_x, dim=1))

            probs_u = torch.softmax(outputs_u, dim=1)

            Lu = F.kl_div(probs_u.log(), targets_u, None, None, 'batchmean')

            Lu_list = F.kl_div(probs_u.log(), targets_u, None, None, 'none')
            Lu_list = torch.sum(Lu_list, dim=1)
            Lu_list = Lu_list.unsqueeze(1)

            Lu2 = torch.mean(torch.clamp(torch.sum(-F.softmax(outputs_u, dim=1)
                                                   * F.log_softmax(outputs_u, dim=1), dim=1) - args.margin, min=0))

        elif args.mix_method == 2:
            if mixed == 0:
                Lx = - \
                    torch.mean(torch.sum(F.logsigmoid(
                        outputs_x) * targets_x, dim=1))

                probs_u = torch.softmax(outputs_u, dim=1)

                Lu = F.kl_div(probs_u.log(), targets_u,
                              None, None, 'batchmean')

                Lu2 = torch.mean(torch.clamp(args.margin - torch.sum(
                    F.softmax(outputs_u_2, dim=1) * F.softmax(outputs_u_2, dim=1), dim=1), min=0))
            else:
                Lx = - \
                    torch.mean(torch.sum(F.log_softmax(
                        outputs_x, dim=1) * targets_x, dim=1))

                probs_u = torch.softmax(outputs_u, dim=1)
                Lu = F.kl_div(probs_u.log(), targets_u,
                              None, None, 'batchmean')

                Lu2 = torch.mean(torch.clamp(args.margin - torch.sum(
                    F.softmax(outputs_u, dim=1) * F.softmax(outputs_u, dim=1), dim=1), min=0))

        return Lx, Lu, args.lambda_u * linear_rampup(epoch), Lu2, args.lambda_u_hinge * linear_rampup(epoch), Lu_list


if __name__ == '__main__':
    main()
