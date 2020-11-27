import argparse
import os
import timeit
import logging
import numpy as np

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim
from torch.utils.data.distributed import DistributedSampler
from tensorboardX import SummaryWriter

import models
import datasets
from core.criterion import CrossEntropy
from core.function import train, validate
from utils.utils import create_logger


def parse_args():
    parser = argparse.ArgumentParser(description='Train segmentation network')

    parser.add_argument('--path',
                        default='F:/spyder_code/HSIseg/data/')
    parser.add_argument('--output_dir',
                        default='output', type=str)
    parser.add_argument('--log_dir',
                        default='log', type=str)

    parser.add_argument('--model',
                        default='resnet')
    parser.add_argument('--model_name',
                        default='resnet50')
    parser.add_argument('--resume',
                        default=False)
    parser.add_argument('--num_classes',
                        default=9, type=int)
    parser.add_argument('--ignore_label',
                        default=-1, type=int)
    parser.add_argument('--learning_rate',
                        default=0.001, type=float)
    parser.add_argument('--batch_size',
                        default=6, type=int)
    parser.add_argument('--epoch',
                        default=30, type=int)
    parser.add_argument('--print_freq',
                        default=10, type=int)

    parser.add_argument('--exp_name',
                        default='hsicity')

    args = parser.parse_args()

    return args


def main():
    args = parse_args()
    logger, final_output_dir, log_dir = create_logger(
        args.output_dir, 'hsicity', args.model, args.log_dir, args.exp_name, 'train'
    )

    writer_dict = {
        'writer': SummaryWriter(args.log_dir),
        'train_global_steps': 0,
        'valid_global_steps': 0,
    }

    # build model
    model = eval('models.' + args.model + '.' +
                 args.model_name)(num_classes=args.num_classes)

    # prepare data
    crop_size = (1773, 1379)
    train_dataset = eval('datasets.hsicity')(
        root='F:/database/HSIcityscapes/',
        list_path=args.path + 'list/hsicity/train.lst',
        num_samples=None,
        num_classes=args.num_classes,
        multi_scale=False,
        flip=True,
        ignore_label=args.ignore_label,
        base_size=1773,
        crop_size=crop_size,
        scale_factor=10
    )

    trainloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        drop_last=False)

    test_size = (1773, 1379)
    test_dataset = eval('datasets.hsicity')(
        root='F:/database/HSIcityscapes/',
        list_path=args.path + 'list/hsicity/val.lst',
        num_samples=None,
        num_classes=args.num_classes,
        multi_scale=False,
        flip=False,
        ignore_label=args.ignore_label,
        base_size=1773,
        crop_size=test_size,
        center_crop_test=False,
        downsample_rate=1)

    testloader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0)

    criterion = nn.CrossEntropyLoss(weight=train_dataset.class_weights,
                                    ignore_index=args.ignore_label)

    model.cuda()

    optimizer = torch.optim.SGD([{'params':
                                filter(lambda p: p.requires_grad,
                                 model.parameters()),
                                'lr': args.learning_rate}],
                                lr=args.learning_rate,
                                momentum=0.9,
                                weight_decay=0.0005,
                                nesterov=False,
                                )

    best_mIoU = 0
    last_epoch = 0
    if args.resume:
        model_state_file = os.path.join(final_output_dir,
                                        'checkpoint.pth.tar')
        if os.path.isfile(model_state_file):
            checkpoint = torch.load(model_state_file,
                                    map_location=lambda storage, loc: storage)
            best_mIoU = checkpoint['best_mIoU']
            last_epoch = checkpoint['epoch']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            logger.info("=> loaded checkpoint (epoch {})"
                        .format(checkpoint['epoch']))

    epoch_iters = np.int(train_dataset.__len__() / args.batch_size)
    start = timeit.default_timer()
    end_epoch = args.epoch

    for epoch in range(last_epoch, end_epoch):
        train(epoch, end_epoch, args.print_freq,
              epoch_iters, args.learning_rate,
              trainloader, optimizer, criterion, model, writer_dict)

        valid_loss, mean_IoU, IoU_array = validate(args.num_classes, args.ignore_label,
                                                   testloader, criterion,
                                                   model, writer_dict)

        logger.info('=> saving checkpoint to {}'.format(
            final_output_dir + 'checkpoint.pth.tar'))
        torch.save({
            'epoch': epoch + 1,
            'best_mIoU': best_mIoU,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
        }, os.path.join(final_output_dir, 'checkpoint.pth.tar'))

        if mean_IoU > best_mIoU:
            best_mIoU = mean_IoU
            torch.save(model.state_dict(),
                       os.path.join(final_output_dir, 'best.pth'))
        msg = 'Loss: {:.3f}, MeanIU: {: 4.4f}, Best_mIoU: {: 4.4f}'.format(
            valid_loss, mean_IoU, best_mIoU)
        logging.info(msg)
        logging.info(IoU_array)

        torch.save(model.state_dict(),
                   os.path.join(final_output_dir, 'final_state.pth'))

        if epoch == end_epoch - 1:
            writer_dict['writer'].close()
            end = timeit.default_timer()
            logger.info('Hours: %d' % np.int((end - start) / 3600))
            logger.info('Done')


if __name__ == '__main__':
    main()
