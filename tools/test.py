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
from core.function import testval
from utils.utils import create_logger


def parse_args():
    parser = argparse.ArgumentParser(description='Train segmentation network')

    parser.add_argument('--path',
                        default='F:/spyder_code/HSIseg/data/')
    parser.add_argument('--output_dir',
                        default='output', type=str)
    parser.add_argument('--log_dir',
                        default='log', type=str)
    parser.add_argument('--model_file',
                        default='../parameter/best25.pth', type=str)

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
    parser.add_argument('--test_row_size',
                        default=2, type=int)

    parser.add_argument('--exp_name',
                        default='hsicity')

    args = parser.parse_args()

    return args


def main():
    args = parse_args()
    logger, final_output_dir, _ = create_logger(
        args.output_dir, 'hsicity', args.model, args.log_dir, args.exp_name, 'test'
    )

    # build model
    model = eval('models.' + args.model + '.' +
                 args.model_name)(num_classes=args.num_classes)

    if args.model_file:
        model_state_file = args.model_file
    else:
        model_state_file = os.path.join(final_output_dir,
                                        'final_state.pth')
    logger.info('=> loading model from {}'.format(model_state_file))

    pretrained_dict = torch.load(model_state_file)
    model_dict = model.state_dict()
    pretrained_dict = {k: v for k, v in pretrained_dict.items()
                       if k in model_dict.keys()}
    for k, _ in pretrained_dict.items():
        logger.info(
            '=> loading {} from pretrained model'.format(k))
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)
    model.cuda()

    # prepare data
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

    start = timeit.default_timer()
    mean_IoU, IoU_array, pixel_acc, mean_acc = testval(args.num_classes,
                                                       args.ignore_label,
                                                       args.test_row_size,
                                                       test_dataset,
                                                       testloader,
                                                       model,
                                                       sv_pred=True,
                                                       sv_dir='../result'
                                                       )

    msg = 'MeanIU: {: 4.4f}, Pixel_Acc: {: 4.4f}, \
        Mean_Acc: {: 4.4f}, Class IoU: '.format(mean_IoU,
                                                pixel_acc, mean_acc)
    logging.info(msg)
    logging.info(IoU_array)

    end = timeit.default_timer()
    logger.info('Mins: %d' % np.int((end - start) / 60))
    logger.info('Done')


if __name__ == '__main__':
    main()
