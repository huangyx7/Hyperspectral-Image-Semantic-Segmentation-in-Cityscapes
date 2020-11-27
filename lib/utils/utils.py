import os
import logging
import time
import numpy as np
from pathlib import Path


def create_logger(output_dir, dataset, model_name, log_dir, cfg_name, phase='train'):
    root_output_dir = Path(output_dir)
    # set up logger
    if not root_output_dir.exists():
        print('=> creating {}'.format(root_output_dir))
        root_output_dir.mkdir()

    dataset = dataset
    model = model_name
    cfg_name = cfg_name

    final_output_dir = root_output_dir / dataset / cfg_name

    print('=> creating {}'.format(final_output_dir))
    final_output_dir.mkdir(parents=True, exist_ok=True)

    time_str = time.strftime('%Y-%m-%d-%H-%M')
    log_file = '{}_{}_{}.log'.format(cfg_name, time_str, phase)
    final_log_file = final_output_dir / log_file
    head = '%(asctime)-15s %(message)s'
    logging.basicConfig(filename=str(final_log_file),
                        format=head)
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    console = logging.StreamHandler()
    logging.getLogger('').addHandler(console)

    tensorboard_log_dir = Path(log_dir) / dataset / model / \
                          (cfg_name + '_' + time_str)
    print('=> creating {}'.format(tensorboard_log_dir))
    tensorboard_log_dir.mkdir(parents=True, exist_ok=True)

    return logger, str(final_output_dir), str(tensorboard_log_dir)


def get_confusion_matrix(label, pred, size, num_class, ignore=-1):
    """
    Calcute the confusion matrix by given label and pred
    """
    output = pred.cpu().numpy()
    seg_pred = np.asarray(np.argmax(output, axis=2), dtype=np.uint8) + 1
    # seg_pred = confidence_label_sort(output)
    seg_gt = np.asarray(
        label.cpu().numpy(), dtype=np.int).squeeze()

    ignore_index = seg_gt != ignore
    seg_gt = seg_gt[ignore_index]
    seg_pred = seg_pred[ignore_index]

    index = (seg_gt * num_class + seg_pred).astype('int32')
    label_count = np.bincount(index)
    confusion_matrix = np.zeros((num_class, num_class))

    for i_label in range(num_class):
        for i_pred in range(num_class):
            cur_index = i_label * num_class + i_pred
            if cur_index < len(label_count):
                confusion_matrix[i_label,
                                 i_pred] = label_count[cur_index]
    return confusion_matrix


def get_confusion_matrix_1d(label, pred, size, num_class, ignore=-1):
    output = pred.cpu().numpy()
    seg_pred = np.asarray(np.argmax(output, axis=1), dtype=np.uint8)
    seg_gt = np.asarray(
        label.cpu().numpy(), dtype=np.int)

    ignore_index = seg_gt != ignore
    seg_gt = seg_gt[ignore_index]
    seg_pred = seg_pred[ignore_index]

    index = (seg_gt * num_class + seg_pred).astype('int32')
    label_count = np.bincount(index)
    confusion_matrix = np.zeros((num_class, num_class))

    for i_label in range(num_class):
        for i_pred in range(num_class):
            cur_index = i_label * num_class + i_pred
            if cur_index < len(label_count):
                confusion_matrix[i_label,
                                 i_pred] = label_count[cur_index]
    return confusion_matrix


def adjust_learning_rate(optimizer, base_lr, max_iters,
                         cur_iters, power=0.9):
    lr = base_lr * ((1 - float(cur_iters) / max_iters) ** (power))
    optimizer.param_groups[0]['lr'] = lr
    return lr


def padWithZeros(X, margin=2):
    newX = np.zeros((X.shape[0] + 2 * margin, X.shape[1] + 2 * margin, X.shape[2]))
    x_offset = margin
    y_offset = margin
    newX[x_offset:X.shape[0] + x_offset, y_offset:X.shape[1] + y_offset, :] = X
    return newX


def createImageCubes(X, y, windowSize, cubeSize, removeZeroLabels=True):
    margin = int((windowSize - 1) / 2)
    zeroPaddedX = padWithZeros(X, margin=margin)
    # split patches
    patchesData = np.zeros((cubeSize, windowSize, windowSize, X.shape[2]), dtype=np.float32)
    patchesLabels = np.zeros(cubeSize, dtype=np.int32)
    patchIndex = 0
    for r in range(margin, zeroPaddedX.shape[0] - margin):
        r = np.random.randint(margin, zeroPaddedX.shape[0] - margin)  # 内存有限，随机选取行
        for c in range(256):

            c = np.random.randint(margin, zeroPaddedX.shape[1] - margin)  # 内存有限，随机选取列

            patch = zeroPaddedX[r - margin:r + margin + 1, c - margin:c + margin + 1]
            patchesData[patchIndex, :, :, :] = patch
            patchesLabels[patchIndex] = y[r - margin, c - margin]
            if patchesLabels[patchIndex] != 0:
                patchIndex += 1

            if patchIndex == cubeSize:  # 跳出双循环
                break
        else:
            continue
        break
    if removeZeroLabels:
        patchesData = patchesData[patchesLabels > 0, :, :, :]
        patchesLabels = patchesLabels[patchesLabels > 0]
        patchesLabels -= 1
    return patchesData, patchesLabels


def createTestCube(x, y, windowSize, r, size):
    margin = int((windowSize - 1) / 2)
    zeroPaddedX = padWithZeros(x, margin=margin)
    # split patches
    patchesData = np.zeros((x.shape[1] * size, windowSize, windowSize, x.shape[2]), dtype=np.float32)
    patchesLabels = np.zeros(x.shape[1] * size, dtype=np.int32)
    patchIndex = 0

    r = r + margin
    for row in range(r, r + size):
        for c in range(margin, zeroPaddedX.shape[1] - margin):
            patch = zeroPaddedX[row - margin:row + margin + 1, c - margin:c + margin + 1]
            patchesData[patchIndex, :, :, :] = patch
            patchesLabels[patchIndex] = y[row - margin, c - margin]
            patchIndex += 1
    return patchesData, patchesLabels


def covertBatch2TrainCubes(x, y, **kwargs):
    batchSize = x.shape[0]
    for i in range(batchSize):
        imageCube, labelCube = createImageCubes(x[i], y[i], windowSize=11, cubeSize=10000)
        if i == 0:
            imageCubes = imageCube
            labelCubes = labelCube
        else:
            imageCubes = np.concatenate((imageCubes, imageCube))
            labelCubes = np.concatenate((labelCubes, labelCube))
    return imageCubes, labelCubes


def confidence_label_softmax(output, threshold=0.7):
    h, w, c = output.shape
    softmax_result = np.zeros((h, w), dtype=np.uint8)

    for i in range(h):
        for j in range(w):
            max_index = np.argmax(output[i, j, :])  # without background
            softmax_num = np.exp(output[i, j, max_index]) / np.sum(np.exp(output[i, j, :]))
            if softmax_num > threshold:
                softmax_result[i, j] = max_index + 1
            else:
                softmax_result[i, j] = 0

    return softmax_result


def confidence_label_sort(output, threshold=0.7):
    h, w, c = output.shape
    softmax_result = np.zeros((h, w), dtype=np.uint8)
    preds = np.asarray(np.argmax(output, axis=2), dtype=np.uint8)

    for k in range(0, c):
        class_map = output[:, :, k][preds == k]
        num = len(class_map)
        if num != 0:
            thre = np.sort(class_map)[-np.int(num * threshold)]
            for i in range(h):
                for j in range(w):
                    if preds[i, j] == k and output[i, j, k] >= thre:
                        softmax_result[i, j] = k + 1

    return softmax_result


def weight_log():
    class_num = [6.75299810e+07, 1.14756730e+07, 1.79191230e+08, 2.21596790e+07,
                 2.05724740e+07, 1.72913843e+08, 6.25138180e+07, 1.72078069e+08,
                 4.94832920e+07]
    weights = 1 / np.log1p(class_num)
    weights = 9 * weights / np.sum(weights)
    # [0.98715601, 1.09478434, 0.93646407, 1.05219084, 1.05683465, 0.93822461, 0.99140051, 0.93846433, 1.00448063]
    return weights
