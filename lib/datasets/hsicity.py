import os

import cv2
import numpy as np
from PIL import Image

import torch
from torch.nn import functional as F

from .base_dataset import BaseDataset
from utils.utils import confidence_label_sort, confidence_label_softmax


class hsicity(BaseDataset):
    def __init__(self,
                 root,
                 list_path,
                 num_samples=None,
                 num_classes=10,
                 multi_scale=True,
                 flip=True,
                 ignore_label=0,
                 base_size=1379,
                 crop_size=(1773, 1379),
                 center_crop_test=False,
                 downsample_rate=1,
                 scale_factor=16):
        super(hsicity, self).__init__(ignore_label, base_size,
                                      crop_size, downsample_rate, scale_factor, )
        self.root = root
        self.list_path = list_path
        self.num_classes = num_classes
        self.class_weights = torch.FloatTensor([1, 1, 1, 1, 1, 1, 1, 1, 1]).cuda()
        self.label_mapping = {0: 0,  # _background_
                              1: 1,  # car
                              2: 2,  # human
                              3: 3,  # road
                              4: 4,  # traffic light
                              5: 5,  # traffic sign
                              6: 6,  # tree
                              7: 7,  # building
                              8: 8,  # sky
                              9: 9  # object
                              }

        self.multi_scale = multi_scale
        self.flip = flip
        self.center_crop_test = center_crop_test

        self.img_list = [line.strip().split() for line in open(list_path)]
        self.files = self.read_files()

        self.hsicity_label = [
            (0, (0, 0, 0)),  # background
            (1, (0, 0, 142)),  # car
            (2, (220, 20, 60)),  # human
            (3, (128, 64, 128)),  # road
            (4, (250, 170, 30)),  # traffic light
            (5, (220, 220, 0)),  # traffic sign
            (6, (107, 142, 35)),  # tree
            (7, (70, 70, 70)),  # building
            (8, (70, 130, 180)),  # sky
            (9, (190, 153, 153)),  # object
        ]

    def read_files(self):
        files = []
        if 'test' in self.list_path:
            for item in self.img_list:
                image_path = item
                name = os.path.splitext(os.path.basename(image_path[0]))[0]
                files.append({
                    "img": image_path[0],
                    "name": name,
                })
        elif 'val' in self.list_path:
            for item in self.img_list:
                item = item[0]
                image_path = os.path.join(item, item.split('/')[1][3:-5] + '.hsd')
                label_path = os.path.join(item, 'label_gray.png')
                name = os.path.splitext(os.path.basename(image_path))[0]
                files.append({
                    "img": image_path,
                    "label": label_path,
                    "name": name,
                    "weight": 1
                })

        else:
            for item in self.img_list:
                item = item[0]
                image_path = os.path.join(item, item.split('/')[1][3:-5] + '.hsd')
                label_path = os.path.join(item, 'label_gray.png')
                # label_path = os.path.join('finetune_label0.6151/', item.split('/')[1][:-4] + 'cropped.png')
                name = os.path.splitext(os.path.basename(image_path))[0]
                files.append({
                    "img": image_path,
                    "label": label_path,
                    "name": name,
                    "weight": 1
                })
        return files

    def convert_label(self, label, inverse=False):
        temp = label.copy()
        if inverse:
            for v, k in self.label_mapping.items():
                label[temp == k] = v
        else:
            for k, v in self.label_mapping.items():
                label[temp == k] = v
        return label

    def __getitem__(self, index):
        item = self.files[index]
        name = item["name"]
        image = self.read_HSD(os.path.join(self.root, item["img"]))
        image = image.transpose(1, 0, 2)[:, ::-1, :]
        size = image.shape

        if 'test' in self.list_path:
            image = self.input_transform(image)
            image = image.transpose((2, 0, 1))

            return image.copy(), np.array(size), name

        label = cv2.imread(os.path.join(self.root, item["label"]),
                           cv2.IMREAD_GRAYSCALE)
        if 'val' in self.list_path:
            label = label.transpose(1, 0)[:, ::-1]  # covert横放label

        image, label = self.gen_sample(image, label,
                                       self.multi_scale, self.flip,
                                       self.center_crop_test)

        return image.copy(), label.copy(), np.array(size), name

    def multi_scale_inference(self, model, image, scales=[1], flip=False):
        batch, _, ori_height, ori_width = image.size()
        assert batch == 1, "only supporting batchsize 1."
        image = image.numpy()[0].transpose((1, 2, 0)).copy()
        stride_h = np.int(self.crop_size[0] * 1.0)
        stride_w = np.int(self.crop_size[1] * 1.0)
        final_pred = torch.zeros([1, self.num_classes,
                                  ori_height, ori_width]).cuda()
        for scale in scales:
            new_img = self.multi_scale_aug(image=image,
                                           rand_scale=scale,
                                           rand_crop=False)
            height, width = new_img.shape[:-1]

            if scale <= 1.0:
                new_img = new_img.transpose((2, 0, 1))
                new_img = np.expand_dims(new_img, axis=0)
                new_img = torch.from_numpy(new_img)
                preds = self.inference(model, new_img, flip)
                preds = preds[:, :, 0:height, 0:width]
            else:
                new_h, new_w = new_img.shape[:-1]
                rows = np.int(np.ceil(1.0 * (new_h -
                                             self.crop_size[0]) / stride_h)) + 1
                cols = np.int(np.ceil(1.0 * (new_w -
                                             self.crop_size[1]) / stride_w)) + 1
                preds = torch.zeros([1, self.num_classes,
                                     new_h, new_w]).cuda()
                count = torch.zeros([1, 1, new_h, new_w]).cuda()

                for r in range(rows):
                    for c in range(cols):
                        h0 = r * stride_h
                        w0 = c * stride_w
                        h1 = min(h0 + self.crop_size[0], new_h)
                        w1 = min(w0 + self.crop_size[1], new_w)
                        h0 = max(int(h1 - self.crop_size[0]), 0)
                        w0 = max(int(w1 - self.crop_size[1]), 0)
                        crop_img = new_img[h0:h1, w0:w1, :]
                        crop_img = crop_img.transpose((2, 0, 1))
                        crop_img = np.expand_dims(crop_img, axis=0)
                        crop_img = torch.from_numpy(crop_img)
                        pred = self.inference(model, crop_img, flip)
                        preds[:, :, h0:h1, w0:w1] += pred[:, :, 0:h1 - h0, 0:w1 - w0]
                        count[:, :, h0:h1, w0:w1] += 1
                preds = preds / count
                preds = preds[:, :, :height, :width]
            preds = F.upsample(preds, (ori_height, ori_width),
                               mode='bilinear')
            final_pred += preds
        return final_pred

    def get_palette_hsicity(self, n):
        palette = [0] * (n * 3)
        for j in range(0, len(self.hsicity_label)):
            palette[j * 3] = self.hsicity_label[j][1][0]
            palette[j * 3 + 1] = self.hsicity_label[j][1][1]
            palette[j * 3 + 2] = self.hsicity_label[j][1][2]
        return palette

    def save_pred(self, preds, sv_path, name):
        palette = self.get_palette_hsicity(256)
        preds = preds.cpu().numpy().copy()
        preds = np.asarray(np.argmax(preds, axis=2), dtype=np.uint8) + 1  # 9 classes
        # preds = confidence_label_sort(preds)

        pred = self.convert_label(preds, inverse=True)
        save_img = Image.fromarray(pred)
        save_img.putpalette(palette)
        save_img.save(os.path.join(sv_path, name[0] + 'vis.png'))

        # preds = preds.squeeze()
        # save_img = Image.fromarray(preds)
        # save_img.save(os.path.join(sv_path, name[0] + '.png'))

    def save_pred_gray(self, preds, sv_path, name):
        preds = preds.cpu().numpy().copy()
        preds = np.asarray(np.argmax(preds, axis=2), dtype=np.uint8)

        preds = preds.squeeze()
        save_img = Image.fromarray(preds)
        save_img.save(os.path.join(sv_path, name + '.png'))

    def read_HSD(self, filename):
        # int32格式
        data = np.fromfile('%s' % filename, dtype=np.int32)
        height = data[0]
        width = data[1]
        SR = data[2]
        D = data[3]
        startw = data[4]  # 起始波段
        endw = data[6]  # 结束波段

        # float32格式
        data = np.fromfile('%s' % filename, dtype=np.float32)
        stepw = data[5]
        a = 7
        average = data[a:a + SR]
        a = a + SR
        coeff = data[a:a + D * SR].reshape((D, SR))
        a = a + D * SR
        scoredata = data[a:a + height * width * D].reshape((height * width, D))

        temp = np.dot(scoredata, coeff)

        data = (temp + average).reshape((height, width, SR))
        # data = (data - data.min()) / (data.max() - data.min())  # 全局归一化

        return data

