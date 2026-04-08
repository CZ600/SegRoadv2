import os
import shutil

import cv2
import matplotlib
import numpy as np
import scipy.signal
import torch
import torch.nn.functional as F
from matplotlib import pyplot as plt
from PIL import Image
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from .utils import cvtColor, preprocess_input, resize_image
from .utils_metrics import compute_mIoU, summarize_segmentation_metrics

matplotlib.use('Agg')


class LossHistory():
    def __init__(self, log_dir, model, input_shape):
        self.log_dir = log_dir
        self.losses = []
        self.val_loss = []
        self.best_road_iou = -1.0

        os.makedirs(self.log_dir, exist_ok=True)
        self.writer = SummaryWriter(self.log_dir)
        try:
            dummy_input = torch.randn(2, 3, input_shape[0], input_shape[1])
            self.writer.add_graph(model, dummy_input)
        except Exception:
            pass

    def append_loss(self, epoch, loss, val_loss):
        os.makedirs(self.log_dir, exist_ok=True)

        self.losses.append(loss)
        self.val_loss.append(val_loss)

        with open(os.path.join(self.log_dir, "epoch_loss.txt"), 'a') as f:
            f.write(str(loss))
            f.write("\n")
        with open(os.path.join(self.log_dir, "epoch_val_loss.txt"), 'a') as f:
            f.write(str(val_loss))
            f.write("\n")

        self.writer.add_scalar('loss', loss, epoch)
        self.writer.add_scalar('val_loss', val_loss, epoch)
        self.loss_plot()

    def loss_plot(self):
        iters = range(len(self.losses))

        plt.figure()
        plt.plot(iters, self.losses, 'red', linewidth=2, label='train loss')
        plt.plot(iters, self.val_loss, 'coral', linewidth=2, label='val loss')
        try:
            num = 5 if len(self.losses) < 25 else 15
            plt.plot(iters, scipy.signal.savgol_filter(self.losses, num, 3), 'green', linestyle='--', linewidth=2, label='smooth train loss')
            plt.plot(iters, scipy.signal.savgol_filter(self.val_loss, num, 3), '#8B4513', linestyle='--', linewidth=2, label='smooth val loss')
        except Exception:
            pass

        plt.grid(True)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend(loc="upper right")
        plt.savefig(os.path.join(self.log_dir, "epoch_loss.png"))
        plt.cla()
        plt.close("all")


class EvalCallback():
    def __init__(self, net, input_shape, num_classes, image_ids, dataset_path, log_dir, cuda,
                 miou_out_path=".temp_miou_out", eval_flag=True, period=1, positive_class=1):
        super(EvalCallback, self).__init__()

        self.net = net
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.image_ids = image_ids
        self.dataset_path = dataset_path
        self.log_dir = log_dir
        self.cuda = cuda
        self.miou_out_path = miou_out_path
        self.eval_flag = eval_flag
        self.period = period
        self.positive_class = positive_class
        self.mious = [0.0]
        self.epoches = [0]
        self.writer = SummaryWriter(self.log_dir)

        if self.eval_flag:
            with open(os.path.join(self.log_dir, "epoch_miou.txt"), 'a') as f:
                f.write("0\n")
            with open(os.path.join(self.log_dir, "epoch_metrics.txt"), 'a', encoding='utf-8') as f:
                f.write("epoch,miou,accuracy,road_iou,precision,recall,f1\n")

    def get_miou_png(self, image):
        image = cvtColor(image)
        original_h = np.array(image).shape[0]
        original_w = np.array(image).shape[1]

        image_data, nw, nh = resize_image(image, (self.input_shape[1], self.input_shape[0]))
        image_data = np.expand_dims(
            np.transpose(preprocess_input(np.array(image_data, np.float32)), (2, 0, 1)),
            0
        )

        with torch.no_grad():
            images = torch.from_numpy(image_data)
            if self.cuda:
                images = images.cuda()

            pr = self.net(images)[0][0]
            pr = F.softmax(pr.permute(1, 2, 0), dim=-1).cpu().numpy()
            pr = pr[
                int((self.input_shape[0] - nh) // 2): int((self.input_shape[0] - nh) // 2 + nh),
                int((self.input_shape[1] - nw) // 2): int((self.input_shape[1] - nw) // 2 + nw)
            ]
            pr = cv2.resize(pr, (original_w, original_h), interpolation=cv2.INTER_LINEAR)
            pr = pr.argmax(axis=-1)

        return Image.fromarray(np.uint8(pr))

    def on_epoch_end(self, epoch, model_eval):
        metrics = None
        if epoch % self.period != 0 or not self.eval_flag:
            return metrics

        self.net = model_eval
        pred_dir = os.path.join(self.miou_out_path, 'detection-results')
        os.makedirs(pred_dir, exist_ok=True)

        print("Get miou.")
        for image_id in tqdm(self.image_ids):
            image_path = image_id.split(' ')[0]
            image_name = os.path.splitext(os.path.basename(image_path))[0]
            image = Image.open(image_path)
            image = self.get_miou_png(image)
            image.save(os.path.join(pred_dir, image_name + ".png"))

        print("Calculate miou.")
        hist, ious, recalls, precisions = compute_mIoU(
            self.dataset_path, pred_dir, self.image_ids, self.num_classes, None
        )
        temp_miou = np.nanmean(ious) * 100
        metrics = summarize_segmentation_metrics(
            hist, ious, recalls, precisions, positive_class=self.positive_class
        )

        self.mious.append(temp_miou)
        self.epoches.append(epoch)

        with open(os.path.join(self.log_dir, "epoch_miou.txt"), 'a') as f:
            f.write(str(temp_miou))
            f.write("\n")
        with open(os.path.join(self.log_dir, "epoch_metrics.txt"), 'a', encoding='utf-8') as f:
            f.write(
                f"{epoch},{metrics['miou'] * 100:.4f},{metrics['accuracy'] * 100:.4f},"
                f"{metrics['road_iou'] * 100:.4f},{metrics['road_precision'] * 100:.4f},"
                f"{metrics['road_recall'] * 100:.4f},{metrics['road_f1'] * 100:.4f}\n"
            )

        self.writer.add_scalar('metrics/miou', metrics['miou'], epoch)
        self.writer.add_scalar('metrics/accuracy', metrics['accuracy'], epoch)
        self.writer.add_scalar('metrics/precision', metrics['road_precision'], epoch)
        self.writer.add_scalar('metrics/recall', metrics['road_recall'], epoch)
        self.writer.add_scalar('metrics/f1', metrics['road_f1'], epoch)
        self.writer.add_scalar('metrics/road_iou', metrics['road_iou'], epoch)

        plt.figure()
        plt.plot(self.epoches, self.mious, 'red', linewidth=2, label='val miou')
        plt.grid(True)
        plt.xlabel('Epoch')
        plt.ylabel('Miou')
        plt.title('A Miou Curve')
        plt.legend(loc="upper right")
        plt.savefig(os.path.join(self.log_dir, "epoch_miou.png"))
        plt.cla()
        plt.close("all")

        print(
            "Eval metrics - "
            f"mIoU: {metrics['miou'] * 100:.2f}%, "
            f"Road IoU: {metrics['road_iou'] * 100:.2f}%, "
            f"Precision: {metrics['road_precision'] * 100:.2f}%, "
            f"Recall: {metrics['road_recall'] * 100:.2f}%, "
            f"F1: {metrics['road_f1'] * 100:.2f}%"
        )
        print("Get miou done.")
        shutil.rmtree(self.miou_out_path, ignore_errors=True)
        return metrics
