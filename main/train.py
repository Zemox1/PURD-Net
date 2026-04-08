import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import numpy as np

import time
import datetime
import os

from data.dataaug import depth_aug
from data.example_dataset.dataset import get_example_dataset
from depth_estimation.model.model import UDFNet
from depth_estimation.utils.loss import (
    LabColorLoss,
    SILogLoss,
    RMSELoss,
    ChamferDistanceLoss, ssim_loss, PerceptualLoss,
)
from depth_estimation.utils.visualization import get_tensorboard_grids

# training parameters
BATCH_SIZE = 4
LEARNING_RATE = 1e-4
LEARNING_RATE_DECAY = 0.90
EPOCHS = 100
DEVICE = "cuda:2"
WEIGHT_PATH = "saved_models/1"

LOSS_FUNCTIONS = {
    "SILog_Loss": SILogLoss(correction=0.85, scaling=10.0),
    "Chamfer_Loss": ChamferDistanceLoss(),
    "L2_Loss": RMSELoss(),
    "L1_Loss": torch.nn.L1Loss(),
}
LOSS_WEIGHTS = {"w_SILog_Loss": 0, "w_Chamfer_Loss": 0, "w_L2_Loss": 0.1}

TRAINING_LOSS_NAMES = [
    "training_loss",
    # "training_loss/SILog Loss",
    # "training_loss/Bins Chamfer Loss",
    "training_loss/L2 Loss (RMSE)",
    # "training_loss/L1 Loss (MAE)",
    # "training_loss/L2 Log Loss (RMSE log)",

]
VALIDATION_LOSS_NAMES = [
    "validation_loss",
    "validation_loss/SILog Loss",
    "validation_loss/Bins Chamfer Loss",
    "validation_loss/L2 Loss (RMSE)",
    "validation_loss/L1 Loss (MAE)",
    "validation_loss/L2 Log Loss (RMSE log)",

]


TRAIN_DATASET = get_example_dataset(train=True, shuffle=True, device=DEVICE)
VALIDATION_DATASET = get_example_dataset(train=False, shuffle=True, device=DEVICE)  # you should change this, this should not be the same as training

# tensorboard output frequencies
WRITE_TRAIN_IMG_EVERY_N_BATCHES = 500
WRITE_VALIDATION_IMG_EVERY_N_BATCHES = 300


def train_UDFNet():
    """Train loop to train a UDFNet model."""

    # print run infos
    run_name = f"lr{LEARNING_RATE}_bs{BATCH_SIZE}_lrd{LEARNING_RATE_DECAY}"
    print(
        f"Training run {run_name} with parameters:\n"
        + f"    learning rate: {LEARNING_RATE}\n"
        + f"    learning rate decay: {LEARNING_RATE_DECAY}\n"
        + f"    batch size: {BATCH_SIZE}\n"
        + f"    device: {DEVICE}"
    )
    # tensorboard summary writer
    global summary_writer

    summary_writer = SummaryWriter(run_name)
    # initialize model

    model = UDFNet(n_bins=100).to(DEVICE)
    # dataloaders

    train_dataloader = DataLoader(TRAIN_DATASET, batch_size=BATCH_SIZE, shuffle=True, num_workers=4,
                                  multiprocessing_context='spawn')

    # 记录训练开始时间
    training_start_time = time.time()
    # train epochs
    for epoch in range(EPOCHS):
        # decayed learning rate
        lr = LEARNING_RATE * (LEARNING_RATE_DECAY ** epoch)
        # epoch info
        print("------------------------")
        print(f"Epoch {epoch}/{EPOCHS} (lr: {lr}, batch_size: {BATCH_SIZE})")
        print("------------------------")
        # 记录epoch开始时间
        epoch_start_time = time.time()
        # train epoch
        training_losses = train_epoch(
            dataloader=train_dataloader,
            model=model,
            learning_rate=lr,
            epoch=epoch,
        )
        # 每10轮保存一次模型，包括最后一轮
        if epoch % 10 == 0 or epoch == EPOCHS - 1:
            save_model(model, epoch, run_name)
        # 计算时间统计
        epoch_elapsed_time = time.time() - epoch_start_time
        total_elapsed_time = time.time() - training_start_time
        overall_progress = (epoch + 1) / EPOCHS

        if epoch >= 0:  # 从第0个epoch就开始显示
            estimated_total_time = total_elapsed_time / overall_progress
            remaining_time = estimated_total_time - total_elapsed_time

            print(f"\n总体进度: {overall_progress:.1%}")
            print(f"本轮耗时: {str(datetime.timedelta(seconds=epoch_elapsed_time))}")
            print(f"已用时间: {str(datetime.timedelta(seconds=total_elapsed_time))}")
            print(f"预计剩余: {str(datetime.timedelta(seconds=remaining_time))}")
            print(
                f"预计完成: {(datetime.datetime.now() + datetime.timedelta(seconds=remaining_time)).strftime('%Y-%m-%d %H:%M:%S')}")

    # 训练结束后显示总时间
    total_training_time = time.time() - training_start_time
    print(f"\n🎉 训练完成! 总耗时: {str(datetime.timedelta(seconds=total_training_time))}")




def train_epoch(
    dataloader,
    model,
    learning_rate,
    epoch=0,
):
    """Train a model for one epoch.
    - dataloader: the dataloader to use
    - model: The model to train
    - learning_rate: the learning rate for the optimizer
    - epoch: epoch id"""

    # set training mode
    model.train()
    if  os.path.exists(WEIGHT_PATH):
        model.load_state_dict(torch.load(WEIGHT_PATH, map_location=DEVICE))
        print(f"已加载权重: {WEIGHT_PATH}")
    optimizer = AdamW(model.parameters(), lr=learning_rate)

    n_batches = len(dataloader)

    training_losses = np.zeros(len(TRAINING_LOSS_NAMES))
    for batch_id, data in enumerate(dataloader):

        I0 =data[0].to(DEVICE).float()
        I45 =data[1].to(DEVICE).float()
        I90 =data[2].to(DEVICE).float()
        I135 =data[3].to(DEVICE).float()
        y = data[4].to(DEVICE).float()  # depth image
        j = data[5].to(DEVICE).float()
        t = data[6].to(DEVICE).float()
        a = data[7].to(DEVICE).float()


        # prediction
        pred, bin_edges,recover,a_out,depth0= model(I0,I45,I90,I135)
        bin_centers = 0.5 * (bin_edges[:, :-1] + bin_edges[:, 1:])


        eps = 1e-8

        # perceptual_loss = PerceptualLoss(
        #     layers=[2, 7, 12],  # 选前3层，更关注细节
        #     weights=[1.0, 0.7, 0.4],
        #     device='cuda'
        # )
        def create_brightness_mask(image, threshold=0.9):
            """基于亮度阈值生成反光掩码"""
            # 转换为灰度计算亮度
            if image.shape[1] == 3:  # RGB图像
                gray = torch.mean(image, dim=1, keepdim=True)
            else:
                gray = image

            mask = (gray < threshold).float()
            return mask

        # mask=create_brightness_mask(j)
        # pred=pred*mask
        # y=y*mask


        # individual losses
        # batch_loss_silog =  LOSS_FUNCTIONS["SILog_Loss"](pred, y)
        # batch_loss_chamfer =  LOSS_FUNCTIONS["Chamfer_Loss"](y, bin_centers)
        batch_loss_l2 = LOSS_FUNCTIONS["L2_Loss"](pred, y)+LOSS_FUNCTIONS["L2_Loss"](recover, j)
        # batch_loss_l1 =  LOSS_FUNCTIONS["L1_Loss"](pred, y)+LOSS_FUNCTIONS["L1_Loss"](recover, j)+ LOSS_FUNCTIONS["L1_Loss"](t_out, t)+LOSS_FUNCTIONS["L1_Loss"](a_out, a)
        # batch_loss_precep=perceptual_loss(a_out, a)
        # batch_loss_l2_log =  LOSS_FUNCTIONS["L2_Loss"](
        #     torch.log(pred+eps), torch.log(y+eps)
        # )

        # learning objective loss
        batch_loss = (
            # batch_loss_silog * LOSS_WEIGHTS["w_SILog_Loss"]
            # + batch_loss_chamfer * LOSS_WEIGHTS["w_Chamfer_Loss"]
            + batch_loss_l2 * LOSS_WEIGHTS["w_L2_Loss"]
            # + batch_loss_l1 * 0
            

        )

        # backpropagation
        optimizer.zero_grad()
        batch_loss.backward()
        optimizer.step()

        # statistics for tensorboard visualization graphs
        batch_losses = np.array(
            [
                batch_loss.item(),
                # batch_loss_silog.item(),
                # batch_loss_chamfer.item(),
                batch_loss_l2.item(),
                # batch_loss_l1.item(),
                # batch_loss_l2_log.item(),

            ]
        )

        training_losses +=  batch_losses

        # tensorboard summary grids for visual inspection
        if (batch_id % WRITE_TRAIN_IMG_EVERY_N_BATCHES == 0) and (
            I0.size(0) == BATCH_SIZE
        ):

            with torch.no_grad():  # no gradients for visualization
                global_step = epoch * len(dataloader) + batch_id

                # 记录各种损失到TensorBoard
                summary_writer.add_scalar('Loss/batch_total', batch_loss.item(), global_step)
                summary_writer.add_scalar('Loss/batch_L2', batch_loss_l2.item(), global_step)
                # summary_writer.add_scalar('Loss/batch_silog', batch_loss_silog.item(), global_step)

        if batch_id % 50 == 0:

            print(f"\n=== Batch {batch_id} 模型输出范围 ===")
            print(f"pred:    {pred.min():.6f} ~ {pred.max():.6f}, mean: {pred.mean():.6f}")
            print(f"recover: {recover.min():.6f} ~ {recover.max():.6f}, mean: {recover.mean():.6f}")
            

           

    avg_batch_losses = training_losses / n_batches
    print(f"Average batch training loss: {avg_batch_losses}")
    return avg_batch_losses


def save_model(model, epoch, run_name):

    print(f"Saving model after epoch {epoch} ...")

    # check if folder exists
    folder_name = "saved_models"
    if not os.path.isdir(folder_name):
        os.mkdir(folder_name)

    # save model
    model_filename = f"{folder_name}/base_2_e{epoch+1}_{run_name}.pth"
    torch.save(model.state_dict(), model_filename)


if __name__ == "__main__":

    train_UDFNet()
