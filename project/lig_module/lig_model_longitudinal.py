import random
from argparse import ArgumentParser
from typing import Any
import pandas as pd

import numpy as np
import pytorch_lightning as pl
import torch
from pytorch_lightning.utilities.parsing import AttributeDict
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from torch.nn import L1Loss, MSELoss, Sigmoid, SmoothL1Loss
from torch.optim.lr_scheduler import CosineAnnealingLR

from model.unet.unet import UNet
from utils.visualize import log_all_info
from utils.const import TEST_ACCURACY, TEST_PREDICTION

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def scale_img_to_0_255(img: np.ndarray, imin: Any = None, imax: Any = None) -> np.ndarray:
    imin = img.min() if imin is None else imin
    imax = img.max() if imax is None else imax
    scaled = np.array(((img - imin) * (1 / (imax - imin))) * 255, dtype="uint8")
    scaled[scaled < 0] = 0
    scaled[scaled >= 255] = 255
    return scaled


class LitModelLongitudinal(pl.LightningModule):
    def __init__(self, hparams: AttributeDict):
        super(LitModelLongitudinal, self).__init__()
        # self.save_hyperparameters(hparams) # this V1.5.0
        self.hparams = hparams # this V1.0.0
        self.model = UNet(
            in_channels=hparams.in_channels,
            out_classes=1,
            dimensions=3,
            padding_mode="zeros",
            activation=hparams.activation,
            conv_num_in_layer=[1, 2, 3, 3, 3],
            residual=False,
            out_channels_first_layer=16,
            kernel_size=5,
            normalization=hparams.normalization,
            downsampling_type="max",
            use_sigmoid=False,
            use_bias=True,
        )
        self.sigmoid = Sigmoid()
        if self.hparams.loss == "l2":
            self.criterion = MSELoss()
        elif self.hparams.loss == "l1":
            self.criterion = L1Loss()
        elif self.hparams.loss == "smoothl1":
            self.criterion = SmoothL1Loss()
        self.train_log_step = random.randint(1, 500)
        self.val_log_step = random.randint(1, 100)
        self.clip_min = self.hparams.clip_min
        self.clip_max = self.hparams.clip_max

    def forward(self, x: Any) -> Any:
        return self.model(x)

    def training_step(self, batch, batch_idx: int):
        inputs, targets = batch
        logits = self(inputs)
        loss = self.criterion(logits.view(-1), targets.view(-1))
        # residuals = targets[0] - logits[0]
        residuals = logits[0] - targets[0]

        if self.current_epoch % 75 == 0 and batch_idx == 0:
            log_all_info(
                module=self,
                img=inputs[0],
                target=targets[0],
                pred=logits[0],
                residual=residuals,
                loss=loss,
                batch_idx=batch_idx,
                state="train",
                kfold_num=self.hparams.kfold_num,
                num_rep=self.hparams.num_rep,
            )
        # Need kfold_num and rep_num to save the model checkpoint
        kfold_num = torch.as_tensor(self.hparams.kfold_num, dtype=torch.float)
        num_rep = torch.as_tensor(self.hparams.num_rep, dtype=torch.float)
        self.log("train_loss", loss, sync_dist=True, on_step=True, on_epoch=True)
        self.log("k_fold", kfold_num.to(device), sync_dist=True, on_step=True, on_epoch=True)
        self.log("num_rep", num_rep.to(device), sync_dist=True, on_step=True, on_epoch=True)
        return {"loss": loss,"k_fold": kfold_num, "num_rep": num_rep}
    
    def training_epoch_end(self, training_step_outputs):
        average_train_mae = torch.mean(
            torch.as_tensor([training_step_output["loss"].to(device) for training_step_output in training_step_outputs])
        )
        self.log("train_loss", average_train_mae.to(device), sync_dist=True, on_step=False, on_epoch=True)
        print(f"Mean absolute error on whole train image: {average_train_mae}")
        average_k_fold = torch.mean(
            torch.as_tensor([training_step_output["k_fold"].to(device) for training_step_output in training_step_outputs])
        )
        self.log("k_fold", average_k_fold.to(device))
        print(f"K_Fold: {average_k_fold}")
        average_num_rep = torch.mean(
            torch.as_tensor([training_step_output["num_rep"].to(device) for training_step_output in training_step_outputs])
        )
        self.log("num_rep", average_num_rep.to(device))
        print(f"Num_Rep: {average_num_rep}")

    def validation_step(self, batch, batch_idx: int):
        inputs, targets = batch
        logits = self(inputs)
        loss = self.criterion(logits.view(-1), targets.view(-1))
        self.log("val_loss", loss, sync_dist=True, on_step=True, on_epoch=True, prog_bar=True)
        # residuals = targets[0] - logits[0]
        residuals = logits[0] - targets[0]

        if self.current_epoch % 75 == 0 and batch_idx == 0:
            log_all_info(
                module=self,
                img=inputs[0],
                target=targets[0],
                pred= logits[0],
                residual=residuals,
                loss=loss,
                batch_idx=batch_idx,
                state="val",
                kfold_num=self.hparams.kfold_num,
                num_rep=self.hparams.num_rep,
            )
        inputs = inputs.cpu().detach().numpy().squeeze()
        targets = targets.cpu().detach().numpy().squeeze() 
        predicts = logits.cpu().detach().numpy().squeeze() 

        mse = np.square(np.subtract(targets, predicts)).mean()
        ssim_ = ssim(targets, predicts, data_range=predicts.max() - predicts.min())
        psnr_ = psnr(targets, predicts, data_range=predicts.max() - predicts.min())

        if self.hparams.in_channels >= 2:
            brain_mask = inputs[0] == inputs[0][0][0][0]
        elif self.hparams.in_channels == 1:
            brain_mask = inputs == inputs[0][0][0]

        pred_clip = np.clip(predicts, -self.clip_min, self.clip_max) - min(
            -self.clip_min, np.min(predicts)
        )
        targ_clip = np.clip(targets, -self.clip_min, self.clip_max) - min(
            -self.clip_min, np.min(targets)
        )
        # What is pred_255
        pred_255 = np.floor(256 * (pred_clip / (self.clip_min + self.clip_max)))
        targ_255 = np.floor(256 * (targ_clip / (self.clip_min + self.clip_max)))
        pred_255[brain_mask] = 0
        targ_255[brain_mask] = 0

        diff_255 = np.absolute(pred_255.ravel() - targ_255.ravel())
        mae = np.mean(diff_255)
        return {"MAE": mae, "MSE": mse, "SSIM": ssim_, "PSNR": psnr_}

    def validation_epoch_end(self, validation_step_outputs):
        average_mae = torch.mean(
            torch.as_tensor([validation_step_output["MAE"] for validation_step_output in validation_step_outputs])
        )
        self.log("val_MAE", average_mae.to(device), sync_dist=True, on_step=False, on_epoch=True)
        print(f"Mean absolute error on whole val image: {average_mae}")
        average_mse = torch.mean(
            torch.as_tensor([validation_step_output["MSE"] for validation_step_output in validation_step_outputs])
        )
        self.log("val_MSE", average_mse.to(device), sync_dist=True, on_step=False, on_epoch=True)
        print(f"Mean square error on whole val image: {average_mse}")
        average_ssim = torch.mean(
            torch.as_tensor([validation_step_output["SSIM"] for validation_step_output in validation_step_outputs])
        )
        self.log("val_SSIM", average_ssim.to(device), sync_dist=True, on_step=False, on_epoch=True)
        print(f"SSIM on whole val image: {average_ssim}")
        average_psnr = torch.mean(
            torch.as_tensor([validation_step_output["PSNR"] for validation_step_output in validation_step_outputs])
        )
        self.log("val_PSNR", average_psnr.to(device), sync_dist=True, on_step=False, on_epoch=True)
        print(f"PSNR on whole val image: {average_psnr}")

    def test_step(self, batch, batch_idx: int):
        inputs, targets = batch
        logits = self(inputs)
        loss = self.criterion(logits.view(-1), targets.view(-1))
        # residuals = targets[0]- logits[0]
        residuals = logits[0] - targets[0]
        if batch_idx <= 8:
            log_all_info(
                module=self,
                img=inputs[0],
                target=targets[0],
                pred=logits[0],
                residual=residuals,
                loss=loss,
                batch_idx=batch_idx,
                state="test",
                kfold_num=self.hparams.kfold_num,
                num_rep=self.hparams.num_rep,
            )

        inputs = inputs.cpu().detach().numpy().squeeze()
        targets = targets.cpu().detach().numpy().squeeze()
        predicts = logits.cpu().detach().numpy().squeeze()
        residuals = targets - predicts

        mse = np.square(np.subtract(targets, predicts)).mean()
        ssim_ = ssim(targets, predicts, data_range=predicts.max() - predicts.min())
        psnr_ = psnr(targets, predicts, data_range=predicts.max() - predicts.min())

        if batch_idx <= 8:
            filename = f"k_{self.hparams.kfold_num}_rep_{self.hparams.num_rep}_bat_{batch_idx}_test.npz"
            outfile = TEST_PREDICTION / filename
            np.savez(outfile, target=targets, predict=predicts, residual=residuals)

        if self.hparams.in_channels >= 2:
            brain_mask = inputs[0] == inputs[0][0][0][0]
        elif self.hparams.in_channels == 1:
            brain_mask = inputs == inputs[0][0][0]

        pred_clip = np.clip(predicts, -self.clip_min, self.clip_max) - min(
            -self.clip_min, np.min(predicts)
        )
        targ_clip = np.clip(targets, -self.clip_min, self.clip_max) - min(
            -self.clip_min, np.min(targets)
        )
        pred_255 = np.floor(256 * (pred_clip / (self.clip_min + self.clip_max)))
        targ_255 = np.floor(256 * (targ_clip / (self.clip_min + self.clip_max)))
        pred_255[brain_mask] = 0
        targ_255[brain_mask] = 0

        diff_255 = np.absolute(pred_255.ravel() - targ_255.ravel())
        mae = np.mean(diff_255)

        diff_255_mask = np.absolute(pred_255[~brain_mask].ravel() - targ_255[~brain_mask].ravel())
        mae_mask = np.mean(diff_255_mask)

        return {"MAE": mae, "MAE_mask": mae_mask, "MSE": mse, "SSIM": ssim_, "PSNR": psnr_}

    def test_epoch_end(self, test_step_outputs):
        average_mae = torch.mean(
            torch.as_tensor([testing_step_output["MAE"] for testing_step_output in test_step_outputs])
        )
        self.log("test_MAE", average_mae.to(device), sync_dist=True, on_step=False, on_epoch=True)
        print(f"average absolute error on whole test image; MAE: {average_mae}")
        average_mae_mask = torch.mean(
            torch.as_tensor([testing_step_output["MAE_mask"] for testing_step_output in test_step_outputs])
        )
        self.log("test_MAE_mask", average_mae_mask.to(device), sync_dist=True, on_step=False, on_epoch=True)
        print(f"average absolute error on test mask; MAE_Mask: {average_mae_mask}")
        average_mse = torch.mean(
            torch.as_tensor([testing_step_output["MSE"] for testing_step_output in test_step_outputs])
        )
        self.log("test_MSE", average_mse.to(device), sync_dist=True, on_step=False, on_epoch=True)
        print(f"average absolute error on test image; MSE: {average_mse}")
        average_ssim = torch.mean(
            torch.as_tensor([testing_step_output["SSIM"] for testing_step_output in test_step_outputs])
        )
        self.log("test_SSIM", average_ssim.to(device), sync_dist=True, on_step=False, on_epoch=True)
        print(f"SSIM on whole test image: {average_ssim}")
        average_psnr = torch.mean(
            torch.as_tensor([testing_step_output["PSNR"] for testing_step_output in test_step_outputs])
        )
        self.log("test_PSNR", average_psnr.to(device), sync_dist=True, on_step=False, on_epoch=True)
        print(f"PSNR on whole test image: {average_psnr}")

        # Creating dataframe to save in the test_accuracy folder
        df = pd.DataFrame({'Avg_MAE': average_mae.cpu().detach().numpy(),
        "avg_MAE_Mask": average_mae_mask.cpu().detach().numpy(),
        "avg_MSE": average_mse.cpu().detach().numpy(),
        "avg_SSIM": average_ssim.cpu().detach().numpy(),
        "avg_PSNR": average_psnr.cpu().detach().numpy(),}, index=[0])
        filename = f"k_{self.hparams.kfold_num}_rep_{self.hparams.num_rep}_result.csv"
        outfile = TEST_ACCURACY / filename
        df.to_csv(outfile, float_format='%.4f')

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.hparams.learning_rate,
            weight_decay=self.hparams.weight_decay,
        )
        # scheduler = ReduceLROnPlateau(optimizer, threshold=1e-10)
        lr_dict = {
            "scheduler": CosineAnnealingLR(optimizer, T_max=300, eta_min=0.000001),
            "monitor": "val_checkpoint_on",  # Default: val_loss
            "reduce_on_plateau": True,
            "interval": "epoch",
            "frequency": 1,
        }
        return [optimizer], [lr_dict]

    @staticmethod
    def add_model_specific_args(parent_parser: ArgumentParser) -> ArgumentParser:
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument("--learning_rate", type=float, default=1e-15)
        parser.add_argument("--loss", type=str, choices=["l1", "l2", "smoothl1"], default="l2")
        parser.add_argument(
            "--activation", type=str, choices=["ReLU", "LeakyReLU"], default="LeakyReLU"
        )
        parser.add_argument(
            "--normalization",
            type=str,
            choices=["Batch", "Group", "InstanceNorm3d"],
            default="InstanceNorm3d",
        )
        parser.add_argument("--weight_decay", type=float, default=1e-6)
        parser.add_argument("--clip_min", type=int, default=2)
        parser.add_argument("--clip_max", type=int, default=5)
        return parser
