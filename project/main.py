import os
from argparse import ArgumentParser, Namespace
from pathlib import Path

import pytorch_lightning as pl
from pytorch_lightning import Trainer, loggers
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from lig_module.data_model_longitudinal import DataModuleLongitudinal
from lig_module.lig_model_longitudinal import LitModelLongitudinal
from utils.const import COMPUTECANADA
from pytorch_lightning.callbacks.early_stopping import EarlyStopping


def main(hparams: Namespace) -> None:
    # Function that sets seed for pseudo-random number generators in: pytorch, numpy,
    # python.random and sets PYTHONHASHSEED environment variable.
    pl.seed_everything(hparams.seed)

    if COMPUTECANADA:
        cur_path = Path(__file__).resolve().parent
        default_root_dir = cur_path
        checkpoint_file = (
            Path(__file__).resolve().parent
            / "checkpoint/{epoch}-{train_loss:0.5f}-{k_fold}-{num_rep}"  # noqa
        )
        if not os.path.exists(Path(__file__).resolve().parent / "checkpoint"):
            os.mkdir(Path(__file__).resolve().parent / "checkpoint")
    else:
        default_root_dir = Path("./log")
        if not os.path.exists(default_root_dir):
            os.mkdir(default_root_dir)
        checkpoint_file = Path("./log/checkpoint")
        if not os.path.exists(checkpoint_file):
            os.mkdir(checkpoint_file)
        checkpoint_file = checkpoint_file / "{epoch}-{train_loss:0.5f}-{k_fold}-{num_rep}"

    ckpt_path = (
        str(Path(__file__).resolve().parent / "checkpoint" / hparams.checkpoint_file)
        if hparams.checkpoint_file
        else None
    )

    # After training finishes, use best_model_path to retrieve the path to the best
    # checkpoint file and best_model_score to retrieve its score.
    checkpoint_callback = ModelCheckpoint(  # type: ignore
        filepath=str(checkpoint_file),
        monitor="train_loss",
        save_top_k=1,
        verbose=True,
        mode="min",
        save_weights_only=False,
    )
    tb_logger = loggers.TensorBoardLogger(hparams.TensorBoardLogger)

    # training
    trainer = Trainer(
        gpus=hparams.gpus,
        distributed_backend="ddp",
        fast_dev_run=hparams.fast_dev_run,
        checkpoint_callback=checkpoint_callback,
        callbacks=[
            LearningRateMonitor(logging_interval="epoch"),
            EarlyStopping("train_loss", patience=40, mode="min"),
        ],
        resume_from_checkpoint=ckpt_path,
        default_root_dir=str(default_root_dir),
        logger=tb_logger,
        max_epochs=230,
        # auto_scale_batch_size="binsearch", # for auto scaling of batch size
        progress_bar_refresh_rate=0
    )

    if hparams.task == "longitudinal":
        model = LitModelLongitudinal(hparams)
        data_module = DataModuleLongitudinal(
            hparams.batch_size, num_scan_training=hparams.in_channels, kfold_num = hparams.kfold_num,
        )

    trainer.fit(model, data_module)

    trainer = Trainer(gpus=hparams.gpus, 
    distributed_backend="ddp"
    )
    trainer.test(
        model=model,
        ckpt_path=ckpt_path,
        datamodule=data_module,
    )


if __name__ == "__main__":  # pragma: no cover
    parser = ArgumentParser(description="Trainer args", add_help=False)
    parser.add_argument("--gpus", type=int, default=1, help="how many gpus")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size", dest="batch_size")
    parser.add_argument(
        "--tensor_board_logger",
        dest="TensorBoardLogger",
        default="/home/x2020fpt/projects/def-jlevman/x2020fpt/rUnet_CC",
        # default="/home/mostafiz/desktop/rUnet_CC",
        help="TensorBoardLogger dir",
    )
    parser.add_argument(
        "--fast_dev_run",
        action="store_true",
        help="whether to run 1 train, val, test batch and program ends",
    )
    parser.add_argument(
        "--fine_tune",
        action="store_true",
        help="using this mode to fine tune, only use 300 images here",
    )
    parser.add_argument("--use_data_augmentation", action="store_true")
    parser.add_argument("--in_channels", type=int, default=288)
    parser.add_argument("--kfold_num", type=int, choices=[1, 2, 3, 4, 5], default=1)
    parser.add_argument("--num_rep", type=int, choices=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10], default=1)
    parser.add_argument("--seed", type=int, choices=[41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57], default=42)
    parser.add_argument(
        "--task",
        type=str,
        choices=["longitudinal",],
        default="longitudinal",
    )
    parser.add_argument("--checkpoint_file", type=str, help="resume from checkpoint file")
    parser = LitModelLongitudinal.add_model_specific_args(parser)
    hparams = parser.parse_args()

    main(hparams)
