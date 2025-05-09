import random
from pathlib import Path
from typing import Optional

import numpy as np
import pytorch_lightning as pl
import torch
from monai.transforms import Compose, LoadNifti, Randomizable, apply_transform
from nibabel.freesurfer.mghformat import MGHImage
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader, Dataset

from utils.const import ADNI_LIST
from utils.transforms import (
    get_longitudinal_label_transforms,
    get_longitudinal_train_img_transforms,
    get_longitudinal_val_img_transforms,
)


class LongitudinalDataset(Dataset, Randomizable):
    def __init__(self, X_path: Path, y_path: Path, transform: Compose, num_scan_training: int):
        self.X_path = X_path
        self.y_path = y_path
        self.num_scan_training = num_scan_training
        self.X_transform = transform
        self.y_transform = get_longitudinal_label_transforms()

    def __len__(self):
        return int(len(self.X_path))

    # What is this used for?
    def randomize(self) -> None:
        MAX_SEED = np.iinfo(np.uint32).max + 1
        self._seed = self.R.randint(MAX_SEED, dtype="uint32")

    def __getitem__(self, i):
        self.randomize()
        loadnifti = LoadNifti()
        y_img, _ = loadnifti(self.y_path[i])
        y_img = apply_transform(self.y_transform, y_img)

        if isinstance(self.X_transform, Randomizable):
            self.X_transform.set_random_state(seed=self._seed)
            self.y_transform.set_random_state(seed=self._seed)

        X_img = []
        if self.num_scan_training > 1:
            for scan in self.X_path[i]:
                img = MGHImage.load(scan).get_fdata()
                img[img < 3] = 0.0
                img = apply_transform(self.X_transform, img)
                X_img.append(img)
            X_img = torch.cat(X_img, dim=0)
        else:
            img = MGHImage.load(self.X_path[i]).get_fdata()
            img[img < 3] = 0.0
            X_img = apply_transform(self.X_transform, img)

        return X_img, y_img


class DataModuleLongitudinal(pl.LightningDataModule):
    def __init__(self, batch_size: int, num_scan_training: int, kfold_num: int):
        super().__init__()
        self.batch_size = batch_size
        self.num_scan_training = num_scan_training
        self.kfold_num = kfold_num

    # perform on every GPU
    def setup(self, stage: Optional[str] = None) -> None:
        y_list_all = set(list(ADNI_LIST[0].glob("**/*.nii.gz")))
        y_list_mask = set(list(ADNI_LIST[0].glob("**/*_mask.nii.gz")))
        y = sorted(list(y_list_all - y_list_mask))

        if self.num_scan_training == 1:
            X_M12 = ADNI_LIST[1]
            X = sorted(list(X_M12.glob("**/*.nii.mgz")))
        elif self.num_scan_training == 2:
            X_M12, X_M06 = ADNI_LIST[1], ADNI_LIST[2]
            X_M12_files, X_M06_files = sorted(list(X_M12.glob("**/*.nii.mgz"))), sorted(
                list(X_M06.glob("**/*.nii.mgz"))
            )
            X = []
            for m12, m06 in zip(X_M12_files, X_M06_files):
                X.append([m12, m06])
        elif self.num_scan_training == 3:
            X_M12, X_M06, X_SC = ADNI_LIST[1], ADNI_LIST[2], ADNI_LIST[3]
            X_M12_files, X_M06_files, X_SC_files = (
                sorted(list(X_M12.glob("**/*.nii.mgz"))),
                sorted(list(X_M06.glob("**/*.nii.mgz"))),
                sorted(list(X_SC.glob("**/*.nii.mgz"))),
            )
            X = []
            for m12, m06, sc in zip(X_M12_files, X_M06_files, X_SC_files):
                X.append([m12, m06, sc])
        
        # Making holdout external test dataset by removing 7 images
        holdout_images_name = [
            "ADNI_005_S_1341_MR_MPR__GradWarp_Br",
            "ADNI_013_S_0996_MR_MPR__GradWarp_Br",
            "ADNI_023_S_0083_MR_MPR__GradWarp_Br",
            "ADNI_036_S_0577_MR_MPR__GradWarp_Br",
            "ADNI_062_S_0730_MR_MPR__GradWarp_Br",
            "ADNI_126_S_0606_MR_MPR__GradWarp_Br",
            "ADNI_137_S_0366_MR_MPR__GradWarp_Br",
        ]

        X_train_val = []
        X_test = []
        y_train_val = []
        y_test = []

        for sub_list in X:
            if any(all(remove in item.name for item in sub_list) for remove in holdout_images_name):
                X_test.append(sub_list)
            else:
                X_train_val.append(sub_list)
        for item in y:
            if any(remove in item.name for remove in holdout_images_name):
                y_test.append(item)
            else:
                y_train_val.append(item)

        random_state = random.randint(0, 100)
        kf = KFold(n_splits=5, shuffle=True, random_state=random_state)

        for idx, (train_idx, val_idx) in enumerate(kf.split(X_train_val)):
            if self.kfold_num != (idx + 1):
                continue

            X, y = np.array(X_train_val), np.array(y_train_val)
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]

            train_transforms = get_longitudinal_train_img_transforms()
            val_transforms = get_longitudinal_val_img_transforms()
            self.train_dataset = LongitudinalDataset(
                X_path=X_train,
                y_path=y_train,
                transform=train_transforms,
                num_scan_training=self.num_scan_training,
            )
            self.val_dataset = LongitudinalDataset(
                X_path=X_val,
                y_path=y_val,
                transform=val_transforms,
                num_scan_training=self.num_scan_training,
            )

            self.test_dataset = LongitudinalDataset(
                X_path=X_test,
                y_path=y_test,
                transform=val_transforms,
                num_scan_training=self.num_scan_training,
            )

    def train_dataloader(self):
        print(f"get {len(self.train_dataset)} training 3D image!")
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=8,
        )

    def val_dataloader(self):
        print(f"get {len(self.val_dataset)} validation 3D image!")
        return DataLoader(self.val_dataset, batch_size=1, num_workers=8)

    def test_dataloader(self):
        print(f"get {len(self.test_dataset)} testing 3D image!")
        return DataLoader(self.test_dataset, batch_size=1, num_workers=1)
