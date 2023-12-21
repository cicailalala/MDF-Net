import os
import cv2
import copy
import torch
import random
import numpy as np
import albumentations as A
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from albumentations.core.transforms_interface import BasicTransform



def load_imgs(data_path, file_list, num=-1):
    image_arr = []
    labels = []

    with open(file_list) as f:
        image_lst = f.read().splitlines()
    for info in image_lst:
        info = info.split(' ')
        img_path, label = info
        img_path = os.path.join(data_path, img_path)
        if label != '2':
            if img_path not in image_arr:
                image_arr.append(img_path)
                labels.append(int(label))

    return image_arr, labels


class BCDataset(Dataset):
    def __init__(self, data_path, file_list, classes=3, cls=False, seg=True, transform=None, transform2=None, inf=False):
    
        self.image_arr, self.labels = load_imgs(data_path, file_list)

        # convert str names to class values on masks
        self.classes = classes
        self.transform = transform
        self.transform2 = transform2
        self.cls = cls
        self.seg = seg
        self.inf = inf



    def __getitem__(self, idx):
        # read data
        img_path = self.image_arr[idx]
        mask_path = img_path.replace('.png', '_mask.png')
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(mask_path, 0)
        label = self.labels[idx]

        # benign = 0, malignant = 1, normal (background) = 2


        mask = mask / 255.
        mask = mask.astype(np.int64)

        # apply augmentations
        if self.transform:
            if self.transform2:
                if random.random() > 0.2:
                    sample = self.transform(image=image, mask=mask)
                else:
                    sample = self.transform2(image=image, mask=mask)
            else:
                sample = self.transform(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']

        if self.cls and self.seg:
            return image, mask, label
        elif self.cls and not self.seg:
            return image, label
        elif not self.cls and self.seg:
            if self.inf:
                return image, mask, label
            else:
                return image, mask

    def __len__(self):
        return len(self.image_arr)


class BCInferenceDataset(Dataset):
    def __init__(self, file_root, classes=2, cls=False, seg=True, transform=None):
        self.image_arr, self.labels = load_imgs(file_root)

        # convert str names to class values on masks
        self.classes = classes
        self.transform = transform
        self.cls = cls
        self.seg = seg

    def __len__(self):
        return len(self.image_arr)

    def __getitem__(self, idx):
        img_path = self.image_arr[idx]
        mask_path = img_path.replace('.png', '_mask.png')
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        original_size = tuple(image.shape[:2])
        if self.transform is not None:
            transformed = self.transform(image=image)
            image = transformed["image"]
        return image, original_size


class ToTensorV2(BasicTransform):
    """Convert image and mask to `torch.Tensor`.
    Args:
        transpose_mask (bool): if True and an input mask has three dimensions, this transform will transpose dimensions
        so the shape `[height, width, num_channels]` becomes `[num_channels, height, width]`. The latter format is a
        standard format for PyTorch Tensors. Default: False.
    """

    def __init__(self, transpose_mask=True, always_apply=True, p=1.0):
        super(ToTensorV2, self).__init__(always_apply=always_apply, p=p)
        self.transpose_mask = transpose_mask

    @property
    def targets(self):
        return {"image": self.apply, "mask": self.apply_to_mask}

    def apply(self, img, **params):  # skipcq: PYL-W0613
        if len(img.shape) not in [2, 3]:
            raise ValueError("Albumentations only supports images in HW or HWC format")

        if len(img.shape) == 2:
            img = np.expand_dims(img, 2)

        return torch.from_numpy(img.transpose(2, 0, 1))

    def apply_to_mask(self, mask, **params):  # skipcq: PYL-W0613
        if self.transpose_mask and mask.ndim == 3:
            mask = mask.transpose(2, 0, 1)
        return torch.from_numpy(mask)

    def get_transform_init_args_names(self):
        return ("transpose_mask",)

    def get_params_dependent_on_targets(self, params):
        return {}



def visualize_augmentations(dataset, idx=0, samples=5):
    dataset = copy.deepcopy(dataset)
    dataset.transform = A.Compose([t for t in dataset.transform if not isinstance(t, (A.Normalize, ToTensorV2))])
    figure, ax = plt.subplots(nrows=samples, ncols=2, figsize=(10, 24))
    print(dataset.image_arr[idx])
    for i in range(samples):
        image, mask = dataset[idx]
        ax[i, 0].imshow(image)
        ax[i, 1].imshow(mask, interpolation="nearest")
        ax[i, 0].set_title("Augmented image")
        ax[i, 1].set_title("Augmented mask")
        ax[i, 0].set_axis_off()
        ax[i, 1].set_axis_off()
    plt.tight_layout()
    plt.show()





def display_image_grid(image_arr, labels, predicted_masks=None, save=False, save_root=''):
    cols = 3 if predicted_masks else 2
    rows = len(image_arr)
    figure, ax = plt.subplots(nrows=rows, ncols=cols, figsize=(10, 24))
    for i, image_filename in enumerate(image_arr):
        image = cv2.imread(image_filename)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        mask_path = image_filename.replace('.png', '_mask.png')
        mask = cv2.imread(mask_path, 0)
        label = labels[i]
        # print('path: ', image_filename, 'shape: ', image.shape)
        # mask = preprocess_mask(mask)
        ax[i, 0].imshow(image)
        ax[i, 1].imshow(mask, interpolation="nearest")

        img_title = image_filename[14:] + '  ' + str(mask.shape)
        ax[i, 0].set_title(img_title)
        ax[i, 1].set_title("Ground truth mask")

        ax[i, 0].set_axis_off()
        ax[i, 1].set_axis_off()

        if predicted_masks:
            predicted_mask = predicted_masks[i]
            predicted_mask = predicted_mask * 255
            ax[i, 2].imshow(predicted_mask, interpolation="nearest")
            ax[i, 2].set_title("Predicted mask")
            ax[i, 2].set_axis_off()
            if save:
                cv2.imwrite(os.path.join(save_root, str(i)+'_img.jpg'), image)
                cv2.imwrite(os.path.join(save_root, str(i) + '_mask.jpg'), mask)
                cv2.imwrite(os.path.join(save_root, str(i) + '_pr.jpg'), predicted_mask)
                # print(image.shape, mask.shape, predicted_mask.shape)
    #plt.tight_layout()
    #plt.show()