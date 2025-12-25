from copy import deepcopy
from typing import Optional, Callable

import torch
import torchvision.datasets

class COCOPersonList(torchvision.datasets.coco.CocoDetection):
    def __init__(self, root: str, annFile: str,
                 transform: Optional[Callable] = None,
                 target_transform: Optional[Callable] = None,
                 transforms: Optional[Callable] = None):
        super().__init__(root, annFile, transform, target_transform, transforms)
        persons = []
        for i in self.ids:
            tgt = self._load_target(i)
            flag = False
            for j in range(len(tgt)):
                if tgt[j]['category_id'] == 1:
                    flag = True
                    break
            if flag:
                persons.append(i)
        self.persons = persons
        print("Person Idx. Selected")

    def __getitem__(self, item):
        id = self.persons[item]
        image = self._load_image(id)
        target = self._load_target(id)

        if self.transforms is not None:
            image, target = self.transforms(image, target)

        return image, target

    def __len__(self):
        return len(self.persons)




class COCOPersonSelect(torchvision.datasets.coco.CocoDetection):
    """
    Output item: (img tensor, target bbox, all bbox)
    """
    def __init__(self, root: str, annFile: str, force_item_dev_test = None,
                 transform: Optional[Callable] = None,
                 target_transform: Optional[Callable] = None,
                 transforms: Optional[Callable] = None,
                 only_single = False,
                 return_ann = False):
        super().__init__(root, annFile, transform, target_transform, transforms)
        persons = []
        self.tgt_id = 1
        for i in self.ids:
            tgt = self._load_target(i)
            img_data = self.coco.loadImgs(i)[0]
            hh = img_data["height"]
            ww = img_data["width"]
            min_area = (max(hh, ww) / 6) **2 if root.__contains__("coco") else 0 # TODO: move to config
            # min_area = (max(hh, ww) / 4)**2
            cnt = 0
            for j in range(len(tgt)):
                ann: dict = tgt[j]
                if ann['category_id'] == self.tgt_id and ann['area'] >= min_area :# and not (ann.__contains__("iscrowd") and ann['iscrowd']):
                    cnt += 1
            if only_single and cnt > 1:
                continue
            for j in range(len(tgt)):
                ann: dict = tgt[j]
                if ann['category_id'] == self.tgt_id and ann['area'] >= min_area and not (ann.__contains__("iscrowd") and ann['iscrowd']):
                # if ann['area'] >= min_area:
                    persons.append((i, j))
        self.persons = persons
        print("Person Idx. Selected")
        self.return_ann = return_ann
        self.force_item_dev_test = force_item_dev_test

    def __getitem__(self, item):
        if self.force_item_dev_test is not None:
            item = self.force_item_dev_test
        # item = item % 16 * 10000009 % self.__len__()
        # item = 123345713453049587 % self.__len__()
        id, tgt = self.persons[item]
        # id = 1
        image = self._load_image(id)
        anns = self._load_target(id)
        target_ann = anns[tgt]
        target = target_ann["bbox"]
        target = [float(i) for i in target]
        if self.return_ann:
            tgt_list = [target] + [a['bbox'] for a in anns if a['category_id'] == self.tgt_id]
            if self.transforms is not None:
                image, tgt_list = self.transforms(image, tgt_list)
            # tgt0 = deepcopy(target_ann)
            # tgt0['bbox'] = target.tolist()
            # print(tgt_list)
            return image, tgt_list[0], tgt_list

        if self.transforms is not None:
            image, target = self.transforms(image, target)
        return image, target


    def __len__(self):
        # return 64
        return len(self.persons)

# class InriaDataset(torch.utils.data.dataset.Dataset):
#     """InriaDataset: representation of the INRIA person dataset.
#
#     Internal representation of the commonly used INRIA person dataset.
#     Available at: http://pascal.inrialpes.fr/data/human/
#     img_dir_train = './data/INRIAPerson/Train/pos'
#     lab_dir_train = './data/train_labels'
#     max_lab
#
#     Attributes:
#         len: An integer number of elements in the
#         img_dir: Directory containing the images of the INRIA dataset.
#         lab_dir: Directory containing the labels of the INRIA dataset.
#         img_names: List of all image file names in img_dir.
#         shuffle: Whether or not to shuffle the dataset.
#
#     """
#
#     def __init__(self, img_dir, lab_dir, imgsize, max_lab = 15, shuffle=True):
#         n_png_images = len(fnmatch.filter(os.listdir(img_dir), '*.png'))
#         n_jpg_images = len(fnmatch.filter(os.listdir(img_dir), '*.jpg'))
#         n_images = n_png_images + n_jpg_images
#         if lab_dir is not None:
#             if not isinstance(lab_dir, list):
#                 lab_dir = [lab_dir]
#             n_labels = len(fnmatch.filter(os.listdir(lab_dir[0]), '*.txt'))
#             assert n_images == n_labels, "Number of images and number of labels don't match"
#         self.len = n_images
#         self.img_dir = img_dir
#         self.lab_dir = lab_dir
#         self.imgsize = imgsize
#         self.img_names = fnmatch.filter(os.listdir(img_dir), '*.png') + fnmatch.filter(os.listdir(img_dir), '*.jpg')
#         self.shuffle = shuffle
#         self.img_paths = []
#         for img_name in self.img_names:
#             self.img_paths.append(os.path.join(self.img_dir, img_name))
#         self.lab_paths = []
#         if self.lab_dir is None:
#             lab_dir = ['']
#         for img_name in self.img_names:
#             lab_path = [os.path.join(ld, img_name).replace('.jpg', '.txt').replace('.png', '.txt') for ld in lab_dir]
#             self.lab_paths.append(lab_path)
#         self.max_n_labels = max_lab
#
#     def __len__(self):
#         return self.len
#
#     def __getitem__(self, idx):
#         assert idx <= len(self), 'index range error'
#         # img_path = os.path.join(self.img_dir, self.img_names[idx])
#         img_path = self.img_paths[idx]
#         # lab_path = os.path.join(self.lab_dir, self.img_names[idx]).replace('.jpg', '.txt').replace('.png', '.txt')
#         lab_path = self.lab_paths[idx]
#         image = Image.open(img_path).convert('RGB')
#
#         if self.lab_dir is not None:
#             label = []
#             for lp in lab_path:
#                 if os.path.getsize(lp):       #check to see if label file contains data.
#                     lb = torch.from_numpy(np.loadtxt(lp)).float()
#                     if lb.dim() == 1:
#                         lb = lb.unsqueeze(0)
#                     label.append(lb)
#                 else:
#                     label.append(torch.ones([1, 5]).float())
#         else:
#             label = None
#
#         image, label = self.pad_and_scale(image, label)
#         transform = transforms.ToTensor()
#         image = transform(image)
#         if self.lab_dir is not None:
#             label = self.pad_lab(label)
#             if len(label) == 1:
#                 label = label[0]
#             return image, label
#         else:
#             return image, lab_path[0]
#
#     def pad_and_scale(self, img, label):
#         """
#
#         Args:
#             img:
#
#         Returns:
#
#         """
#         w, h = img.size
#         if w == h:
#             padded_img = img
#         else:
#             dim_to_pad = 1 if w < h else 2
#             if dim_to_pad == 1:
#                 padding = (h - w) / 2
#                 padded_img = Image.new('RGB', (h,h), color=(127,127,127))
#                 padded_img.paste(img, (int(padding), 0))
#                 if label is not None:
#                     for i in range(len(label)):
#                         label[i][:, [1]] = (label[i][:, [1]] * w + padding) / h
#                         label[i][:, [3]] = (label[i][:, [3]] * w / h)
#             else:
#                 padding = (w - h) / 2
#                 padded_img = Image.new('RGB', (w, w), color=(127,127,127))
#                 padded_img.paste(img, (0, int(padding)))
#                 if label is not None:
#                     for i in range(len(label)):
#                         label[i][:, [2]] = (label[i][:, [2]] * h + padding) / w
#                         label[i][:, [4]] = (label[i][:, [4]] * h / w)
#         resize = transforms.Resize((self.imgsize, self.imgsize))
#         padded_img = resize(padded_img)     #choose here
#         return padded_img, label
#
#     def pad_lab(self, label):
#         padded_lab = []
#         for lab in label:
#             pad_size = self.max_n_labels - lab.shape[0]
#             padded_lab.append(F.pad(lab, (0, 0, 0, pad_size), value=-1) if pad_size > 0 else lab)
#         return padded_lab
#


