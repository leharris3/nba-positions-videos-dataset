# <3 
# stolen from: https://github.com/jfzhang95/pytorch-deeplab-xception/blob/master/dataloaders/datasets/coco.py

import numpy as np
import torch
import os
import ujson as json

from glob import glob
from typing import List, Dict, Optional
from torch.utils.data import Dataset
from mypath import Path
from tqdm import trange
from pycocotools.coco import COCO
from pycocotools import mask
from torchvision import transforms
from dataloaders import custom_transforms as tr
from PIL import Image, ImageFile
from torchvision.io import read_video

ImageFile.LOAD_TRUNCATED_IMAGES = True

class BoundingBox():
    """
    Simple obj. representing a bounding box and it's relative position in an annotation file.
    Note: the `annotations` param should only be a pointer to annotations dict we load and store once in shared mem.
    """
    
    def __init__(self, annotations:Dict, frame_idx:int, bbx_idx:int, bbx:Dict) -> None:
        self.annotations=annotations
        self.frame_idx=frame_idx
        self.bbx_idx=bbx_idx
        self.bbx=bbx


class NBAClips(Dataset):
    def __init__(self, annotations_dir:str, results_dir:str, device:str) -> None:
        super().__init__()
        self.annotations_dir = annotations_dir
        self.results_dir = results_dir
        self.annotation_file_paths = self.get_annotation_file_paths()
        self.device = device
        
        # map of {abs_bbx_idx: BoundingBox}
        # how big is the obj. going to be?
        self.bbx_map = {}
        # current video frame tensors
        self.current_vid_frames = None
        
        # current annotation obj and bbx
        self.current_annotations = None
        self.current_bbx = None
        
        # abs idx of the current annotation file loaded
        self.current_annotation_fp_idx = 0
        self.current_frame_idx = 0
        
        # within an annotation file
        self.current_rel_bbx_idx = 0
        # global pointer
        self.current_abs_bbx_idx = 0
        
    def __getitem__(self, index):
        
        # TODO: load video from mem + annotations obj from mem, del everything else
        if self.current_annotations is None:
            # congrats, we reached the end of the dataset
            if self.current_annotation_fp_idx >= len(self.annotation_file_paths):
                return None
            
            # get next annotations fp
            next_fp = self.annotation_file_paths[self.current_annotation_fp_idx]
            self.current_annotations = NBAClips.load_annotations(next_fp)
            
            # release memory
            del self.current_vid_frames
            # load video into tensors
            self.current_vid_frames, _, _ = read_video(next_fp)
            # copy to GPU
            self.current_vid_frames:torch.Tensor = self.current_vid_frames.to(self.device)

            # reset pointers
            self.current_frame_idx = 0
            self.current_rel_bbx_idx = 0
            
        # load the next bounding box
        self.current_bbx = self.current_annotations["frames"][self.current_frame_idx]["bbox"][self.current_rel_bbx_idx]
        x, y, w, h = self.current_bbx['x'], self.current_bbx['y'], self.current_bbx['width'], self.current_bbx['height']
        
        # load the current cropped frame
        curr_frame = self.current_vid_frames[self.current_frame_idx][y:y+h, x:x+w]
        # next obj to be returned
        # TODO: this needs to be a tensor copied to the same GPU device
        curr_data_item = (self.current_annotation_fp_idx, self.current_frame_idx, self.current_rel_bbx_idx, curr_frame)
        
        # update bbx pointers
        # always need to do this before checking for EOF
        self.current_abs_bbx_idx += 1
        self.current_rel_bbx_idx += 1
        
        # check for EOF
        total_frames = len(self.current_annotations["frames"])
        num_bbxs_in_current_frames = len(self.current_annotations["frames"][self.current_frame_idx]["bbox"])
        if self.current_bbx_idx >= num_bbxs_in_current_frames:
            # we've reached the end of the current annotation file
            if self.current_frame_idx >= total_frames:
                self.current_annotation_fp_idx += 1
            else:
                # move to next frame
                self.current_rel_bbx_idx = 0
                self.current_frame_idx += 1
                
        return curr_data_item
                
    def __len__(self) -> int:
        return len(self.annotation_file_paths)
    
    @staticmethod
    def load_annotations(fp:str) -> Dict:
        # slightly faster way to read a json file
        with open(fp, 'rb') as f:
            return json.load(f)
        
    def get_video_fp(self, root_dir:str) -> Optional[str]:
        video_path = root_dir + self.current_annotations["video_path"]
        if not os.path.exists(video_path):
            return None
        return video_path
        
    def get_annotation_file_paths(self) -> List[str]:
        """
        Return a list of all remaining annotation file paths to be processed.
        """
        annotation_file_paths = glob(self.annotations_dir + "/*/*/*.json")
        to_process_names = {'_'.join(os.path.basename(x).split('_')[0: -2]): x for x in annotation_file_paths}
        already_processed_names = ['_'.join(os.path.basename(x).split('_')[0: -2]) for x in glob(self.results_dir + "/*/*/*.json")]
        for name in already_processed_names: 
            if name in to_process_names:
                del to_process_names[name]
        annotation_file_paths = list(to_process_names.values())
        

class COCOSegmentation(Dataset):
    NUM_CLASSES = 21
    CAT_LIST = [0, 5, 2, 16, 9, 44, 6, 3, 17, 62, 21, 67, 18, 19, 4,
        1, 64, 20, 63, 7, 72]

    def __init__(self,
                 args,
                 base_dir=Path.db_root_dir('coco'),
                 split='train',
                 year='2017'):
        super().__init__()
        ann_file = os.path.join(base_dir, 'annotations/instances_{}{}.json'.format(split, year))
        ids_file = os.path.join(base_dir, 'annotations/{}_ids_{}.pth'.format(split, year))
        self.img_dir = os.path.join(base_dir, 'images/{}{}'.format(split, year))
        self.split = split
        self.coco = COCO(ann_file)
        self.coco_mask = mask
        if os.path.exists(ids_file):
            self.ids = torch.load(ids_file)
        else:
            ids = list(self.coco.imgs.keys())
            self.ids = self._preprocess(ids, ids_file)
        self.args = args

    def __getitem__(self, index):
        _img, _target = self._make_img_gt_point_pair(index)
        sample = {'image': _img, 'label': _target}

        if self.split == "train":
            return self.transform_tr(sample)
        elif self.split == 'val':
            return self.transform_val(sample)

    def _make_img_gt_point_pair(self, index):
        coco = self.coco
        img_id = self.ids[index]
        img_metadata = coco.loadImgs(img_id)[0]
        path = img_metadata['file_name']
        _img = Image.open(os.path.join(self.img_dir, path)).convert('RGB')
        cocotarget = coco.loadAnns(coco.getAnnIds(imgIds=img_id))
        _target = Image.fromarray(self._gen_seg_mask(
            cocotarget, img_metadata['height'], img_metadata['width']))

        return _img, _target

    def _preprocess(self, ids, ids_file):
        print("Preprocessing mask, this will take a while. " + \
              "But don't worry, it only run once for each split.")
        tbar = trange(len(ids))
        new_ids = []
        for i in tbar:
            img_id = ids[i]
            cocotarget = self.coco.loadAnns(self.coco.getAnnIds(imgIds=img_id))
            img_metadata = self.coco.loadImgs(img_id)[0]
            mask = self._gen_seg_mask(cocotarget, img_metadata['height'],
                                      img_metadata['width'])
            # more than 1k pixels
            if (mask > 0).sum() > 1000:
                new_ids.append(img_id)
            tbar.set_description('Doing: {}/{}, got {} qualified images'. \
                                 format(i, len(ids), len(new_ids)))
        print('Found number of qualified images: ', len(new_ids))
        torch.save(new_ids, ids_file)
        return new_ids

    def _gen_seg_mask(self, target, h, w):
        mask = np.zeros((h, w), dtype=np.uint8)
        coco_mask = self.coco_mask
        for instance in target:
            rle = coco_mask.frPyObjects(instance['segmentation'], h, w)
            m = coco_mask.decode(rle)
            cat = instance['category_id']
            if cat in self.CAT_LIST:
                c = self.CAT_LIST.index(cat)
            else:
                continue
            if len(m.shape) < 3:
                mask[:, :] += (mask == 0) * (m * c)
            else:
                mask[:, :] += (mask == 0) * (((np.sum(m, axis=2)) > 0) * c).astype(np.uint8)
        return mask

    def transform_tr(self, sample):
        composed_transforms = transforms.Compose([
            tr.RandomHorizontalFlip(),
            tr.RandomScaleCrop(base_size=self.args.base_size, crop_size=self.args.crop_size),
            tr.RandomGaussianBlur(),
            tr.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            tr.ToTensor()])

        return composed_transforms(sample)

    def transform_val(self, sample):

        composed_transforms = transforms.Compose([
            tr.FixScaleCrop(crop_size=self.args.crop_size),
            tr.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            tr.ToTensor()])

        return composed_transforms(sample)


    def __len__(self):
        return len(self.ids)



if __name__ == "__main__":
    from dataloaders import custom_transforms as tr
    from dataloaders.utils import decode_segmap
    from torch.utils.data import DataLoader
    from torchvision import transforms
    import matplotlib.pyplot as plt
    import argparse

    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    args.base_size = 513
    args.crop_size = 513

    coco_val = COCOSegmentation(args, split='val', year='2017')

    dataloader = DataLoader(coco_val, batch_size=4, shuffle=True, num_workers=0)

    for ii, sample in enumerate(dataloader):
        for jj in range(sample["image"].size()[0]):
            img = sample['image'].numpy()
            gt = sample['label'].numpy()
            tmp = np.array(gt[jj]).astype(np.uint8)
            segmap = decode_segmap(tmp, dataset='coco')
            img_tmp = np.transpose(img[jj], axes=[1, 2, 0])
            img_tmp *= (0.229, 0.224, 0.225)
            img_tmp += (0.485, 0.456, 0.406)
            img_tmp *= 255.0
            img_tmp = img_tmp.astype(np.uint8)
            plt.figure()
            plt.title('display')
            plt.subplot(211)
            plt.imshow(img_tmp)
            plt.subplot(212)
            plt.imshow(segmap)

        if ii == 1:
            break

    plt.show(block=True)