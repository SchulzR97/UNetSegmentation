import os
import torch
import cv2 as cv
import json
import numpy as np
from tqdm import tqdm
import gc

class MSCocoLoader:
    def __init__(self, train_size = None, val_size = None, device = None):
        if device == "cuda" and torch.cuda.is_available():
            self.device = "cuda"
        elif device == "mps" and torch.backends.mps.is_available():
            self.device = "mps"
        else:
            self.device = "cpu"

        X_val, T_val = self.__load_data('data/MSCoco/annotations/instances_val2017.json', 'data/MSCoco/val2017', val_size)
        X_train, T_train = self.__load_data('data/MSCoco/annotations/instances_train2017.json', 'data/MSCoco/train2017', train_size)
        
        self.X_val = X_val
        self.T_val = T_val
        self.X_train = X_train
        self.T_train = T_train

        pass

    def sample(self, batch_size:int = 32, phase = 'train'):
        if phase == 'train':
            indices = torch.randint(0, self.X_train.shape[0], (batch_size,))
            return self.X_train[indices], self.T_train[indices]
        elif phase == 'val':
            indices = torch.randint(0, self.X_val.shape[0], (batch_size,))
            return self.X_val[indices], self.T_val[indices]
        raise Exception(f'Phase {phase} is not supported. Please use train or val.')

    def __get_image_by_id(self, images, id):
        for img in images:
            if img['id'] == id:
                return img
        return None
    
    def __draw_segmentation(self, img, segmentation, category_id):
        for seg in segmentation:
            # 239, 260
            seg = np.array(seg, dtype=np.int32)
            seg = seg.reshape((int(len(seg) / 2), 2))
            #img = cv.polylines(img, [seg], 1, (category_id, 0, 0))
            img = cv.fillPoly(img, [seg], (category_id, 0, 0))
        return img

    def __load_data(self, annotation_file, image_directory, size:int = None):
        with open(annotation_file, 'r') as f:
            content = json.load(f)

        image_id = -1
        images = content['images']

        if size is None or len(images) < size:
            size = len(images)

        w = 640
        h = 640
        X = torch.zeros((size, w, h, 3))
        T = torch.zeros((size, w, h, 1))

        i = 0

        for annotation in tqdm(content['annotations'][:size], desc=annotation_file):
            if image_id != annotation['image_id']:
                if image_id != -1:
                    X[i, :img.shape[0], :img.shape[1], :] = torch.tensor(img / 255)
                    T[i, :img.shape[0], :img.shape[1], :] = torch.tensor(seg_img / 255)
                    i += 1

                    if i%100 == 0:
                        garbage_collected = gc.collect()
                    pass

                image_id = annotation['image_id']
                img_info = self.__get_image_by_id(images, annotation['image_id'])
                fname = img_info['file_name']
                img = cv.imread(f'{image_directory}/{fname}')
                seg_img = np.zeros((img.shape[0], img.shape[1], 1), dtype=np.uint8)

            seg = annotation['segmentation']
            seg_img = self.__draw_segmentation(seg_img, seg, annotation['category_id'])
            pass
        
        pass
        
        gc.collect()
        return X.to(self.device), T.to(self.device)