from torch.utils.data.dataset import Dataset
import json
from tqdm import tqdm
import cv2 as cv
import torch
import numpy as np

class MSCocoDataset(Dataset):
    def __init__(self, phase = 'train', size = None, device = 'mps', input_size = (640, 640), num_categories = None,
                 max_rot = 0, min_scale = 1., max_scale = 1.):
        self.phase = phase
        self.device = device
        self.size = size
        self.input_size = input_size
        self.num_categories = 90 if num_categories is None else num_categories
        self.max_rot = max_rot
        self.min_scale = min_scale
        self.max_scale = max_scale
        self.categories = {}
        self.__info = self.__load_data_info(f'data/MSCoco/annotations/instances_{phase}2017.json', size)
        self.__images = f'data/MSCoco/{phase}2017'
        #self.__train_info = self.__load_data_info('data/MSCoco/annotations/instances_train2017.json', train_size)
        #self.__images_train = 'data/MSCoco/train2017'
        pass

    def __load_data_info(self, annotation_file, size):
        print(annotation_file)
        with open(annotation_file, 'r') as f:
            content = json.load(f)

        for cat in tqdm(content['categories'], desc='Loading categories...'):
            self.categories[cat['id']] = cat['name']

        self.annotations = {}
        for annotation in tqdm(content['annotations'], desc=f'Loading annotations...'):
            image_id = annotation['image_id']
            annotation = self.__convert_annotation(annotation)
            if image_id not in self.annotations:
                self.annotations[image_id] = []
            self.annotations[image_id].append(annotation)

        data_info = []
        size = len(content['images']) if size is None or size > len(content['images']) else size

        for image in tqdm(content['images'][:size], desc='Loading images...'):
            if image['id'] not in self.annotations:
                self.annotations[image['id']] = []
            annotations = self.annotations[image['id']]#self.__get_annotations_by_image_id(content['annotations'], image['id'])

            scale = self.input_size[0] / image['width'] if image['width'] > image['height'] else self.input_size[1] / image['height']
            for a, annotation in enumerate(annotations):
                for s, seg in enumerate(annotation['segmentation']):
                    seg[:,0] *= scale#self.input_size[0] / image['width']
                    seg[:,1] *= scale#self.input_size[1] / image['height']
                    annotations[a]['segmentation'][s] = seg

            image_dict = {
                'file_name': image['file_name'],
                'annotations': annotations
            }
            data_info.append(image_dict)
            pass
        return data_info

    def __len__(self):
        return len(self.__info)
    
    def __get_annotations_by_image_id(self, annotations, image_id):
        img_annotations = []
        for annotation in annotations:
            if annotation['image_id'] == image_id:
                annotation = self.__convert_annotation(annotation)
                img_annotations.append(annotation)

        return img_annotations
    
    def __convert_annotation(self, annotation):
        converted = {}
        segs = []
        for seg in annotation['segmentation']:
            if 'counts' in annotation['segmentation']:
                continue
            seg = np.array(seg)
            seg = seg.reshape((int(seg.shape[0]/2), 2))
            #seg[:,0] *= self.size[0] / 640
            #seg[:,1] *= self.size[1] / 640
            segs.append(seg)
        converted['segmentation'] = segs
        converted['category_id'] = annotation['category_id']
        return converted

    def __getitem__(self, idx):
        #X = torch.zeros((3, self.input_size[0], self.input_size[1]), dtype=torch.float32)
        T = torch.zeros((self.num_categories, self.input_size[0], self.input_size[1]), dtype=torch.float32)

        record_info = self.__info[idx]
        fname = record_info['file_name']
        #x = np.zeros((self.input_size[0], self.input_size[1], 3))
        img = cv.imread(f'data/MSCoco/{self.phase}2017/{fname}')
        #img = cv.resize(img, self.input_size)
        #x[:img.shape[0], :img.shape[1], :] = img

        scale = self.input_size[0] / img.shape[0] if img.shape[0] > img.shape[1] else self.input_size[1] / img.shape[1]
        X = np.zeros((self.input_size[0], self.input_size[1], 3))
        w = int(img.shape[0]*scale)
        h = int(img.shape[1]*scale)
        X[:w, :h] = cv.resize(img, (h, w))
        #X = cv.resize(img, self.input_size)
        if scale != 1.:
            pass

        X = torch.tensor(X, dtype=torch.float32).permute(2, 0, 1) / 255
        #X[i, :, :img.shape[0], :img.shape[1]] = torch.tensor(img).permute(2, 0, 1) / 255

        for annotation in record_info['annotations']:
            cat_id = annotation['category_id']
            if cat_id == 0:
                pass
            if cat_id-1 >= self.num_categories:
                continue
            T[cat_id-1] = self.__draw_segmentation(T[cat_id-1], annotation['segmentation'])

        max_shift_x = self.input_size[0] - h
        max_shift_y = self.input_size[1] - w
        if self.max_rot != 0 or self.min_scale != 1. or self.max_scale != 1. or\
            max_shift_x != 0 or max_shift_y != 0:
            X, T = self.__augment(X, T, max_shift_x, max_shift_y)

        return X.to(self.device), T.to(self.device)
    
    def __augment(self, X, T, max_shift_x, max_shift_y):
        img_X = X.permute(1, 2, 0).numpy()
        channels_T = T.numpy()

        tX = int(np.random.random() * max_shift_x)
        tY = int(np.random.random() * max_shift_y)

        trans_mat = np.array([
            [1, 0, tX],
            [0, 1, tY]
        ], dtype=np.float32)
        img_X = cv.warpAffine(img_X, trans_mat, (img_X.shape[0], img_X.shape[1]))

        angle = -self.max_rot + 2 * np.random.random() * self.max_rot
        scale = self.min_scale + np.random.random() * (self.max_scale - self.min_scale)
        center = (img_X.shape[0] // 2, img_X.shape[1] // 2)
        rot_mat = cv.getRotationMatrix2D(center, angle, scale)

        img_X = cv.warpAffine(img_X, rot_mat, (X.shape[1], X.shape[2]))

        for c, channel_T in enumerate(channels_T):
            channels_T[c] = cv.warpAffine(channel_T, trans_mat, (channel_T.shape[0], channel_T.shape[1]))
            channels_T[c] = cv.warpAffine(channel_T, rot_mat, (channel_T.shape[0], channel_T.shape[1]))

        X_rot = torch.tensor(img_X).permute(2, 0, 1)
        T_rot = torch.tensor(channels_T)

        return X_rot, T_rot

    def __draw_segmentation(self, img, segmentation):
        for seg in segmentation:
            seg = np.array(seg)
            #seg = seg.reshape((int(len(seg) / 2), 2))
            #seg[:,0] *= self.size[0] / 640
            #seg[:,1] *= self.size[1] / 640
            seg = np.array(seg, dtype=np.int32)
            #img = cv.polylines(img, [seg], 1, (category_id, 0, 0))
            img = torch.tensor(cv.fillPoly(np.array(img), [seg], (1, 1, 1)), dtype=torch.float32)
        return img

    def __get_image_by_id(self, images, id):
        for img in images:
            if img['id'] == id:
                return img['file_name']
        return None
    