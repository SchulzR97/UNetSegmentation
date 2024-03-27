from data.dataset import MSCocoDataset
import numpy as np
import cv2 as cv
from model import UNet
import torch
import torch.functional as F

DEVICE = 'mps'
NUM_CATEGORIES = 20
BATCH_SIZE = 1

def draw_segmentation(img, segmentation):
    for seg in segmentation:
        seg = np.array(seg)
        seg = seg.reshape((int(len(seg) / 2), 2))

        seg = np.array(seg, dtype=np.int32)
        #img = cv.polylines(img, [seg], 1, (category_id, 0, 0))
        img = cv.fillPoly(np.array(img), [seg], (1, 1, 1))
    return img

if __name__ == '__main__':
    softmax = torch.nn.Softmax(dim=0)

    x = torch.full((1, 2, 4, 4), 1.)
    

    for channel in x[0]:
        print(softmax(channel.reshape(16)))

    size = (320, 320)
    data_set = MSCocoDataset(phase = 'val', size=None, device=DEVICE, input_size=(320, 320), num_categories=NUM_CATEGORIES,
                             max_rot=20, min_scale=1.1, max_scale=1.5)
    data_loader = torch.utils.data.DataLoader(
        data_set,
        batch_size=BATCH_SIZE,
        num_workers=0,
        pin_memory=True,
        shuffle=True,
    )
    model = UNet(device = DEVICE, num_categories=NUM_CATEGORIES, scale_size=4)
    model.load_state_dict(torch.load('model.sd'))

    #for i in range(100):
    #    print(i)
    #    X, T = data_set[i]

    #    img_X = X.permute(1, 2, 0).cpu().numpy()
    #    cv.imshow('X', img_X)

    #    for t in range(3):
    #        img_T = T[t].cpu().numpy()
    #        cv.imshow(f'T[{t}]', img_T)

    #    cv.waitKey()
    #cv.destroyAllWindows()

    iterator = iter(data_loader)
    model.eval()

    while True:
        X, T = next(iterator)
        with torch.no_grad():
            Y = model(X)

        X = X.cpu()
        Y = Y.cpu()
        T = T.cpu()

        for x, y, t in zip(X, Y, T):
            x = x.permute(1, 2, 0)
            x = x.numpy()

            cv.imshow('x', x)

            for i, (layer_y, layer_t) in enumerate(zip(y, t)):
                img_y = layer_y.numpy()
                img_t = layer_t.numpy()

                category = f'{i+1} {data_set.categories[i+1]}' if i+1 in data_set.categories else f'{i+1} unknown'
                #if img_y.max() > 0.0:
                    #img_y = cv.cvtColor(img_y, cv.COLOR_GRAY2BGR)
                img_y = cv.putText(img_y, f'{category} prop: {img_y.max():0.4f}', (10, size[1]-20), cv.FONT_HERSHEY_SIMPLEX, size[0] / 500, (1., 0., 0.), 1)
                cv.imshow('y', img_y)
                #if img_t.max() > 0:
                img_t = cv.cvtColor(img_t, cv.COLOR_GRAY2BGR)
                img_t = cv.putText(img_t, category, (10, size[1]-20), cv.FONT_HERSHEY_SIMPLEX, size[0] / 500, (1., 0., 0.), 1)
                cv.imshow('t', img_t)

                img_y_scaled = (img_y - img_y.min()) / (img_y.max() - img_y.min())
                cv.imshow('y scaled', img_y_scaled)

                if layer_t.max() > 0.:
                    cv.waitKey()
                else:
                    cv.waitKey(1)
                pass
            #cv.waitKey()
    pass