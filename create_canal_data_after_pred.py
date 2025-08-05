from models.tooth_fairy_1 import build_hyspark_tooth_fairy_1
import os
import numpy as np
from processing.utils import embed_in_center
import torch as tc
import torch.nn as nn 
from monai.losses import DiceLoss
from tqdm import tqdm

if __name__ == "__main__":

    loss = DiceLoss(sigmoid=False, include_background=True)
    input_size = (96, 96, 96)
    model = build_hyspark_tooth_fairy_1(in_channel=1, n_classes=1, img_size=96, freeze=False, weight_path='models/weights/tooth_fairy_3d_best.pt')
    model = model.cuda()
    model.eval()
    lista = []

    for file, label in tqdm(zip(sorted(os.listdir('data/snippets_img')), sorted(os.listdir('data/snippets_mask')))):

        file = os.path.join('data/snippets_img', file)
        label = os.path.join('data/snippets_mask', label)

        print(file, label)

        image = np.load(file)
        mask = np.load(label)

        masked = image[mask>0]
        image[mask>0] = (masked - masked.min())/(masked.max() - masked.min() + 1e-5) #normalizacja dla lepszego kontrastu
        image[mask == 0] = 0 #zalewanie wszystkiego co nie jest kanałem

        mask[mask<=1] = 0
        mask[mask>1] = 1

        if image.shape[0] < input_size[0] and image.shape[1] < input_size[1] and image.shape[2] < input_size[2]: 
            image = embed_in_center(image, input_size[0])
            mask = embed_in_center(mask, input_size[0])
        else:
            print('ok')
            z, y, x = image.shape
            temp_img = np.zeros((96, 96, 96))
            temp_mask = np.zeros((96, 96, 96))
            temp_img[0:min(96, z), 0:min(96, y):, 0:min(96, x)] = image[0:min(96, z), 0:min(96, y):, 0:min(96, x)]
            temp_mask[0:min(96, z), 0:min(96, y):, 0:min(96, x)] = mask[0:min(96, z), 0:min(96, y):, 0:min(96, x)]

            image = temp_img
            mask = temp_mask

        mask = tc.tensor(mask[None, None, ...], dtype = tc.float32).cuda()
        input = tc.tensor(image[None, None, ...], dtype = tc.float32).cuda()
        pred = nn.Sigmoid()(model(input))

        print(mask.shape)

        print(1 - loss(pred, mask).item())
        lista.append(1 - loss(pred, mask).item())

        pred = pred.detach().cpu().numpy()[0, 0]

        pred[pred>0.5] = 1
        pred[pred<0.5] = 0
    
        np.save(file, image)
        np.save(label, pred)

    print(sum(lista)/len(lista))







