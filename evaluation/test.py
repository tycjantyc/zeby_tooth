import numpy as np
import sys
import os
import nibabel as nib
sys.path.append(os.getcwd().removesuffix('\\evaluation'))
print(sys.path)

from models.class_segmentor_model import build_class_segmentor_model
import torch as tc
import torch.nn as nn
from evaluation.evaluate_agh_segmentor import evaluate_agh

model = build_class_segmentor_model(inchannels=1, outchannels=34, depths=[16, 32, 64, 128, 192])
model.load_state_dict(tc.load('../models/weights/agh_segmentor_best.pt', weights_only=True))
file = '../data/snippets_img/1.npy'

image = np.load(file).astype(np.float32)

print(image.shape)

model = model.cuda()

image = tc.tensor(image[None, ...], dtype=tc.float32).cuda()

output = nn.Softmax(dim=1)(model(image))

output = tc.argmax(output, dim=1)

# output[output > 0.5] = 1
# output[output < 0.5] = 0

output = output[0].detach().cpu().numpy().astype(np.uint8)

print(output.shape)

affine = np.eye(4)
nifti_img = nib.Nifti1Image(output, affine)
nib.save(nifti_img, '../evaluation/evaluation_results/output_image.nii.gz')


