import cv2
import os
import sys
import glob
import numpy as np
from PIL import Image, ImageOps
import torchvision

input_dir = '/home/NewHDD/wc/GuanDao/result/hr/test'
output_dir = '/home/NewHDD/wc/GuanDao/result/lr/test'

input_list = sorted(glob.glob(os.path.join(input_dir, '*')))

for f in input_list:
    HR_name = os.path.basename(f)
    # print(HR_name)
    HR = cv2.imread(f)
    h, w, c = HR.shape
    LR = np.array(Image.fromarray(HR).resize((w//4, h//4), Image.BICUBIC))
    LR_HR = np.array(Image.fromarray(LR).resize((w, h), Image.BICUBIC))
    # LR_HR = np.array(Image.fromarray(HR).resize((448, 320), Image.BICUBIC))

    output = os.path.join(output_dir, HR_name)
    print(output)
    cv2.imwrite(output, LR_HR)