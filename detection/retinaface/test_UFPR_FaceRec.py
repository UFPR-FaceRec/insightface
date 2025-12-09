import os
import sys
import numpy as np
import cv2
import argparse
from retinaface import RetinaFace
from insightface.utils import face_align


parser = argparse.ArgumentParser()
parser.add_argument('--img', type=str, default='examples/Aaron_Eckhart/aligned_detect_1.183.jpg')
args = parser.parse_args()

assert os.path.isfile(args.img), f"Error, image file not found: \'{args.img}\'"

thresh = 0.8
scales = [1.0]    # 1.0 == original size
flip = False
count = 1

# gpuid = 0
gpuid = -1    # -1 == CPU
detector = RetinaFace('./model/retinaface-R50/R50', 0, gpuid, 'net3')

img = cv2.imread(args.img)
print(img.shape)
im_shape = img.shape

for c in range(count):
    faces, landmarks = detector.detect(img, thresh, scales=scales, do_flip=flip)
    print(c, faces.shape, landmarks.shape)

if faces is not None:
    print('found', faces.shape[0], 'faces')
    for i in range(faces.shape[0]):
        box = faces[i].astype(int)
        color = (0, 0, 255)
        img_copy = img.copy()
        cv2.rectangle(img_copy, (box[0], box[1]), (box[2], box[3]), color, 2)
        if landmarks is not None:
            landmark5 = landmarks[i].astype(int)
            #print(landmark.shape)
            for l in range(landmark5.shape[0]):
                color = (0, 0, 255)
                if l == 0 or l == 3:
                    color = (0, 255, 0)
                cv2.circle(img_copy, (landmark5[l][0], landmark5[l][1]), 1, color, 2)

            aligned_img = face_align.norm_crop(img, landmark=landmark5, image_size=112)
            aligned_filename = f"{os.path.splitext(args.img)[0]}_aligned_crop_112x112.png"
            print('Saving', aligned_filename)
            cv2.imwrite(aligned_filename, aligned_img)

    detect_filename = f"{os.path.splitext(args.img)[0]}_detected.png"
    print('Saving', detect_filename)
    cv2.imwrite(detect_filename, img_copy)

else:
    print('No face detected!')
