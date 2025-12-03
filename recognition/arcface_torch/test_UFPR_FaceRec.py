import os
import sys
import argparse
import cv2
import numpy as np
import torch
from sklearn import preprocessing, metrics

sys.path.insert(0, "../")
from backbones import get_model


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--img1', default='examples/Aaron_Eckhart/0.png', help='')
    parser.add_argument('--img2', default='examples/Aaron_Eckhart/1.png', help='')
    parser.add_argument('--network', default='r100', type=str, help='')
    parser.add_argument('--model', default='trained_models/ms1mv3_arcface_r100_fp16/backbone.pth', help='path to load model.')
    parser.add_argument('--gpu', default=0, type=int, help='gpu id')
    parser.add_argument('--batch-size', default=2, type=int, help='')
    args = parser.parse_args()
    return args


def load_model(network='r100', model='/path/to/backbone.pth/or/model.pt'):
    weight = torch.load(model)
    resnet = get_model(network, dropout=0, fp16=False).cuda()    # move model to GPU
    resnet.load_state_dict(weight)
    model = torch.nn.DataParallel(resnet)
    model.eval()
    return model


def load_normalize_img(path_img='/path/to/img.png'):
    img = cv2.imread(path_img)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = np.transpose(img, axes=(2, 0, 1))
    img = ((img / 255) - 0.5) / 0.5
    img = torch.from_numpy(img).cuda()   # move image to GPU
    return img


def cosine_similarity(embedd1, embedd2):
    if len(embedd1.shape) == 1:   # if (512,) make it (1, 512)
        embedd1 = np.expand_dims(embedd1, axis=0)
    if len(embedd2.shape) == 1:   # if (512,) make it (1, 512)
        embedd2 = np.expand_dims(embedd2, axis=0)
    cosine_sim = metrics.pairwise.cosine_similarity(embedd1, embedd2)
    cosine_sim = np.squeeze(cosine_sim)
    return cosine_sim


if __name__ == '__main__':
    args = parse_arguments()
    assert os.path.isfile(args.model), f"Error, no such model file: \'{args.model}\'"
    assert os.path.isfile(args.img1),  f"Error, no such image: \'{args.img1}\'"
    assert os.path.isfile(args.img2),  f"Error, no such image: \'{args.img2}\'"
    
    print(f'Loading model {args.network}: \'{args.model}\'')
    model = load_model(args.network, args.model)

    print(f'Loading img1 \'{args.img1}\'')
    img1 = load_normalize_img(args.img1)
    print('    img1.shape:', img1.shape)
    
    print(f'Loading img2 \'{args.img2}\'')
    img2 = load_normalize_img(args.img2)
    print('    img2.shape:', img2.shape)

    print('Making batch')
    image_size = (112, 112)
    batch_img = torch.empty(args.batch_size, 3, image_size[0], image_size[1])
    batch_img[0] = img1
    batch_img[1] = img2
    print('    batch.shape:', batch_img.shape)
    
    print('Computing embeddings')
    embeddings = model(batch_img)
    embeddings = embeddings.detach().cpu().numpy()
    embeddings_norm = preprocessing.normalize(embeddings)
    print('    embeddings.shape:', embeddings.shape)

    cosine_sim = cosine_similarity(embeddings_norm[0], embeddings_norm[1])
    print('Similarity:', cosine_sim)
    
    thresh = 0.5
    if cosine_sim >= thresh:
        print('Same person!')
    else:
        print('Different person!')
