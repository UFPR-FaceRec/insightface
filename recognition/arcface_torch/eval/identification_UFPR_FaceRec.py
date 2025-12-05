import datetime
import os
import sys
import pickle
import argparse
from PIL import Image
import torch
import sklearn

import mxnet as mx
import numpy as np
import sklearn
import torch
from mxnet import ndarray as nd
from scipy import interpolate
from sklearn.decomposition import PCA
from sklearn.model_selection import KFold

sys.path.insert(0, "../")
from backbones import get_model

from eval.loader_YouTubeFacesTINY import Loader_YouTubeFacesTINY


def parse_arguments():
    parser = argparse.ArgumentParser(description='do verification')
    parser.add_argument('--data-dir', default='../examples/YouTubeFaces_TINY/aligned_images_DB_DETECTED_FACES_RETINAFACE_scales=[1.0]_nms=0.4/imgs', help='')
    parser.add_argument('--network', default='r100', type=str, help='')
    parser.add_argument('--model', default='../trained_models/ms1mv3_arcface_r100_fp16/backbone.pth', help='path to load model.')
    parser.add_argument('--gpu', default=0, type=int, help='gpu id')
    parser.add_argument('--batch-size', default=32, type=int, help='')
    parser.add_argument('--nfolds', default=10, type=int, help='')
    args = parser.parse_args()
    return args


class LFold:
    def __init__(self, n_splits=2, shuffle=False):
        self.n_splits = n_splits
        if self.n_splits > 1:
            self.k_fold = KFold(n_splits=n_splits, shuffle=shuffle)

    def split(self, indices):
        if self.n_splits > 1:
            return self.k_fold.split(indices)
        else:
            return [(indices, indices)]


def calculate_roc(thresholds,
                  embeddings1,
                  embeddings2,
                  actual_issame,
                  nrof_folds=10,
                  pca=0):
    assert (embeddings1.shape[0] == embeddings2.shape[0])
    assert (embeddings1.shape[1] == embeddings2.shape[1])
    nrof_pairs = min(len(actual_issame), embeddings1.shape[0])
    nrof_thresholds = len(thresholds)
    k_fold = LFold(n_splits=nrof_folds, shuffle=False)

    tprs = np.zeros((nrof_folds, nrof_thresholds))
    fprs = np.zeros((nrof_folds, nrof_thresholds))
    accuracy = np.zeros((nrof_folds))
    indices = np.arange(nrof_pairs)

    if pca == 0:
        diff = np.subtract(embeddings1, embeddings2)
        dist = np.sum(np.square(diff), 1)

    for fold_idx, (train_set, test_set) in enumerate(k_fold.split(indices)):
        if pca > 0:
            print('doing pca on', fold_idx)
            embed1_train = embeddings1[train_set]
            embed2_train = embeddings2[train_set]
            _embed_train = np.concatenate((embed1_train, embed2_train), axis=0)
            pca_model = PCA(n_components=pca)
            pca_model.fit(_embed_train)
            embed1 = pca_model.transform(embeddings1)
            embed2 = pca_model.transform(embeddings2)
            embed1 = sklearn.preprocessing.normalize(embed1)
            embed2 = sklearn.preprocessing.normalize(embed2)
            diff = np.subtract(embed1, embed2)
            dist = np.sum(np.square(diff), 1)

        # Find the best threshold for the fold
        acc_train = np.zeros((nrof_thresholds))
        for threshold_idx, threshold in enumerate(thresholds):
            _, _, acc_train[threshold_idx] = calculate_accuracy(
                threshold, dist[train_set], actual_issame[train_set])
        best_threshold_index = np.argmax(acc_train)
        for threshold_idx, threshold in enumerate(thresholds):
            tprs[fold_idx, threshold_idx], fprs[fold_idx, threshold_idx], _ = calculate_accuracy(
                threshold, dist[test_set],
                actual_issame[test_set])
        _, _, accuracy[fold_idx] = calculate_accuracy(
            thresholds[best_threshold_index], dist[test_set],
            actual_issame[test_set])

    tpr = np.mean(tprs, 0)
    fpr = np.mean(fprs, 0)
    return tpr, fpr, accuracy


def calculate_accuracy(threshold, dist, actual_issame):
    predict_issame = np.less(dist, threshold)
    tp = np.sum(np.logical_and(predict_issame, actual_issame))
    fp = np.sum(np.logical_and(predict_issame, np.logical_not(actual_issame)))
    tn = np.sum(
        np.logical_and(np.logical_not(predict_issame),
                       np.logical_not(actual_issame)))
    fn = np.sum(np.logical_and(np.logical_not(predict_issame), actual_issame))

    tpr = 0 if (tp + fn == 0) else float(tp) / float(tp + fn)
    fpr = 0 if (fp + tn == 0) else float(fp) / float(fp + tn)
    acc = float(tp + tn) / dist.size
    return tpr, fpr, acc


def calculate_val(thresholds,
                  embeddings1,
                  embeddings2,
                  actual_issame,
                  far_target,
                  nrof_folds=10):
    assert (embeddings1.shape[0] == embeddings2.shape[0])
    assert (embeddings1.shape[1] == embeddings2.shape[1])
    nrof_pairs = min(len(actual_issame), embeddings1.shape[0])
    nrof_thresholds = len(thresholds)
    k_fold = LFold(n_splits=nrof_folds, shuffle=False)

    val = np.zeros(nrof_folds)
    far = np.zeros(nrof_folds)

    diff = np.subtract(embeddings1, embeddings2)
    dist = np.sum(np.square(diff), 1)
    indices = np.arange(nrof_pairs)

    for fold_idx, (train_set, test_set) in enumerate(k_fold.split(indices)):

        # Find the threshold that gives FAR = far_target
        far_train = np.zeros(nrof_thresholds)
        for threshold_idx, threshold in enumerate(thresholds):
            _, far_train[threshold_idx] = calculate_val_far(
                threshold, dist[train_set], actual_issame[train_set])
        if np.max(far_train) >= far_target:
            f = interpolate.interp1d(far_train, thresholds, kind='slinear')
            threshold = f(far_target)
        else:
            threshold = 0.0

        val[fold_idx], far[fold_idx] = calculate_val_far(
            threshold, dist[test_set], actual_issame[test_set])

    val_mean = np.mean(val)
    far_mean = np.mean(far)
    val_std = np.std(val)
    return val_mean, val_std, far_mean


def calculate_val_far(threshold, dist, actual_issame):
    predict_issame = np.less(dist, threshold)
    true_accept = np.sum(np.logical_and(predict_issame, actual_issame))
    false_accept = np.sum(
        np.logical_and(predict_issame, np.logical_not(actual_issame)))
    n_same = np.sum(actual_issame)
    n_diff = np.sum(np.logical_not(actual_issame))
    # print(true_accept, false_accept)
    # print(n_same, n_diff)
    val = float(true_accept) / float(n_same)
    far = float(false_accept) / float(n_diff)
    return val, far


def evaluate_identification_majority_voting(dict_pred_labels_probe, dict_true_labels_probe):
    num_tracks = int(len(dict_pred_labels_probe))
    num_hit_rank1, num_miss_rank1 = 0, 0
    for subj_track in list(dict_pred_labels_probe.keys()):
        counts_pred_label = np.bincount(dict_pred_labels_probe[subj_track])
        majority_voting_track_pred_label = np.argmax(counts_pred_label)

        counts_true_label = np.bincount(np.squeeze(dict_true_labels_probe[subj_track]))
        majority_voting_track_true_label = np.argmax(counts_true_label)
        
        if majority_voting_track_pred_label == majority_voting_track_true_label:
            num_hit_rank1 += 1
        else:
            num_miss_rank1 += 1
  
    acc_rank1 = num_hit_rank1 / num_tracks
    return acc_rank1


def evaluate_identification_indiv_samples(dict_pred_labels_probe, dict_true_labels_probe):
    num_frames = sum([len(dict_pred_labels_probe[subj_track]) for subj_track in dict_pred_labels_probe.keys()])
    num_hit_rank1, num_miss_rank1 = 0, 0
    for subj_track in list(dict_pred_labels_probe.keys()):
        # print(f'dict_pred_labels_probe[{subj_track}].shape:', dict_pred_labels_probe[subj_track].shape)
        # print(f'np.squeeze(dict_true_labels_probe[{subj_track}].shape:', np.squeeze(dict_true_labels_probe[subj_track]).shape)
        comparison_bool = dict_pred_labels_probe[subj_track] == np.squeeze(dict_true_labels_probe[subj_track].detach().cpu().numpy())
        # print('comparison_bool:', comparison_bool)
        # print('sum(comparison_bool):', sum(comparison_bool))
        num_hit_rank1  += sum(comparison_bool)
        num_miss_rank1 += sum(~comparison_bool)
        # print('num_hit_rank1:', num_hit_rank1)
        # print('num_miss_rank1:', num_miss_rank1)
        # sys.exit(0)
            
  
    acc_indiv_samples_rank1 = num_hit_rank1 / num_frames
    return acc_indiv_samples_rank1


def compute_similarities(A, B):
    norm_A = np.linalg.norm(A, axis=1, keepdims=True)
    norm_A[norm_A == 0] = 1.0 
    A_normalized = A / norm_A

    norm_B = np.linalg.norm(B, axis=1, keepdims=True)
    norm_B[norm_B == 0] = 1.0
    B_normalized = B / norm_B
    
    similarity_matrix = A_normalized @ B_normalized.T
    return similarity_matrix


@torch.no_grad()
def compute_embeddings(data, backbone, batch_size):
    embeddings = None
    ba = 0
    while ba < data.shape[0]:
        bb = min(ba + batch_size, data.shape[0])
        _data = data[ba:bb, :]
        img = ((_data / 255) - 0.5) / 0.5
        net_out: torch.Tensor = backbone(img)
        _embeddings = net_out.detach().cpu().numpy()
        if embeddings is None:
            embeddings = np.zeros((data.shape[0], _embeddings.shape[1]))
        embeddings[ba:bb, :] = _embeddings
        ba = bb
    return embeddings


@torch.no_grad()
def test_identification(data_set, backbone, batch_size, nfolds=10):
    print('testing identification...')
    data_gallery           = data_set[0]
    true_labels_gallery    = data_set[1]
    dict_data_probe        = data_set[2]
    dict_true_labels_probe = data_set[3]

    embeddings_gallery = compute_embeddings(data_gallery, backbone, batch_size)
    embeddings_gallery = sklearn.preprocessing.normalize(embeddings_gallery)

    dict_embeddings_probe = {}
    for subj_track in list(dict_data_probe.keys()):
        embeddings_probe = compute_embeddings(dict_data_probe[subj_track], backbone, batch_size)
        dict_embeddings_probe[subj_track] = embeddings_probe

    dict_similarities_probe = {}
    dict_pred_labels_probe = {}
    for subj_track in list(dict_data_probe.keys()):
        similarities_probe = compute_similarities(dict_embeddings_probe[subj_track], embeddings_gallery)
        dict_similarities_probe[subj_track] = similarities_probe
        track_pred_labels = np.argmax(similarities_probe, axis=1)
        dict_pred_labels_probe[subj_track] = track_pred_labels
    
    acc_rank1_major_voting = evaluate_identification_majority_voting(dict_pred_labels_probe, dict_true_labels_probe)
    acc_rank1_indiv_samples = evaluate_identification_indiv_samples(dict_pred_labels_probe, dict_true_labels_probe)
    return acc_rank1_major_voting, acc_rank1_indiv_samples





if __name__ == '__main__':
    args = parse_arguments()


    # Load pytorch model
    assert os.path.isfile(args.model), f"Error, no such model file: \'{args.model}\'"
    nets = []
    time0 = datetime.datetime.now()
    print(f'Loading trained model \'{args.model}\'...')
    weight = torch.load(args.model)
    resnet = get_model(args.network, dropout=0, fp16=False).cuda()
    resnet.load_state_dict(weight)
    model = torch.nn.DataParallel(resnet)
    model.eval()
    nets.append(model)
    time_now = datetime.datetime.now()
    diff = time_now - time0
    print('model loading time', diff.total_seconds(), 'seconds')


    # Load dataset
    assert os.path.exists(args.data_dir), f"Error, no such dataset dir: \'{args.data_dir}\'"
    ver_list = []
    ver_name_list = []
    image_size = [112, 112]
    print('image_size', image_size)
    if 'YouTubeFaces_TINY' in args.data_dir:
        print('Loading \'YouTubeFaces_TINY\' dataset...')
        dataset = Loader_YouTubeFacesTINY().load_dataset(args.data_dir, image_size, 'identification')
        ver_list.append(dataset)
        ver_name_list.append('YouTubeFaces_TINY')


    # Do verification
    for i in range(len(ver_list)):
        results = []
        for model in nets:
            acc_rank1_major_voting, acc_rank1_indiv_samples = test_identification(ver_list[i], model, args.batch_size, args.nfolds)
            print('[%s] Accuracy-Majority-Voting-Rank1: %1.5f' % (ver_name_list[i], acc_rank1_major_voting))
            print('[%s] Accuracy-Individ-Samples-Rank1: %1.5f' % (ver_name_list[i], acc_rank1_indiv_samples))

