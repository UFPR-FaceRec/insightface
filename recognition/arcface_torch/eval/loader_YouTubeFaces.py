import sys, os
import cv2
import numpy as np
import torch
import mxnet as mx
from mxnet import ndarray as nd
import copy
import re
import random


def natural_sort(l):
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]
    return sorted(l, key=alphanum_key)


def list_immediate_subdirectories(directory_path=''):
    subdirectories = []
    for item in os.listdir(directory_path):
        item_path = os.path.join(directory_path, item)
        if os.path.isdir(item_path):
            subdirectories.append(item)
    subdirectories = natural_sort(subdirectories)
    return subdirectories


def get_all_files_in_path(folder_path, file_extension=['.jpg','.jpeg','.png'], pattern=''):
    file_list = []
    for root, _, files in os.walk(folder_path):
        for filename in files:
            path_file = os.path.join(root, filename)
            for ext in file_extension:
                if pattern in path_file and path_file.lower().endswith(ext.lower()):
                    file_list.append(path_file)
                    # print(f'Found files: {len(file_list)}', end='\r')
    # print()
    file_list = natural_sort(file_list)
    return file_list


def get_tracks_files_in_path(folder_path, file_extension=['.jpg','.jpeg','.png'], pattern=''):
    tracks_dirs = list_immediate_subdirectories(folder_path)
    dict_tracks_files = {track_name:get_all_files_in_path(f"{folder_path}/{track_name}", file_extension=file_extension) for track_name in tracks_dirs}
    return dict_tracks_files



class Loader_YouTubeFaces:
    def __init__(self):
        pass


    def make_verification_protocol(self, dict_paths_gallery, dict_paths_probe):
        protocol = []
        num_pos_pairs, num_neg_pairs = 0, 0
        subjs_list = list(dict_paths_gallery.keys())

        # make positive pairs
        for subj in subjs_list:
            for gallery_sample in dict_paths_gallery[subj]:
                for probe_sample in dict_paths_probe[subj]:
                    # print(f"{subj} - {gallery_sample} - {probe_sample}")
                    pair = {}
                    pair['sample0'] = gallery_sample
                    pair['sample1'] = probe_sample
                    pair['pair_label'] = 1
                    protocol.append(pair)
                    num_pos_pairs += 1

        # make negative pairs
        random.seed(440)
        for i, subj_gal in enumerate(subjs_list):
            for gallery_sample in dict_paths_gallery[subj_gal]:
                possible_proble_indices = random.sample(range(len(subjs_list)), len(subjs_list))
                possible_proble_indices.remove(i)
                j = possible_proble_indices[0]
                subj_prob = subjs_list[j]
                assert subj_gal != subj_prob
                for probe_sample in dict_paths_probe[subj_prob]:
                    pair = {}
                    pair['sample0'] = gallery_sample
                    pair['sample1'] = probe_sample
                    pair['pair_label'] = 0
                    protocol.append(pair)
                    num_neg_pairs += 1

        random.shuffle(protocol)
        print(f'num_pos_pairs: {num_pos_pairs}    num_neg_pairs: {num_neg_pairs}')
        assert len(protocol) == num_pos_pairs+num_neg_pairs
        return protocol


    def load_dataset(self, data_dir='', image_size=[112,112], task='verification'):
        path_dir_gallery = f"{data_dir}/gallery"
        path_dir_probe   = f"{data_dir}/probe"
        subjs_list = list_immediate_subdirectories(path_dir_gallery)
        print('num subjs:', len(subjs_list))
        subjs_labels = list(range(len(subjs_list)))

        if task == 'verification':
            dict_paths_gallery = {subj:get_all_files_in_path(f"{path_dir_gallery}/{subj}", file_extension=['.jpg','.jpeg','.png']) for subj in subjs_list}
            dict_paths_probe   = {subj:get_all_files_in_path(f"{path_dir_probe}/{subj}", file_extension=['.jpg','.jpeg','.png']) for subj in subjs_list}

            pairs_orig = self.make_verification_protocol(dict_paths_gallery, dict_paths_probe)

            # Load images
            data_list = []
            for flip in [0, 1]:
                data = torch.empty((len(pairs_orig)*2, 3, image_size[0], image_size[1]))
                data_list.append(data)

            issame_list = np.array([bool(pairs_orig[i]['pair_label']) for i in range(len(pairs_orig))])

            for idx in range(len(pairs_orig) * 2):
                idx_pair = int(idx/2)
                if idx % 2 == 0:
                    img_path = pairs_orig[idx_pair]['sample0']
                else:
                    img_path = pairs_orig[idx_pair]['sample1']
                assert os.path.isfile(img_path), f"Error, file not found: '{img_path}'"
                img = cv2.imread(img_path)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = mx.nd.array(img)

                if img.shape[1] != image_size[0]:
                    img = mx.image.resize_short(img, image_size[0])
                img = nd.transpose(img, axes=(2, 0, 1))
                for flip in [0, 1]:
                    if flip == 1:
                        img = mx.ndarray.flip(data=img, axis=2)
                    data_list[flip][idx][:] = torch.from_numpy(img.asnumpy())
                if idx % 100 == 0:
                    print(f"loading pairs {idx}/{len(pairs_orig)*2}", end='\r')
            print('\n    ', data_list[0].shape)
            return (data_list, issame_list)


        elif task == 'identification':
            # dict_paths_gallery = {subj:get_all_files_in_path(f"{path_dir_gallery}/{subj}", file_extension=['.jpg','.jpeg','.png']) for subj in subjs_list}
            # dict_paths_tracks_probe = {subj:get_tracks_files_in_path(f"{path_dir_probe}/{subj}", file_extension=['.jpg','.jpeg','.png']) for subj in subjs_list}
            dict_paths_gallery, dict_paths_tracks_probe = {}, {}
            for idx_subj, subj in enumerate(subjs_list):
                print(f"{idx_subj}/{len(subjs_list)} - Loading subj path \'{subj}\'                ", end="\r")
                dict_paths_gallery[subj] = get_all_files_in_path(f"{path_dir_gallery}/{subj}", file_extension=['.jpg','.jpeg','.png'])
                dict_paths_tracks_probe[subj] = get_tracks_files_in_path(f"{path_dir_probe}/{subj}", file_extension=['.jpg','.jpeg','.png'])
            print()

            # Add labels to paths
            for subj_name, subj_label in zip(subjs_list, subjs_labels):
                for idx_img_path, img_path in enumerate(dict_paths_gallery[subj_name]):
                    dict_paths_gallery[subj_name][idx_img_path] = [img_path, subj_label]
            
                for idx_track_name, track_name in enumerate(list(dict_paths_tracks_probe[subj_name].keys())):
                    for idx_img_path, img_path in enumerate(dict_paths_tracks_probe[subj_name][track_name]):
                        dict_paths_tracks_probe[subj_name][track_name][idx_img_path] = [img_path, subj_label]

            # Load gallery images 
            data_gallery   = torch.zeros(len(dict_paths_gallery), 3, image_size[0], image_size[1])
            true_labels_gallery = torch.zeros(len(dict_paths_gallery), 1)
            for idx_subj, (subj_name, subj_label) in enumerate(zip(subjs_list, subjs_labels)):
                print(f"{idx_subj}/{len(subjs_list)} - Loading gallery imgs \'{subj_name}\'                ", end="\r")
                img = cv2.imread(dict_paths_gallery[subj_name][0][0])
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = mx.nd.array(img)
                img = nd.transpose(img, axes=(2, 0, 1))
                data_gallery[idx_subj][:] = torch.from_numpy(img.asnumpy())
                true_labels_gallery[idx_subj]  = subj_label
            print()

            # Load probe images 
            dict_data_probe        = {}
            dict_true_labels_probe = {}
            for idx_subj, (subj_name, subj_label) in enumerate(zip(subjs_list, subjs_labels)):
                dict_data_probe[subj_name]        = {}
                dict_true_labels_probe[subj_name] = {}
                for idx_track_name, track_name in enumerate(list(dict_paths_tracks_probe[subj_name].keys())):
                    data_track_probe     = torch.zeros(len(dict_paths_tracks_probe[subj_name][track_name]), 3, image_size[0], image_size[1])
                    labels_track_gallery = torch.zeros(len(dict_paths_tracks_probe[subj_name][track_name]), 1)
                    for idx_img_path, img_path in enumerate(dict_paths_tracks_probe[subj_name][track_name]):
                        print(f"{idx_subj}/{len(subjs_list)} - Loading probe imgs \'{subj_name}\' - track {track_name} - img {idx_img_path}/{len(dict_paths_tracks_probe[subj_name][track_name])}                     ", end="\r")
                        img = cv2.imread(dict_paths_tracks_probe[subj_name][track_name][idx_img_path][0])
                        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                        img = mx.nd.array(img)
                        img = nd.transpose(img, axes=(2, 0, 1))
                        data_track_probe[idx_img_path][:]  = torch.from_numpy(img.asnumpy())
                        labels_track_gallery[idx_img_path] = subj_label
                    dict_data_probe[subj_name][track_name] = data_track_probe
                    dict_true_labels_probe[subj_name][track_name] = labels_track_gallery
            print()

            dataset = {
                "subjs_list":              subjs_list,
                "subjs_labels":            subjs_labels, 
                "dict_paths_gallery":      dict_paths_gallery,
                "data_gallery":            data_gallery,
                "true_labels_gallery":     true_labels_gallery,
                "dict_paths_tracks_probe": dict_paths_tracks_probe,
                "dict_data_probe":         dict_data_probe,
                "dict_true_labels_probe":  dict_true_labels_probe
            }
            return dataset
