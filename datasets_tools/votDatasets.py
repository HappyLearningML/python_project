# --------------------------------------------------------
# Python Single Object Tracking Evaluation
# Licensed under The MIT License [see LICENSE for details]
# Written by Fangyi Zhang
# @author fangyi.zhang@vipl.ict.ac.cn
# @project https://github.com/StrangerZhang/pysot-toolkit.git
# Revised for SiamMask by foolwood
# --------------------------------------------------------
import os
import json
import cv2
import numpy as np

from glob import glob
from tqdm import tqdm


class Dataset(object):
    def __init__(self, name, dataset_root):
        self.name = name
        self.dataset_root = dataset_root
        self.videos = None

    def __getitem__(self, idx):
        if isinstance(idx, str):
            return self.videos[idx]
        elif isinstance(idx, int):
            return self.videos[sorted(list(self.videos.keys()))[idx]]

    def __len__(self):
        return len(self.videos)

    def __iter__(self):
        keys = sorted(list(self.videos.keys()))
        for key in keys:
            yield self.videos[key]

    def set_tracker(self, path, tracker_names):
        """
        Args:
            path: path to tracker results,
            tracker_names: list of tracker name
        """
        self.tracker_path = path
        self.tracker_names = tracker_names


class Video(object):
    def __init__(self, name, root, video_dir, init_rect, img_names,
            gt_rect, attr):
        self.name = name
        self.video_dir = video_dir
        self.init_rect = init_rect
        self.gt_traj = gt_rect
        self.attr = attr
        self.pred_trajs = {}
        self.img_names = [os.path.join(root, x) for x in img_names]
        self.imgs = None

    def load_tracker(self, path, tracker_names=None, store=True):
        """
        Args:
            path(str): path to result
            tracker_name(list): name of tracker
        """
        if not tracker_names:
            tracker_names = [x.split('/')[-1] for x in glob(path)
                    if os.path.isdir(x)]
        if isinstance(tracker_names, str):
            tracker_names = [tracker_names]
        for name in tracker_names:
            traj_file = os.path.join(path, name, self.name+'.txt')
            if os.path.exists(traj_file):
                with open(traj_file, 'r') as f :
                    pred_traj = [list(map(float, x.strip().split(',')))
                            for x in f.readlines()]
                if len(pred_traj) != len(self.gt_traj):
                    print(name, len(pred_traj), len(self.gt_traj), self.name)
                if store:
                    self.pred_trajs[name] = pred_traj
                else:
                    return pred_traj
            else:
                print(traj_file)
        self.tracker_names = list(self.pred_trajs.keys())

    def load_img(self):
        if self.imgs is None:
            self.imgs = [cv2.imread(x)
                            for x in self.img_names]
            self.width = self.imgs[0].shape[1]
            self.height = self.imgs[0].shape[0]

    def free_img(self):
        self.imgs = None

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, idx):
        if self.imgs is None:
            return cv2.imread(self.img_names[idx]), \
                    self.gt_traj[idx]
        else:
            return self.imgs[idx], self.gt_traj[idx]

    def __iter__(self):
        for i in range(len(self.img_names)):
            if self.imgs is not None:
                yield self.imgs[i], self.gt_traj[i]
            else:
                yield cv2.imread(self.img_names[i]), \
                        self.gt_traj[i]

		
class VOTVideo(Video):
    """
    Args:
        name: video name
        root: dataset root
        video_dir: video directory
        init_rect: init rectangle
        img_names: image names
        gt_rect: groundtruth rectangle
        camera_motion: camera motion tag
        illum_change: illum change tag
        motion_change: motion change tag
        size_change: size change
        occlusion: occlusion
    """
    def __init__(self, name, root, video_dir, init_rect, img_names, gt_rect,
            camera_motion, illum_change, motion_change, size_change, occlusion, width, height):
        super(VOTVideo, self).__init__(name, root, video_dir, init_rect, img_names, gt_rect, None)
        self.tags= {'all': [1] * len(gt_rect)}
        self.tags['camera_motion'] = camera_motion
        self.tags['illum_change'] = illum_change
        self.tags['motion_change'] = motion_change
        self.tags['size_change'] = size_change
        self.tags['occlusion'] = occlusion

        self.width = width
        self.height = height

        # empty tag
        all_tag = [v for k, v in self.tags.items() if len(v) > 0 ]
        self.tags['empty'] = np.all(1 - np.array(all_tag), axis=1).astype(np.int32).tolist()

        self.tag_names = list(self.tags.keys())

    def select_tag(self, tag, start=0, end=0):
        if tag == 'empty':
            return self.tags[tag]
        return self.tags[tag][start:end]

    def load_tracker(self, path, tracker_names=None, store=True):
        """
        Args:
            path(str): path to result
            tracker_name(list): name of tracker
        """
        if not tracker_names:
            tracker_names = [x.split('/')[-1] for x in glob(path)
                    if os.path.isdir(x)]
        if isinstance(tracker_names, str):
            tracker_names = [tracker_names]
        for name in tracker_names:
            traj_files = glob(os.path.join(path, name, 'baseline', self.name, '*0*.txt'))
            if len(traj_files) == 15:
                traj_files = traj_files
            else:
                traj_files = traj_files[0:1]
            pred_traj = []
            for traj_file in traj_files:
                with open(traj_file, 'r') as f:
                    traj = [list(map(float, x.strip().split(',')))
                            for x in f.readlines()]
                    pred_traj.append(traj)
            if store:
                self.pred_trajs[name] = pred_traj
            else:
                return pred_traj


class VOTDataset(Dataset):
    """
    Args:
        name: dataset name, should be 'VOT2018', 'VOT2016'
        dataset_root: dataset root
        load_img: wether to load all imgs
    """
    def __init__(self, name, dataset_root):
        super(VOTDataset, self).__init__(name, dataset_root)
        try:
            with open(os.path.join(dataset_root, name+'.json'), 'r') as f:
                meta_data = json.load(f)
        except:
            download_str = '# download json file for eval toolkit\n'+\
                           'cd $SiamMask/data\n'+\
                           'wget http://www.robots.ox.ac.uk/~qwang/VOT2016.json\n'+\
                           'wget http://www.robots.ox.ac.uk/~qwang/VOT2018.json'
            print(download_str)
            exit()

        # load videos
        pbar = tqdm(meta_data.keys(), desc='loading '+name, ncols=100)
        self.videos = {}
        for video in pbar:
            pbar.set_postfix_str(video)
            self.videos[video] = VOTVideo(video,
                                          dataset_root,
                                          meta_data[video]['video_dir'],
                                          meta_data[video]['init_rect'],
                                          meta_data[video]['img_names'],
                                          meta_data[video]['gt_rect'],
                                          meta_data[video]['camera_motion'],
                                          meta_data[video]['illum_change'],
                                          meta_data[video]['motion_change'],
                                          meta_data[video]['size_change'],
                                          meta_data[video]['occlusion'],
                                          meta_data[video]['width'],
                                          meta_data[video]['height'])

        self.tags = ['all', 'camera_motion', 'illum_change', 'motion_change',
                     'size_change', 'occlusion', 'empty']
