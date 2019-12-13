#-*-coding:utf-8-*-
"""
本功能主要是用来导入关键点列表数据的。通过读取关键点列表。来导入图片
本功能使用了多线程来进行。目前支持5个关键点和83个关键点。后面会增加其他关键点个数
"""

import os
import sys
import numpy as np
import cv2
import math
import signal
import random
import time
from multiprocessing import Process, Queue, Event

exitEvent = Event() # for noitfy all process exit.

def handler(sig_num, stack_frame):
    global exitEvent
    exitEvent.set()
signal.signal(signal.SIGINT, handler)

class LandmarkHelper(object):
    '''
    Helper for different landmark type
    '''
    @classmethod
    def parse(cls, line, landmark_type):
        '''
        use for parse txt line to get file path and landmarks and so on
        Args:
            cls: this class
            line: line of input txt
            landmark_type: len of landmarks
        Return:
            see child parse
        Raises:
            unsupport type
        '''
        if landmark_type == 5:
            return cls.__landmark5_txt_parse(line)
        elif landmark_type == 68:
            return cls.__landmark68_txt_parse(line)
        elif landmark_type == 83:
            return cls.__landmark83_txt_parse(line)
        else:
            raise Exception("Unsupport landmark type...")

    @staticmethod
    def flip(a, landmark_type):
        '''
        use for flip landmarks. Because we have to renumber it after flip
        Args:
            a: original landmarks
            landmark_type: len of landmarks
        Returns:
            landmarks: new landmarks
        Raises:
            unsupport type
        '''
        if landmark_type == 5:
            landmarks = np.concatenate((a[1,:], a[0,:], a[2,:], a[4,:], a[3,:]), axis=0)
        elif landmark_type == 68:
            pass
        elif landmark_type == 83:
            landmarks = np.concatenate((a[10:19][::-1], a[9:10], a[0:9][::-1], a[35:36],
                a[36:43][::-1], a[43:48][::-1], a[48:51][::-1], a[19:20], a[20:27][::-1],
                a[27:32][::-1], a[32:35][::-1], a[56:60][::-1], a[55:56], a[51:55][::-1],
                a[60:61], a[61:72][::-1], a[72:73], a[73:78][::-1], a[80:81], a[81:82],
                a[78:79], a[79:80], a[82:83]), axis=0)
        else:
            raise Exception("Unsupport landmark type...")
        return landmarks.reshape([-1, 2])

    @staticmethod
    def get_scales(landmark_type):
        '''
        use for scales bbox according to bbox of landmarks
        Args:
            landmark_type: len of landmarks
        Returns:
            (min, max), min crop
        Raises:
            unsupport type
        '''
        if landmark_type == 5:
            return (2.7, 3.3), 4.5
        elif landmark_type == 68:
            pass
        elif landmark_type == 83:
            return (1.2, 1.5), 2.6
        else:
            raise Exception("Unsupport landmark type...")

    @staticmethod
    def __landmark5_txt_parse(line):
        '''
        Args:
            line: 0=file path, 1=[0:4] is bbox and [4:] is landmarks
        Returns:
            file path and landmarks with numpy type
        Raises:
            No
        '''
        a = line.split()
        data = list(map(int, a[1:]))
        pts = data[4:] # x1,y1,x2,y2...
        return a[0], np.array(pts).reshape((-1, 2))

    @staticmethod
    def __landmark68_txt_parse(line):
        '''
        Args:
            line: 0=file path, 1 = landmarks68, 2=bbox, 4=pose
        Returns:
            file path and landmarks with numpy type
        Raises:
            No
        '''
        pass

    @staticmethod
    def __landmark83_txt_parse(line):
        '''
        Args:
            line: 0=file path, 1=landmarks83, 2=bbox, 4=pose
        Returns:
            file path and landmarks with numpy type
        Raises:
            No
        '''
        a = line.split()
        a1 = np.fromstring(a[1], dtype=int, count=166, sep=',')
        a1 = a1.reshape((-1, 2))
        return a[0], a1


class LandmarkAugment(object):
    '''
    Facial landmarks augmentation.
    '''
    def __init__(self):
        pass

    def augment(self, image, landmarks, output_size, max_angle, scale_range):
        '''Do image augment.
        Args:
            image: a numpy type
            landmarks: face landmarks with format numpy [(x1, y1), (x2, y2), ...]
            output_size: target image size with format (w, h)
            max_angle: random to rotate in [-max_angle, max_angle]. range is 0-180.
            scale_range: scale bbox in (min, max). eg: (13.0, 15.0)
        Returns:
            an image with target size will be return
        Raises:
            No
        '''
        image, landmarks = self.__flip(image, landmarks)
        image, landmarks = self.__rotate(image, landmarks, max_angle)
        image, landmarks = self.__scale_and_shift(image, landmarks, scale_range, output_size)
        landmarks = landmarks.flatten()
        return image, landmarks

    def mini_crop_by_landmarks(self, sample_list, pad_rate, img_format):
        '''Crop full image to mini. Only keep vaild image to save
        Args:
            sample_list: (image, landmarks)
            pad_rate: up scale rate
            img_format: "RGB" or "BGR"
        Returns:
            new sample list
        Raises:
            No
        '''
        new_sample_list = []
        for sample in sample_list:
            image = cv2.imread(sample[0])
            if img_format == 'RGB':
                image= cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            landmarks = sample[1]
            (x1, y1, x2, y2), _, _, _ = self.get_bbox_of_landmarks(image, landmarks, pad_rate, 0.5)
            new_sample_list.append(
                (cv2.imencode('.jpg', image[y1:y2, x1:x2])[1], landmarks - (x1, y1))
            )
        return new_sample_list

    def __flip(self, image, landmarks, run_prob=0.5):
        '''Do image flip. Only for horizontal
        Args:
            image: a numpy type
            landmarks: face landmarks with format [(x1, y1), (x2, y2), ...]
            run_prob: probability to do this operate. 0.0-1.0
        Returns:
            an image and landmarks will be returned
        Raises:
            Unsupport count of landmarks
        '''
        if np.random.rand() < run_prob:
            return image, landmarks
        image = np.fliplr(image)
        landmarks[:, 0] = image.shape[1] - landmarks[:, 0]
        landmarks = LandmarkHelper.flip(landmarks, landmarks.shape[0])
        return image, landmarks

    def __rotate(self, image, landmarks, max_angle):
        '''Do image rotate.
        Args:
            image: a numpy type
            landmarks: face landmarks with format [(x1, y1), ...]. range is 0-w or h in int
            max_angle: random to rotate in [-max_angle, max_angle]. range is 0-180.
        Returns:
            an image and landmarks will be returned
        Raises:
            No
        '''
        c_x = (min(landmarks[:, 0]) + max(landmarks[:, 0])) / 2
        c_y = (min(landmarks[:, 1]) + max(landmarks[:, 1])) / 2
        h, w = image.shape[:2]
        angle = np.random.randint(-max_angle, max_angle)
        M = cv2.getRotationMatrix2D((c_x, c_y), angle, 1)
        image = cv2.warpAffine(image, M, (w, h)) 
        b = np.ones((landmarks.shape[0], 1))
        d = np.concatenate((landmarks, b), axis=1)
        landmarks = np.dot(d, np.transpose(M))
        return image, landmarks

    def __scale_and_shift(self, image, landmarks, scale_range, output_size):
        '''Auto generate bbox and then random to scale and shift it.
        Args:
            image: a numpy type
            landmarks: face landmarks with format [(x1, y1), ...]. range is 0-w or h in int
            scale_range: scale bbox in (min, max). eg: (1.3, 1.5)
            output_size: output size of image
        Returns:
            an image and landmarks will be returned
        Raises:
            No
        '''
        (x1, y1, x2, y2), new_size, need_pad, (p_x, p_y, p_w, p_h) = self.get_bbox_of_landmarks(
            image, landmarks, scale_range, shift_rate=0.3)
        box_image = image[y1:y2, x1:x2]
        if need_pad:
            box_image = np.lib.pad(box_image, ((p_y, p_h), (p_x, p_w), (0,0)), 'constant')
        box_image = cv2.resize(box_image, (output_size, output_size))
        landmarks = (landmarks - (x1 - p_x, y1 - p_y)) / (new_size, new_size)
        return box_image, landmarks

    def get_bbox_of_landmarks(self, image, landmarks, scale_range, shift_rate=0.3):
        '''According to landmark box to generate a new bigger bbox
        Args:
            image: a numpy type
            landmarks: face landmarks with format [(x1, y1), ...]. range is 0-w or h in int
            scale_range: scale bbox in (min, max). eg: (1.3, 1.5)
            shift_rate: up,down,left,right to shift
        Returns:
            return new bbox and other info
        Raises:
            No
        '''
        ori_h, ori_w = image.shape[:2]
        x = int(min(landmarks[:, 0]))
        y = int(min(landmarks[:, 1]))
        w = int(max(landmarks[:, 0]) - x)
        h = int(max(landmarks[:, 1]) - y)
        if type(scale_range) == float:
            scale = scale_range
        else:
            scale = np.random.randint(int(scale_range[0]*100.0), int(scale_range[1]*100.0)) / 100.0
        new_size = int(max(w, h) * scale)
        if shift_rate >= 0.5:
            x1 = x - (new_size - w) / 2
            y1 = y - (new_size - h) / 2
        else:
            x1 = x - np.random.randint(int((new_size-w)*shift_rate), int((new_size-w)*(1.0-shift_rate)))
            y1 = y - np.random.randint(int((new_size-h)*shift_rate), int((new_size-h)*(1.0-shift_rate)))
        x2 = x1 + new_size
        y2 = y1 + new_size
        need_pad = False
        p_x, p_y, p_w, p_h = 0, 0, 0, 0
        if x1 < 0:
            p_x = -x1
            x1 = 0
            need_pad = True
        if y1 < 0:
            p_y = -y1
            y1 = 0
            need_pad = True
        if x2 > ori_w:
            p_w = x2 - ori_w
            x2 = ori_w
            need_pad = True
        if y2 > ori_h:
            p_h = y2 - ori_h
            y2 = ori_h
            need_pad = True
        return (x1, y1, x2, y2), new_size, need_pad, (p_x, p_y, p_w, p_h)

class BatchReader():
    def __init__(self, **kwargs):
        # param
        self._kwargs = kwargs
        self._batch_size = kwargs['batch_size']
        self._process_num = kwargs['process_num']
        # total lsit
        self._sample_list = [] # each item: (filepath, landmarks, ...)
        self._total_sample = 0
        # real time buffer
        self._process_list = []
        self._output_queue = []
        for i in range(self._process_num):
            self._output_queue.append(Queue(maxsize=3)) # for each process
        # epoch
        self._idx_in_epoch = 0
        self._curr_epoch = 0
        self._max_epoch = kwargs['max_epoch']
        # start buffering
        self._start_buffering(kwargs['input_paths'], kwargs['landmark_type'])

    def batch_generator(self):
        __curr_queue = 0
        while True:
            self.__update_epoch()
            while True:
                __curr_queue += 1
                if __curr_queue >= self._process_num:
                    __curr_queue = 0
                try:
                    image_list, landmarks_list = self._output_queue[__curr_queue].get(block=True, timeout=0.01)
                    break
                except Exception as ex:
                    pass
            yield image_list, landmarks_list

    def get_epoch(self):
        return self._curr_epoch

    def should_stop(self):
        if exitEvent.is_set() or self._curr_epoch > self._max_epoch:
            exitEvent.set()
            self.__clear_and_exit()
            return True
        else:
            return False

    def __clear_and_exit(self):
        print ("[Exiting] Clear all queue.")
        while True:
            time.sleep(1)
            _alive = False
            for i in range(self._process_num):
                try:
                    self._output_queue[i].get(block=True, timeout=0.01)
                    _alive = True
                except Exception as ex:
                    pass
            if _alive == False: break
        print ("[Exiting] Confirm all process is exited.")
        for i in range(self._process_num):
            if self._process_list[i].is_alive():
                print ("[Exiting] Force to terminate process %d"%(i))
                self._process_list[i].terminate()
        print ("[Exiting] Batch reader clear done!")

    def _start_buffering(self, input_paths, landmark_type):
        if type(input_paths) in [str]:
            input_paths = [input_paths]
        for input_path in input_paths:
            for line in open(input_path):
                info = LandmarkHelper.parse(line, landmark_type)
                self._sample_list.append(info)
        self._total_sample = len(self._sample_list)
        num_per_process = int(math.ceil(self._total_sample / float(self._process_num)))
        for idx, offset in enumerate(range(0, self._total_sample, num_per_process)):
            p = Process(target=self._process, args=(idx, self._sample_list[offset: offset+num_per_process]))
            p.start()
            self._process_list.append(p)

    def _process(self, idx, sample_list):
        __landmark_augment = LandmarkAugment()
        # read all image to memory to speed up!
        if self._kwargs['buffer2memory']:
            print ("[Process %d] Start to read image to memory! Count=%d"%(idx, len(sample_list)))
            sample_list = __landmark_augment.mini_crop_by_landmarks(
                sample_list, LandmarkHelper.get_scales(self._kwargs['landmark_type'])[1], self._kwargs['img_format'])
            print ("[Process %d] Read all image to memory finish!"%(idx))
        sample_cnt = 0 # count for one batch
        image_list, landmarks_list = [], [] # one batch list
        while True:
            for sample in sample_list:
                # preprocess
                if type(sample[0]) in [str]:
                    image = cv2.imread(sample[0])
                    if self._kwargs['img_format'] == 'RGB':
                        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                else:
                    image = cv2.imdecode(sample[0], cv2.CV_LOAD_IMAGE_COLOR)
                landmarks = sample[1].copy()# keep deep copy
                scale_range = LandmarkHelper.get_scales(self._kwargs['landmark_type'])[0]
                image_new, landmarks_new = __landmark_augment.augment(image, landmarks, self._kwargs['img_size'],
                                            self._kwargs['max_angle'], scale_range)
                # sent a batch
                sample_cnt += 1
                image_list.append(image_new)
                landmarks_list.append(landmarks_new)
                if sample_cnt >= self._kwargs['batch_size']:
                    self._output_queue[idx].put((np.array(image_list), np.array(landmarks_list)))
                    sample_cnt = 0
                    image_list, landmarks_list = [], []
                # if exit
                if exitEvent.is_set():
                    break
            if exitEvent.is_set():
                break
            np.random.shuffle(sample_list)

    def __update_epoch(self):
        self._idx_in_epoch += self._batch_size
        if self._idx_in_epoch > self._total_sample:
            self._curr_epoch += 1
            self._idx_in_epoch = 0

# use for unit test
if __name__ == '__main__':
    kwargs = {
        'input_paths': "/world/data-c9/liubofang/dataset_original/CelebA/CelebA_19w_83points_bbox_angle.txt",
        'landmark_type': 83,  #这里可以理解为label的数量。可以改为
        'batch_size': 512,
        'process_num': 2,
        'img_format': 'RGB',
        'img_size': 128,
        'max_angle': 10,
        'max_epoch':1,
        'buffer2memory': False,
    }
    b = BatchReader(**kwargs)
    g = b.batch_generator()
    output_folder = "output_tmp/"
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    import time
    start_time = time.time()
    while not b.should_stop():
        end_time = time.time()
        print ("get new batch...epoch: %d. cost: %.3f"%(
                b.get_epoch(), end_time-start_time))
        start_time = end_time
        batch_image, batch_landmarks = g.next()
        for idx, (image, landmarks) in enumerate(zip(batch_image, batch_landmarks)):
            if idx > 20: # only see first 10
                break
            landmarks = landmarks.reshape([-1, 2])
            image = cv2.resize(image, (1080, 1080)) # for debug
            for i, l in enumerate(landmarks):
                ii = tuple(l * image.shape[:2])
                cv2.circle(image, (int(ii[0]), int(ii[1])), 1, (0,255,0), -1)
                cv2.putText(image, str(i), (int(ii[0]), int(ii[1])), cv2.FONT_HERSHEY_SIMPLEX, 1, 2)
            cv2.imwrite("%s/%d.jpg"%(output_folder, idx), image)