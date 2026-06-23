import os
import json
import numpy as np
from pytracking.evaluation.data import Sequence, BaseDataset, SequenceList

ScaleDict = {'infrared': np.array([[960 / 640, 540 / 512, 960 / 640, 540 / 512]]),
             'visible': np.array([[960 / 1920, 540 / 1080, 960 / 1920, 540 / 1080]])}


class UAV300Dataset_BothConfig(BaseDataset):
    def __init__(self, split='test', mode='infrared', uav300_path="", full=False):
        super().__init__()
        self.split = split
        self.mode = mode
        self.base_path = uav300_path
        if not full:
            with open(os.path.join(self.base_path, split + '_list.txt'), 'r') as f:
                self.sequence_info_list = [line.split('\t') for line in f.readlines()]
            self.sequence_list = SequenceList([self._construct_sequence(s) for s in self.sequence_info_list])
        else:
            self.sequence_info_list = os.listdir(os.path.join(self.base_path, split))
            self.sequence_list = SequenceList([self._construct_sequence2(s) for s in self.sequence_info_list])

    def _construct_sequence(self, sequence_info, init_omit=0):
        name, name_frag, start, end = sequence_info
        start, end = int(start), int(end) + 1

        if self.mode == 'join':
            frames = [[os.path.join(self.base_path, self.split, name, 'infrared', "{}.jpg".format(num)),
                       os.path.join(self.base_path, self.split, name, 'visible', "{}.jpg".format(num))] for num in
                      range(start + init_omit, end)]

            with open(os.path.join(self.base_path, self.split, name, 'infrared.json'), 'r') as f:
                ground_truth_rect_inf = json.load(f)['gt_rect'][start + init_omit:end]
                ground_truth_rect_inf = np.array(ground_truth_rect_inf) * ScaleDict['infrared']
                ground_truth_rect_inf = ground_truth_rect_inf.astype(np.int16)
            with open(os.path.join(self.base_path, self.split, name, 'visible.json'), 'r') as f:
                ground_truth_rect_vis = json.load(f)['gt_rect'][start + init_omit:end]
                ground_truth_rect_vis = np.array(ground_truth_rect_vis) * ScaleDict['visible']
                ground_truth_rect_vis = ground_truth_rect_vis.astype(np.int16)
            ground_truth_rect = [ground_truth_rect_inf, ground_truth_rect_vis]

        else:
            frames = [os.path.join(self.base_path, self.split, name, self.mode, "{}.jpg".format(num)) for num in
                      range(start + init_omit, end)]
            with open(os.path.join(self.base_path, self.split, name, self.mode + '.json'), 'r') as f:
                ground_truth_rect = json.load(f)['gt_rect'][start + init_omit:end]
                ground_truth_rect = np.array(ground_truth_rect) * ScaleDict[self.mode]
                ground_truth_rect = ground_truth_rect.astype(np.int16)

        return Sequence("{}({})".format(name, name_frag), frames, 'uav300({})'.format(self.mode), ground_truth_rect,
                        object_class='uav')

    def _construct_sequence2(self, name):
        if self.mode == 'join':
            with open(os.path.join(self.base_path, self.split, name, 'infrared.json'), 'r') as f:
                ground_truth_rect_inf = [u if len(u) == 4 else [0, 0, 0, 0] for u in json.load(f)['gt_rect']]
                ground_truth_rect_inf = (np.array(ground_truth_rect_inf) * ScaleDict['infrared']).astype(np.int16)
            with open(os.path.join(self.base_path, self.split, name, 'visible.json'), 'r') as f:
                ground_truth_rect_vis = [u if len(u) == 4 else [0, 0, 0, 0] for u in json.load(f)['gt_rect']]
                ground_truth_rect_vis = (np.array(ground_truth_rect_vis) * ScaleDict['visible']).astype(np.int16)
            ground_truth_rect = [ground_truth_rect_inf, ground_truth_rect_vis]
            frames = [[os.path.join(self.base_path, self.split, name, 'infrared', "{}.jpg".format(num)),
                       os.path.join(self.base_path, self.split, name, 'visible', "{}.jpg".format(num))] for num in
                      range(ground_truth_rect_vis.shape[0])]
        else:
            with open(os.path.join(self.base_path, self.split, name, self.mode + '.json'), 'r') as f:
                ground_truth_rect = [u if len(u) == 4 else [0, 0, 0, 0] for u in json.load(f)['gt_rect']]
                ground_truth_rect = (np.array(ground_truth_rect) * ScaleDict[self.mode]).astype(np.int16)

            frames = [os.path.join(self.base_path, self.split, name, self.mode, "{}.jpg".format(num)) for num in
                      range(ground_truth_rect.shape[0])]
        return Sequence(name, frames, 'uav300({})'.format(self.mode), ground_truth_rect, object_class='uav')

    def get_sequence_list(self):
        return self.sequence_list

    def __len__(self):
        return len(self.sequence_info_list)
