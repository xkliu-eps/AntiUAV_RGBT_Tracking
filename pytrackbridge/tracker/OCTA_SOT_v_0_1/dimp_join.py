from pytracking.tracker.base import BaseTracker
import torch
import torch.nn.functional as F
import math
import time
import importlib
from _collections import OrderedDict
import cv2
import numpy as np
from pytracking.features.preprocessing import numpy_to_torch
from pytracking.utils.plotting import show_tensor, plot_graph
#from ltr.models.dimp_state_machine.CameraMotionNet import CameraMotionNet
import matplotlib.pyplot as plt
import torchvision.transforms as T
from . import KalmanFilter
from scipy.ndimage import maximum_filter

plt.ion()


def caluatePeak(heat_map, peak_rang=0.3, min_peak=0.25):
    peak_map = maximum_filter(heat_map, size=5, mode='constant')
    is_peak = np.abs(peak_map - heat_map) < 1e-6
    peak_map = is_peak * peak_map
    max_val = heat_map.max()
    mask_top = (peak_map > max(min_peak, max_val - peak_rang))
    peak_map = mask_top * peak_map
    peak_num = mask_top.sum()

    return peak_num, peak_map


def compute_iou(box1, box2, scale=[1, 1]):

    lx1, ly1, w1, h1 = box1
    lx2, ly2, w2, h2 = box2

    x1_1, y1_1, x2_1, y2_1 = lx1 + w1 * (1 - scale[0]) * 0.5, ly1 + h1 * (1 - scale[1]) * 0.5, lx1 + w1 * (
            1 + scale[0]) * 0.5, ly1 + h1 * (1 + scale[1]) * 0.5
    x1_2, y1_2, x2_2, y2_2 = lx2, ly2, lx2 + w2, ly2 + h2

    inter_x1 = max(x1_1, x1_2)
    inter_y1 = max(y1_1, y1_2)
    inter_x2 = min(x2_1, x2_2)
    inter_y2 = min(y2_1, y2_2)

    inter_w = max(0, inter_x2 - inter_x1)
    inter_h = max(0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    area1 = w1 * h1
    area2 = w2 * h2

    union_area = area1 + area2 - inter_area

    if union_area == 0:
        return 0.0

    iou = inter_area / union_area
    return iou


def track(self, image, info: dict = None) -> dict:
    self.debug_info = {}

    self.frame_num += 1
    self.debug_info['frame_num'] = self.frame_num

    # Convert image
    im = numpy_to_torch(image)

    # ------- LOCALIZATION ------- #

    # Extract backbone features
    backbone_feat, sample_coords, im_patches = self.extract_backbone_features(im, self.get_centered_sample_pos(),
                                                                              self.target_scale * self.params.scale_factors,
                                                                              self.img_sample_sz)
    # Extract classification features
    test_x = self.get_classification_features(backbone_feat)

    # Location of sample
    sample_pos, sample_scales = self.get_sample_location(sample_coords)

    # Compute classification scores
    scores_raw = self.classify_target(test_x)

    # Localize the target
    translation_vec, scale_ind, s, flag = self.localize_target(scores_raw, sample_pos, sample_scales)
    new_pos = sample_pos[scale_ind, :] + translation_vec

    # Update position and scale
    if flag != 'not_found':
        if self.params.get('use_iou_net', True):
            update_scale_flag = self.params.get('update_scale_when_uncertain', True) or flag != 'uncertain'
            if self.params.get('use_classifier', True):
                self.update_state(new_pos)
            self.refine_target_box(backbone_feat, sample_pos[scale_ind, :], sample_scales[scale_ind], scale_ind,
                                   update_scale_flag)
        elif self.params.get('use_classifier', True):
            self.update_state(new_pos, sample_scales[scale_ind])

    # ------- UPDATE ------- #

    update_flag = flag not in ['not_found', 'uncertain']
    hard_negative = (flag == 'hard_negative')
    learning_rate = self.params.get('hard_negative_learning_rate', None) if hard_negative else None

    if update_flag and self.params.get('update_classifier', False):
        # Get train sample
        train_x = test_x[scale_ind:scale_ind + 1, ...]

        # Create target_box and label for spatial sample
        target_box = self.get_iounet_box(self.pos, self.target_sz, sample_pos[scale_ind, :], sample_scales[scale_ind])

        # Update the classifier model
        self.update_classifier(train_x, target_box, learning_rate, s[scale_ind, ...])

    # Set the pos of the tracker to iounet pos
    if self.params.get('use_iou_net', True) and flag != 'not_found' and hasattr(self, 'pos_iounet'):
        self.pos = self.pos_iounet.clone()

    score_map = s[scale_ind, ...]
    max_score = torch.max(score_map).item()

    # Visualize and set debug info
    self.search_area_box = torch.cat(
        (sample_coords[scale_ind, [1, 0]], sample_coords[scale_ind, [3, 2]] - sample_coords[scale_ind, [1, 0]] - 1))
    self.debug_info['flag' + self.id_str] = flag
    self.debug_info['max_score' + self.id_str] = max_score
    if self.visdom is not None:
        self.visdom.register(score_map, 'heatmap', 2, 'Score Map' + self.id_str)
        self.visdom.register(self.debug_info, 'info_dict', 1, 'Status')
    elif self.params.debug >= 2:
        show_tensor(score_map, 5, title='Max score = {:.2f}'.format(max_score))

    # Compute output bounding box
    new_state = torch.cat((self.pos[[1, 0]] - (self.target_sz[[1, 0]] - 1) / 2, self.target_sz[[1, 0]]))

    if self.params.get('output_not_found_box', False) and flag == 'not_found':
        output_state = [-1, -1, -1, -1]
    else:
        output_state = new_state.tolist()

    out = {'target_bbox': output_state,
           'fid': self.frame_num,
           'sample_coords': sample_coords,
           'scores': scores_raw.max().item(),
           'scores_raw': scores_raw.data,
           # 'xywh': [new_pos[0].item() / image.shape[0], new_pos[1].item() / image.shape[1],
           #          output_state[3] / image.shape[0], output_state[2] / image.shape[1]]}
           'xywh': [new_pos[1].item(), new_pos[0].item(), output_state[2], output_state[3]]}
    return out


def multiMatchTemplate(target, template, cen, scales=16, show=False, mask=None):
    if mask is not None:
        target = cv2.inpaint(target, mask, 3, cv2.INPAINT_TELEA)
        template = cv2.inpaint(template, mask, 3, cv2.INPAINT_TELEA)

    mh, mw = target.shape[:2]
    cen = cen.clip(0, mh)
    if isinstance(scales, int):
        scales = [scales]
    result = np.zeros((mh, mw))
    for idx, scale in enumerate(scales):
        try:
            lh, rh = max(min(3, cen[0]), cen[0] - scale), min(max(mh - 3, cen[0] + 1), cen[0] + scale + 1)
            lw, rw = max(min(3, cen[1]), cen[1] - scale), min(max(mw - 3, cen[1] + 1), cen[1] + scale + 1)

            padded_target = np.pad(target, ((cen[0] - lh, rh - cen[0] - 1), (cen[1] - lw, rw - cen[1] - 1), (0, 0)),
                                   mode='edge')

            res = cv2.matchTemplate(padded_target[:, :], template[lh:rh, lw:rw], cv2.TM_CCOEFF_NORMED)
            w = 1 if idx == 0 else (result > result.max() - 0.4)

            # if template[lh:rh, lw:rw].max() < 10:
            #     w = w * 0
            if template[lh:rh, lw:rw].reshape([-1, 3]).std(axis=0).max() < 3:
                w = w * 0
            result += res * w
        except Exception as e:
            print(e)
            import pdb
            pdb.set_trace()
        if show:
            plt.figure(idx + 1)
            plt.subplot(1, 3, 1)
            plt.imshow(res)
            plt.subplot(1, 3, 2)
            plt.imshow(result)
            plt.subplot(1, 3, 3)
            plt.imshow(template[lh:rh, lw:rw])

    return result


# def multiMatchTemplate(target, template, cen, scales=16, show=False):
#     mh, mw = target.shape[:2]
#     cen = cen.clip(0, mh)
#     if isinstance(scales, int):
#         scales = [scales]
#     result = np.zeros((mh, mw))
#     for idx, scale in enumerate(scales):
#         lh, rh = max(min(3, cen[0]), cen[0] - scale), min(max(mh - 3, cen[0] + 1), cen[0] + scale + 1)
#         lw, rw = max(min(3, cen[1]), cen[1] - scale), min(max(mw - 3, cen[1] + 1), cen[1] + scale + 1)
#         padded_target = np.pad(target, ((cen[0] - lh, rh - cen[0] - 1), (cen[1] - lw, rw - cen[1] - 1), (0, 0)),
#                                mode='edge')
#         res = cv2.matchTemplate(padded_target, template[lh:rh, lw:rw], cv2.TM_CCOEFF_NORMED)
#         try:
#             result += res * 0.4 ** idx
#         except Exception as e:
#             print(e)
#             import pdb
#             pdb.set_trace()
#         if show:
#             plt.figure(idx + 1)
#             plt.subplot(1, 3, 1)
#             plt.imshow(res)
#             plt.subplot(1, 3, 2)
#             plt.imshow(result)
#             plt.subplot(1, 3, 3)
#             plt.imshow(template[lh:rh, lw:rw])
#     return result


def offsetLocation(result, target, template, cen, show=False):
    # result = (result - result.min()) / (result.max() - result.min() + 1e-6)
    # peak_num, peak_map = caluatePeak(result, 0.3)
    #
    # print(peak_num)
    # peak_list = np.where(peak_map > 0)
    # i = 0
    # max_val, max_id = 0, (-1, -1)
    # x, y = cen
    # for u, v in zip(*peak_list):
    #     left_max = min(u, x)
    #     right_max = min(63 - u, 63 - x)
    #     up_max = min(v, y)
    #     down_max = min(63 - v, 63 - y)
    #     res = cv2.matchTemplate(target[v - up_max:v + down_max, u - left_max:u + right_max],
    #                             template[y - up_max:y + down_max, x - left_max:x + right_max],
    #                             cv2.TM_CCOEFF_NORMED)[0, 0]
    #     if max_val < res:
    #         max_val = res
    #         max_id = np.array([u, v])
    #
    #     if show:
    #         plt.figure(i)
    #         i += 1
    #         plt.subplot(1, 2, 1)
    #         plt.imshow(target[v - up_max:v + down_max, u - left_max:u + right_max])
    #         plt.subplot(1, 2, 2)
    #         plt.imshow(template[y - up_max:y + down_max, x - left_max:x + right_max])
    #
    # return max_id
    _, value, _, loc = cv2.minMaxLoc(result)
    return np.array(loc)[[1, 0]]


def llbox2ccbox(box, offset):
    lx, ly, w, h = box
    return [lx + w * 0.5 - offset[0], ly + h * 0.5 - offset[1], w, h]


class DimpJoin(BaseTracker):
    def __init__(self, params):
        super().__init__(params)
        self.params = params
        tracker_class = importlib.import_module('pytracking.tracker.dimp').get_tracker_class()
        params = importlib.import_module('pytracking.parameter.{}'.format(self.params.infrared_params)).parameters()
        self.tracker_inf = tracker_class(params)
        params = importlib.import_module('pytracking.parameter.{}'.format(self.params.visible_params)).parameters()
        self.tracker_vis = tracker_class(params)
        self.kf_inf = KalmanFilter.KalmanFilter()
        self.kf_vis = KalmanFilter.KalmanFilter()
        self.rls_inf_c_w = KalmanFilter.ScalarKalmanRegressor()
        self.rls_inf_c_h = KalmanFilter.ScalarKalmanRegressor()
        self.rls_vis_c_w = KalmanFilter.ScalarKalmanRegressor()
        self.rls_vis_c_h = KalmanFilter.ScalarKalmanRegressor()
        # self.rls_vis_c = KalmanFilter.RLS(1, delta=100)
        self.rls_inf_s = KalmanFilter.RLS(1, delta=100)
        self.rls_vis_s = KalmanFilter.RLS(1, delta=100)

        self.mask_inf = np.load("../ourExtra/mask/uav300/mask_inf.npy").astype(np.uint8) * 255
        self.mask_vis = np.load("../ourExtra/mask/uav300/mask_vis.npy").astype(np.uint8) * 255

        # self.CameraMotionNet = CameraMotionNet().to(0)
        # self.CameraMotionNet.load_state_dict(
        #     torch.load('../checkpoints/ltr/dimp_state_machine/checkpoint27.pth', map_location='cpu'))
        # self.CameraMotionNet.eval()
        # self.normalize = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    def processImage(self, img):
        img = cv2.resize(img, (256, 256))
        img = img.astype(np.float32) / 255.0
        img = torch.from_numpy(img).permute(2, 0, 1)
        return self.normalize(img)

    def initialize(self, image, info: dict) -> dict:

        out_a = self.tracker_inf.initialize(image[0], {'init_bbox': info['init_bbox'][0]})
        out_b = self.tracker_vis.initialize(image[1], {'init_bbox': info['init_bbox'][1]})
        self.sample_coords_a = info['init_bbox'][0]
        self.sample_coords_b = info['init_bbox'][1]
        self.pre_a = info['init_bbox'][0]
        self.pre_b = info['init_bbox'][1]
        self.o_p_a = info['init_bbox'][0]
        self.o_p_b = info['init_bbox'][1]
        self.scores_a = 1.0
        self.scores_b = 1.0
        self.scores_raw_a = None
        self.scores_raw_b = None
        self.offset_inf = None
        self.offset_vis = None
        self.inf_offset_res = None
        self.vis_offset_res = None
        self.last_image = {
            'inf': [image[0]],
            'vis': [image[1]],
        }

        self.kf_inf.init_state([(info['init_bbox'][0][0] + info['init_bbox'][0][2] * 0.5),
                                (info['init_bbox'][0][1] + info['init_bbox'][0][3] * 0.5),
                                info['init_bbox'][0][2],
                                info['init_bbox'][0][3]])
        self.kf_vis.init_state([(info['init_bbox'][1][0] + info['init_bbox'][1][2] * 0.5),
                                (info['init_bbox'][1][1] + info['init_bbox'][1][3] * 0.5),
                                info['init_bbox'][1][2],
                                info['init_bbox'][1][3]])

        self.trajectory = torch.zeros([1024, 6 + 6])
        self.trajectory[1][:4] = torch.tensor([(info['init_bbox'][0][0] + info['init_bbox'][0][2] * 0.5),
                                               (info['init_bbox'][0][1] + info['init_bbox'][0][3] * 0.5),
                                               info['init_bbox'][0][2],
                                               info['init_bbox'][0][3]])
        self.trajectory[1][6:10] = torch.tensor([(info['init_bbox'][1][0] + info['init_bbox'][1][2] * 0.5),
                                                 (info['init_bbox'][1][1] + info['init_bbox'][1][3] * 0.5),
                                                 info['init_bbox'][1][2],
                                                 info['init_bbox'][1][3]])
        self.confidence = torch.zeros([1024, 2])
        #
        # self.rls_inf_c.update(self.trajectory[1:2, 6:8].numpy(), self.trajectory[1:2, :2].numpy())
        # self.rls_inf_s.update(self.trajectory[1:2, 8:10].numpy(), self.trajectory[1:2, 2:4].numpy())
        # self.rls_vis_c.update(self.trajectory[1:2, :2].numpy(), self.trajectory[1:2, 6:8].numpy())
        # self.rls_vis_s.update(self.trajectory[1:2, 2:4].numpy(), self.trajectory[1:2, 8:10].numpy())

        # self.rls_inf_c.Theta = np.array(
        #     [
        #         [0.0909, 0] * 11 + [info['init_bbox'][0][0] - info['init_bbox'][1][0]],
        #         [0, 0.0909] * 11 + [info['init_bbox'][0][1] - info['init_bbox'][1][1]]
        #     ]
        # )
        #
        # self.rls_vis_c.Theta = np.array(
        #     [
        #         [0.0909, 0] * 11 + [info['init_bbox'][1][0] - info['init_bbox'][0][0]],
        #         [0, 0.0909] * 11 + [info['init_bbox'][1][1] - info['init_bbox'][0][1]]
        #     ]
        # )

        # self.rls_inf_c_w.update((info['init_bbox'][1][0] + info['init_bbox'][1][2] * 0.5),
        #                         (info['init_bbox'][0][0] + info['init_bbox'][0][2] * 0.5))
        # self.rls_inf_c_h.update((info['init_bbox'][1][1] + info['init_bbox'][1][3] * 0.5),
        #                         (info['init_bbox'][0][1] + info['init_bbox'][0][3] * 0.5))
        #
        # self.rls_vis_c_w.update((info['init_bbox'][0][0] + info['init_bbox'][0][2] * 0.5),
        #                         (info['init_bbox'][1][0] + info['init_bbox'][1][2] * 0.5))
        # self.rls_vis_c_h.update((info['init_bbox'][0][1] + info['init_bbox'][0][3] * 0.5),
        #                         (info['init_bbox'][1][1] + info['init_bbox'][1][3] * 0.5))
        self.rls_inf_s.Theta = np.array(
            [
                [0, 0] * 10 + [info['init_bbox'][0][2] / info['init_bbox'][1][2], 0] * 1 + [0],
                [0, 0] * 10 + [0, info['init_bbox'][0][3] / info['init_bbox'][1][3]] * 1 + [0],
            ]
        )

        self.rls_vis_s.Theta = np.array(
            [
                [0, 0] * 10 + [info['init_bbox'][1][2] / info['init_bbox'][0][2], 0] * 1 + [0],
                [0, 0] * 10 + [0, info['init_bbox'][1][3] / info['init_bbox'][0][3]] * 1 + [0],
            ]
        )

        self.data = []
        self.lostTarget = [1.5, 1.5]

        self.all_history = []
        return {'time': max(out_a['time'], out_b['time']), 'times': [out_a['time'], out_b['time']]}

    def track(self, image, info: dict = None) -> dict:
        ############################################
        # aligned image part
        ###########################################
        match_size = 64
        inf_center = np.clip(
            (self.tracker_inf.pos.numpy() / image[0].shape[:2] * (match_size - 1) + 0.5).astype(np.uint8), 0,
            match_size - 1)
        vis_center = np.clip(
            (self.tracker_vis.pos.numpy() / image[1].shape[:2] * (match_size - 1) + 0.5).astype(np.uint8), 0,
            match_size - 1)
        self.inf_res = multiMatchTemplate(cv2.resize(image[0], (match_size, match_size)),
                                          cv2.resize(self.last_image['inf'][0], (match_size, match_size)),
                                          inf_center, [4, 8, 16, 32],
                                          mask=cv2.resize(self.mask_inf, (match_size, match_size)))
        self.vis_res = multiMatchTemplate(cv2.resize(image[1], (match_size, match_size)),
                                          cv2.resize(self.last_image['vis'][0], (match_size, match_size)),
                                          vis_center, [4, 8, 16, 32],
                                          mask=cv2.resize(self.mask_vis, (match_size, match_size)))
        inf_max_loc = offsetLocation(self.inf_res, cv2.resize(image[0], (match_size, match_size)),
                                     cv2.resize(self.last_image['inf'][0], (match_size, match_size)), inf_center)
        vis_max_loc = offsetLocation(self.vis_res, cv2.resize(image[1], (match_size, match_size)),
                                     cv2.resize(self.last_image['vis'][0], (match_size, match_size)), vis_center)

        d_inf_max_loc = inf_max_loc - inf_center
        d_vis_max_loc = vis_max_loc - vis_center

        aux_center = np.array([match_size // 2, match_size // 2])
        aux_inf_res = multiMatchTemplate(cv2.resize(image[0], (match_size, match_size)),
                                         cv2.resize(self.last_image['inf'][0], (match_size, match_size)),
                                         aux_center, [4, 8, 16, 32],
                                         mask=cv2.resize(self.mask_inf, (match_size, match_size)))
        aux_vis_res = multiMatchTemplate(cv2.resize(image[1], (match_size, match_size)),
                                         cv2.resize(self.last_image['vis'][0], (match_size, match_size)),
                                         aux_center, [4, 8, 16, 32],
                                         mask=cv2.resize(self.mask_vis, (match_size, match_size)))

        _, value_aux_inf, _, _loc_aux_inf = cv2.minMaxLoc(aux_inf_res)
        _, value_aux_vis, _, _loc_aux_vis = cv2.minMaxLoc(aux_vis_res)

        loc_aux_inf = inf_max_loc - inf_center + aux_center
        loc_aux_vis = vis_max_loc - vis_center + aux_center

        if (loc_aux_inf.min() < 0 or loc_aux_inf.max() > match_size - 1 or
                aux_inf_res[
                max(0, loc_aux_inf[0] - 1):loc_aux_inf[0] + 2,
                max(0, loc_aux_inf[1] - 1):loc_aux_inf[1] + 2].max() < value_aux_inf * 0.8):
            print("fixed")
            d_inf_max_loc = np.array(_loc_aux_inf)[[1, 0]] - aux_center
        if (loc_aux_vis.min() < 0 or loc_aux_vis.max() > match_size - 1 or
                aux_vis_res[
                max(0, loc_aux_vis[0] - 1):loc_aux_vis[0] + 2,
                max(0, loc_aux_vis[1] - 1):loc_aux_vis[1] + 2].max() < value_aux_vis * 0.8):
            print("fixed")
            d_vis_max_loc = np.array(_loc_aux_vis)[[1, 0]] - aux_center

        self.offset_inf = d_inf_max_loc / match_size * image[0].shape[:2]
        self.offset_vis = d_vis_max_loc / match_size * image[1].shape[:2]
        self.tracker_inf.pos += torch.tensor(self.offset_inf)
        self.tracker_vis.pos += torch.tensor(self.offset_vis)

        #################################
        # track part by image
        #################################
        output_state_a = track(self.tracker_inf, image[0], info)  # self.tracker_inf.track(image[0], info)
        output_state_b = track(self.tracker_vis, image[1], info)  # self.tracker_vis.track(image[1], info)

        f_id = self.tracker_inf.frame_num
        # if (np.abs(self.offset_inf).max() > 0 and self.confidence[f_id - 1, 0] > 0 and
        #         (self.scores_a < 0.25 or )):
        #     self.tracker_inf.pos -= torch.tensor(self.offset_inf)
        #     self.offset_inf = self.offset_inf * 0
        # if np.abs(self.offset_vis).max() > 0 and self.scores_b < 0.25 and self.confidence[f_id - 1, 1] > 0:
        #     self.tracker_vis.pos -= torch.tensor(self.offset_vis)
        #     self.offset_vis = self.offset_vis * 0

        #################################
        # recode for vis
        #################################
        sa_a, sa_b = output_state_a['sample_coords'][0], output_state_b['sample_coords'][0]
        self.inf_offset_res = self.inf_res
        self.vis_offset_res = self.vis_res
        self.sample_coords_a = (sa_a[[1, 0]].tolist() + (sa_a[[3, 2]] - sa_a[[1, 0]]).tolist())
        self.sample_coords_b = (sa_b[[1, 0]].tolist() + (sa_b[[3, 2]] - sa_b[[1, 0]]).tolist())
        self.detection_a = output_state_a['target_bbox']
        self.detection_b = output_state_b['target_bbox']
        self.scores_a = output_state_a['scores']
        self.scores_b = output_state_b['scores']
        self.scores_raw_a = output_state_a['scores_raw']
        self.scores_raw_b = output_state_b['scores_raw']
        a_xywh, b_xywh = list(output_state_a['xywh']), list(output_state_b['xywh'])
        self.trajectory[f_id] = torch.tensor(a_xywh + list(self.offset_inf) + b_xywh + list(self.offset_vis))
        self.trajectory[f_id][[4, 5]] += self.trajectory[f_id - 1][[4, 5]]
        self.trajectory[f_id][[10, 11]] += self.trajectory[f_id - 1][[10, 11]]

        #################################
        # print info
        # there have three position:
        # detection result
        # predict result from history
        # predict result from another history
        #################################

        tt = self.trajectory[f_id]
        history = self.trajectory[max(1, f_id - 11):f_id]
        pos_tra, box_tra = (history[:, 6:8] - history[:, [11, 10]]).numpy(), history[:, 8:10].numpy()
        # a_c, a_s = [rls.predict(data) for rls, data in zip([self.rls_inf_c, self.rls_inf_s], [pos_tra, box_tra])]
        a_c = np.array([self.rls_inf_c_w.predict(pos_tra[-1][0:1]), self.rls_inf_c_h.predict(pos_tra[-1][1:2])])
        a_s = self.rls_inf_s.predict(box_tra)

        pos_tra, box_tra = (history[:, 0:2] - history[:, [5, 4]]).numpy(), history[:, 2:4].numpy()
        # b_c, b_s = [rls.predict(data) for rls, data in zip([self.rls_vis_c, self.rls_vis_s], [pos_tra, box_tra])]
        b_c = np.array([self.rls_vis_c_w.predict(pos_tra[-1][0:1]), self.rls_vis_c_h.predict(pos_tra[-1][1:2])])
        b_s = self.rls_inf_s.predict(box_tra)

        self.o_p_a = [a_c[0] - a_s[0] * 0.5 + tt[5].item(), a_c[1] - a_s[1] * 0.5 + tt[4].item()] + a_s
        self.o_p_b = [b_c[0] - b_s[0] * 0.5 + tt[11].item(), b_c[1] - b_s[1] * 0.5 + tt[10].item()] + b_s
        #
        pre_a, pre_b = self.kf_inf.predict()[:4, 0].tolist(), self.kf_vis.predict()[:4, 0].tolist()  # cx,cy,w,h
        pre_a_box, pre_b_box = [[x - w * 0.5, y - h * 0.5, w, h] for (x, y, w, h) in [pre_a, pre_b]]  # lx,ly,w,h
        self.pre_a = [pre_a_box[0] + tt[5].item(), pre_a_box[1] + tt[4].item(), pre_a_box[2], pre_a_box[3]]
        self.pre_b = [pre_b_box[0] + tt[11].item(), pre_b_box[1] + tt[10].item(), pre_b_box[2], pre_b_box[3]]
        #
        # # # print(["%.2f" % self.kf_inf.P[u, u] for u in range(8)])
        print('o', list(map(int, self.detection_a)), list(map(int, self.detection_b)))
        print('m', list(map(int, self.o_p_a)), list(map(int, self.o_p_b)))
        print('p', list(map(int, self.pre_a)), list(map(int, self.pre_b)))
        # # # print(history[:, [0, 1, 6, 7]])
        print('*' * 20, self.tracker_inf.frame_num)

        # print(["%.2f,%.2f" % (u, v) for u, v in self.rls_vis_c.Theta.T])

        #################################
        # fixed result
        # detection result: detection_a, detection_b
        # predict result from history:  pre_a,pre_b
        # predict result from another history:  o_p_a,o_p_b
        #################################
        final_res = [[0, 0, 0, 0], [0, 0, 0, 0]]
        confid = self.confidence[min(f_id - 11, 0):f_id].mean(dim=0).tolist()
        self.lostTarget = [self.lostTarget[0] + 0.1, self.lostTarget[1] + 0.1]

        for (m_id, m, mode) in [(0, 'a', 'inf'), (1, 'b', 'vis')]:
            # final_res[m_id] = np.array(getattr(self, 'detection_' + m)).tolist()

            offset = self.trajectory[f_id][[m_id * 6 + 5, m_id * 6 + 4]].tolist()
            res, res_w = np.zeros([4]), 0
            target_ocr = False
            if (compute_iou(getattr(self, 'detection_' + m), getattr(self, 'pre_' + m),
                            [self.lostTarget[m_id]] * 2) > 0.0 and getattr(self, 'scores_' + m) > 0.25 and
                    caluatePeak(getattr(self, 'scores_raw_' + m)[0, 0].cpu().numpy())[0] == 1):
                res, res_w = res + np.array(getattr(self, 'detection_' + m)) * 2, res_w + 2
                self.confidence[f_id, m_id] = 1
                if self.lostTarget[m_id] >= 2.0:
                    target_ocr = True
                self.lostTarget[m_id] = 1.5
            #
            elif compute_iou(getattr(self, 'o_p_' + m), getattr(self, 'pre_' + m),
                             [self.lostTarget[m_id]] * 2) > 0.0 and confid[m_id ^ 1] > 0.7:
                res, res_w = res + np.array(getattr(self, 'o_p_' + m)), res_w + 1

            if res_w > 0:
                final_res[m_id] = (res / res_w).tolist()
                getattr(self, 'tracker_' + mode).pos = torch.tensor(llbox2ccbox(final_res[m_id], [0, 0]))[[1, 0]]

                getattr(self, 'kf_' + mode).update(np.array(llbox2ccbox(final_res[m_id], offset)), target_ocr)
                # print(m, np.array(llbox2ccbox(final_res[m_id], offset)))

                if res_w > 1.8:
                    mm_id = (m_id ^ 1) * 6
                    pos_tra = (history[:, 0 + mm_id:2 + mm_id] - history[:, [mm_id + 5, mm_id + 4]]).numpy()
                    box_tra = history[:, mm_id + 2:mm_id + 4].numpy()
                    target_pos = np.array(llbox2ccbox(final_res[m_id], offset))[:2]
                    print(mode, getattr(self, 'rls_' + mode + '_c_w').theta,
                          "%.2f" % pos_tra[-1, 0],
                          "%.2f" % target_pos[0], end=' ')

                    getattr(self, 'rls_' + mode + '_c_w').update(pos_tra[-1, 0:1], target_pos[0:1])
                    print(getattr(self, 'rls_' + mode + '_c_w').theta)
                    getattr(self, 'rls_' + mode + '_c_h').update(pos_tra[-1, 1:2], target_pos[1:2])
                    if mode == 'inf' and f_id > 200:
                        p = 10 * 10

            else:
                if self.lostTarget[m_id] >= 2.5:
                    cx, cy, w, h = getattr(self, 'o_p_' + m)
                    final_res[m_id] = [cx - w * 0.75, cy - h * 0.75, w * 1.5, h * 1.5]
                    getattr(self, 'kf_' + mode).update(np.array(llbox2ccbox(final_res[m_id], offset)), True)
                else:
                    final_res[m_id] = getattr(self, 'pre_' + m)
                # final_res[m_id] = getattr(self, 'pre_' + m)
                getattr(self, 'tracker_' + mode).pos = torch.tensor(llbox2ccbox(final_res[m_id], [0, 0]))[[1, 0]]
            getattr(self, 'kf_' + mode).x[4:, 0] *= 0.9

        # self.visdom_draw_tracking(image, [[output_state_a['target_bbox'], output_state_b['target_bbox']]])
        self.trajectory[self.tracker_inf.frame_num][0] = self.tracker_inf.pos[1]
        self.trajectory[self.tracker_inf.frame_num][1] = self.tracker_inf.pos[0]
        self.trajectory[self.tracker_inf.frame_num][2] = final_res[0][2]
        self.trajectory[self.tracker_inf.frame_num][3] = final_res[0][3]
        self.trajectory[self.tracker_inf.frame_num][6] = self.tracker_vis.pos[1]
        self.trajectory[self.tracker_inf.frame_num][7] = self.tracker_vis.pos[0]
        self.trajectory[self.tracker_inf.frame_num][8] = final_res[1][2]
        self.trajectory[self.tracker_inf.frame_num][9] = final_res[1][3]

        self.all_history.append({
            'scores_raw': [self.scores_raw_a, self.scores_raw_b],
            'heatmap': [self.inf_offset_res, self.vis_offset_res],
            'box': [[final_res[0], self.sample_coords_a, self.pre_a, self.o_p_a],
                    [final_res[1], self.sample_coords_b, self.pre_b, self.o_p_b]],
            'image': image,
            'x': [self.kf_inf.x, self.kf_vis.x],
            'posadd': [torch.tensor(self.offset_inf), torch.tensor(self.offset_vis)],
            'loc': [[self.inf_res, cv2.resize(image[0], (match_size, match_size)),
                     cv2.resize(self.last_image['inf'][0], (match_size, match_size)), inf_center],
                    [self.vis_res, cv2.resize(image[1], (match_size, match_size)),
                     cv2.resize(self.last_image['vis'][0], (match_size, match_size)), vis_center]],
            'offset': [self.offset_inf, self.offset_vis]
        })
        self.last_image = {'inf': [image[0]], 'vis': [image[1]]}
        self.visdom_draw_tracking(image, [[output_state_a['target_bbox'], output_state_b['target_bbox']]])
        # if f_id >= 21:
        #     traject = self.trajectory[max(1, 1):f_id + 1]
        #     print(traject[:, [0, 1, 6, 7]] - traject[:, [5, 4, 11, 10]])
        #     self.visdom_draw_tracking(image, [[output_state_a['target_bbox'], output_state_b['target_bbox']]])
        #     print('pause')

        return {'target_bbox': final_res}

    def show_history(self, t):
        self.visdom.register((self.all_history[t]['image'][0], *self.all_history[t]['box'][0]), 'Tracking', 1,
                             'Tracking_inf')
        self.visdom.register((self.all_history[t]['image'][1], *self.all_history[t]['box'][1]), 'Tracking', 1,
                             'Tracking_vis')

        scores_raw_a = self.all_history[t]['scores_raw'][0]
        scores_raw_b = self.all_history[t]['scores_raw'][1]
        inf_offset_res = self.all_history[t]['heatmap'][0]
        vis_offset_res = self.all_history[t]['heatmap'][1]
        h_d_a, _ = caluatePeak(scores_raw_a[0, 0].cpu().numpy())
        h_d_b, _ = caluatePeak(scores_raw_b[0, 0].cpu().numpy())
        o_d_a, _ = caluatePeak(inf_offset_res)
        o_d_b, _ = caluatePeak(vis_offset_res)
        self.visdom.register("score:%.3f,%.3f \t "
                             "heat peaknum:%d %d \t "
                             "off peaknum:%d %d" % (self.scores_a, self.scores_b, h_d_a, h_d_b, o_d_a, o_d_b),
                             'text',
                             1, 'information')

        self.visdom.register(scores_raw_a * (scores_raw_a > 0), 'heatmap', 1, 'heatmap_a')
        self.visdom.register(scores_raw_b * (scores_raw_b > 0), 'heatmap', 1, 'heatmap_b')

        self.visdom.register(torch.tensor(inf_offset_res[None, None]), 'heatmap', 1, 'heatmap_offset_a')
        self.visdom.register(torch.tensor(vis_offset_res[None, None]), 'heatmap', 1, 'heatmap_offset_b')

    def visdom_draw_tracking(self, image, box, segmentation=None):
        if box is None:
            box = []
        elif isinstance(box, OrderedDict):
            box = [v for k, v in box.items()]
        elif isinstance(box, list):
            box = box
        else:
            box = (box,)
        if segmentation is None:
            self.visdom.register((image[0], box[0][0], self.sample_coords_a, self.pre_a, self.o_p_a), 'Tracking', 1,
                                 'Tracking_inf')
            self.visdom.register((image[1], box[0][1], self.sample_coords_b, self.pre_b, self.o_p_b), 'Tracking', 1,
                                 'Tracking_vis')
        else:
            self.visdom.register((image, *box, segmentation), 'Tracking', 1, 'Tracking')

        if self.scores_raw_a is not None:
            h_d_a, _ = caluatePeak(self.scores_raw_a[0, 0].cpu().numpy())
            h_d_b, _ = caluatePeak(self.scores_raw_b[0, 0].cpu().numpy())
            o_d_a, _ = caluatePeak(self.inf_offset_res)
            o_d_b, _ = caluatePeak(self.vis_offset_res)
            self.visdom.register("score:%.3f,%.3f \t "
                                 "heat peaknum:%d %d \t "
                                 "off peaknum:%d %d" % (self.scores_a, self.scores_b, h_d_a, h_d_b, o_d_a, o_d_b),
                                 'text',
                                 1, 'information')
            if False:
                self.visdom.register(self.scores_raw_a * (self.scores_raw_a > 0), 'heatmap', 1, 'heatmap_a')
                self.visdom.register(self.scores_raw_b * (self.scores_raw_b > 0), 'heatmap', 1, 'heatmap_b')

                self.visdom.register(torch.tensor(self.inf_offset_res[None, None]), 'heatmap', 1, 'heatmap_offset_a')
                self.visdom.register(torch.tensor(self.vis_offset_res[None, None]), 'heatmap', 1, 'heatmap_offset_b')

        # if self.scores_a < 0.4 and self.scores_b < 0.4:
        # import pdb
        # pdb.set_trace()
        if self.tracker_inf.frame_num % 16 == 0:
            f_id = self.tracker_inf.frame_num
            plt.clf()
            plt.subplot(2, 2, 1)
            # plt.clf()
            # plt.plot(self.trajectory[:, 0] + self.trajectory[:, 2])
            plt.plot(self.trajectory[:f_id, 0] - self.trajectory[:f_id, 5])
            plt.plot(self.trajectory[:f_id, 0])
            plt.plot(self.trajectory[:f_id, 5])
            # plt.plot(self.trajectory_fixed[:, 0] + 0.1)

            plt.subplot(2, 2, 2)
            # plt.clf()
            plt.plot(self.trajectory[:f_id, 1] - self.trajectory[:f_id, 4])
            plt.plot(self.trajectory[:f_id, 1])
            plt.plot(self.trajectory[:f_id, 4])
            # plt.plot(self.trajectory_fixed[:, 1] + 0.1)
            plt.subplot(2, 2, 3)
            # plt.clf()
            # plt.plot(self.trajectory[:, 0] + self.trajectory[:, 2])
            plt.plot(self.trajectory[:f_id, 6] - self.trajectory[:f_id, 11])
            plt.plot(self.trajectory[:f_id, 6])
            plt.plot(self.trajectory[:f_id, 11])
            # plt.plot(self.trajectory_fixed[:, 6] + 0.1)

            plt.subplot(2, 2, 4)
            # plt.clf()
            plt.plot(self.trajectory[:f_id, 7] - self.trajectory[:f_id, 10])
            plt.plot(self.trajectory[:f_id, 7])
            plt.plot(self.trajectory[:f_id, 10])
            # plt.plot(self.trajectory_fixed[:, 7] + 0.1)

            plt.pause(0.1)

            # for idx in range(f_id - 1):
            #     print("".join(["%7s" % ("%.2f" % self.all_history[idx]['x'][0][u]) for u in [0, 1, 4, 5]]) + "|"
            #           + "".join(["%7s" % ("%.2f" % self.all_history[idx]['x'][1][u]) for u in [0, 1, 4, 5]]))

        # if self.offset_vis is not None and abs(self.offset_vis[0]) > 0.05:
        #     import pdb
        #     pdb.set_trace()
        # import pdb
        # pdb.set_trace()
        # if 'line' not in self.visdom.registered_blocks.keys():
        #     from pytracking.utils.visdom import VisLinePlot
        #     self.visdom.visdom.properties(self.visdom.blocks_list, opts={'title': 'Block List'}, win='block_list')
        #     self.visdom.registered_blocks['line'] = VisLinePlot(self.visdom, True, 'line')
        #
        # self.visdom.registered_blocks['line'].save_data((self.trajectory[:, 0], None))
        # self.visdom.registered_blocks['line'].draw_data()
        # try:
        #     pass
        # except Exception as e:
        #     print(e)
        #     self.visdom.register((self.trajectory[:, 0], None), 'lineplot', 1, 'line')
