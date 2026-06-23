# -*- coding: utf-8 -*-
# Based on DIMP execution template, adapted for multi-modal tracking.
import sys

import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import torch
torch.set_num_threads(1)

# Locate the project root by matching the pyTrackBridge directory in the path.
_file_path = os.path.abspath(__file__)
_idx = _file_path.rfind('pyTrackBridge')
if _idx != -1:
    _project_root = _file_path[:_idx].rstrip(os.sep)
    sys.path.insert(0, _project_root)
print("project_path:",_project_root)

target_script = os.path.join('..', 'pytracking', 'run_experiment.py')
script_args = ['myexperiments', 'uav_test', '--debug=0']

call_tree_text = """
pytracking.experiments.myexperiments @ uav_test -> uav_test  # Override dataset and model parameters for UAV test.
pytracking.evaluation.data @ Sequence._construct_init_data -> construct_init_data  # Patch init data: one image → two images.
pytracking.evaluation.tracker @ Tracker._read_image -> read_image # Patch image loading: single image → dual images.
pytracking.evaluation.tracker @ Tracker.create_tracker -> create_tracker # Replace tracker factory to support dual-image input.
pytracking.evaluation.tracker @ Tracker._track_sequence -> track_sequence # Remove two lines of OXuVA output handling (unused in multi-modal mode; causes errors otherwise).
pytracking.evaluation.running @ run_sequence -> run_sequence # Patch save / existence check to store results for both modalities.
pytracking.evaluation.running @ run_dataset -> run_dataset # Patch multi-process section for parallel sequence execution (slightly slower than the original).
"""

import cv2 as cv
# import ourExtra.lib.dataset.anti_uav300dataset as antiUav300
from pytracking.tracker.base import BaseTracker
from _collections import OrderedDict
import time, tempfile, subprocess, importlib, pickle
import numpy as np
from itertools import product
from concurrent.futures import ThreadPoolExecutor
from http.server import HTTPServer, BaseHTTPRequestHandler
import threading

def get_tracker_list():
    raise NotImplementedError(
        "You must define get_tracker_list() in your config file."
    )


def get_sequence_list():
    raise NotImplementedError(
        "You must define SequenceList() in your config file."
    )


def uav_test():
    import pytracking.evaluation.tracker as tracker
    from pytracking.evaluation.environment import env_settings
    env = env_settings()
    trackers = []
    for tracker_info in globals()['get_tracker_list']():
        run_ids = tracker_info['run_ids'] if tracker_info.__contains__('run_ids') else None
        if run_ids is None or isinstance(run_ids, int):
            run_ids = [run_ids]
        for run_id in run_ids:
            tracker_item = tracker.Tracker('dimp', 'dimp50', run_id, None)
            if run_id is not None:
                tracker_item.results_dir = '{}/{}_{:03d}'.format(env.results_path, tracker_info['name'], run_id)
            else:
                tracker_item.results_dir = '{}/{}'.format(env.results_path, tracker_info['name'])
            tracker_item.ext_parameter = tracker_info
            trackers.append(tracker_item)

    SequenceList = globals()['get_sequence_list']()
    

    return trackers, SequenceList



def construct_init_data(self, init_data):
    init_data = {0: dict()}
    init_data[0]['object_ids'] = self.object_ids
    init_data[0]['bbox'] = [list(self.ground_truth_rect[0][0, :]), list(self.ground_truth_rect[1][0, :])]
    return init_data


def read_image(self, image_file):
    im_inf = cv.imread(image_file[0])
    im_vis = cv.imread(image_file[1])
    return [cv.cvtColor(im_inf, cv.COLOR_BGR2RGB), cv.cvtColor(im_vis, cv.COLOR_BGR2RGB)]


class MultiTracker(BaseTracker):
    def __init__(self, tracker_inf, tracker_vis):
        super().__init__({})
        self.tracker_inf = tracker_inf
        self.tracker_vis = tracker_vis

    def initialize(self, image, info: dict) -> dict:
        out_a = self.tracker_inf.initialize(image[0], {'init_bbox': info['init_bbox'][0]})
        out_b = self.tracker_vis.initialize(image[1], {'init_bbox': info['init_bbox'][1]})
        if out_a is None or out_b is None:
            return {'time': -1, 'times': -1}
        return {'time': max(out_a['time'], out_b['time']), 'times': [out_a['time'], out_b['time']]}

    def track(self, image, info: dict = None) -> dict:
        output_state_a = self.tracker_inf.track(image[0], info)
        output_state_b = self.tracker_vis.track(image[1], info)
        return {'target_bbox': [output_state_a['target_bbox'], output_state_b['target_bbox']]} 

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
            self.visdom.register((image[0], (box[0][0])), 'Tracking', 1, 'Tracking_inf')
            self.visdom.register((image[1], (box[0][1])), 'Tracking', 1, 'Tracking_vis')
        else:
            self.visdom.register((image, *box, segmentation), 'Tracking', 1, 'Tracking')


def create_tracker(self, params: dict):
    tracker_setting = self.ext_parameter.get('tracker_setting', None)

    if tracker_setting is None:
        raise ValueError("ext_parameter must contain 'tracker_setting' field")

    tracker = None

    if tracker_setting['type'] == 'combin':
        inf_path = tracker_setting['infrared_params']
        tracker_module = importlib.import_module(f'pytracking.tracker.{inf_path.split(".")[0]}')
        param_module = importlib.import_module(f'pytracking.parameter.{inf_path}')
        inf_tracker = tracker_module.get_tracker_class()(param_module.parameters())

        vis_path = tracker_setting['visible_params']
        tracker_module = importlib.import_module(f'pytracking.tracker.{vis_path.split(".")[0]}')
        param_module = importlib.import_module(f'pytracking.parameter.{vis_path}')
        vis_tracker = tracker_module.get_tracker_class()(param_module.parameters())

        tracker = MultiTracker(inf_tracker, vis_tracker)
        tracker.params = inf_tracker.params

    elif tracker_setting['type'] == 'params':

        params_path = tracker_setting['params']
        tracker_module = importlib.import_module(f'pyTrackBridge.tracker.{params_path.split(".")[0]}')
        param_module = importlib.import_module(f'pyTrackBridge.parameter.{params_path}')
        tracker = tracker_module.get_tracker_class()(param_module.parameters())

    elif tracker_setting['type'] == 'function':
        func = tracker_setting['function']
        if not callable(func):
            raise TypeError("'function' must be a callable object")
        tracker = func()

    else:
        raise NotImplementedError(f"Unsupported tracker type: {tracker_setting['type']}")

    if tracker is None:
        raise RuntimeError("Tracker was not created successfully")

    tracker.visdom = self.visdom
    return tracker


def track_sequence(self, tracker, seq, init_info):
    output = {'target_bbox': [],
              'time': [],
              'segmentation': [],
              'object_presence_score': []}

    def _store_outputs(tracker_out: dict, defaults=None):
        defaults = {} if defaults is None else defaults
        for key in output.keys():
            val = tracker_out.get(key, defaults.get(key, None))
            if key in tracker_out or val is not None:
                output[key].append(val)

    # Initialize
    image = self._read_image(seq.frames[0])

    if tracker.params.visualization and self.visdom is None:
        self.visualize(image, init_info.get('init_bbox'))

    start_time = time.time()
    out = tracker.initialize(image, init_info)
    if out is None:
        out = {}

    prev_output = OrderedDict(out)

    init_default = {'target_bbox': init_info.get('init_bbox'),
                    'clf_target_bbox': init_info.get('init_bbox'),
                    'time': time.time() - start_time,
                    'segmentation': init_info.get('init_mask'),
                    'object_presence_score': 1.}

    _store_outputs(out, init_default)

    segmentation = out['segmentation'] if 'segmentation' in out else None
    bboxes = [init_default['target_bbox']]
    if 'clf_target_bbox' in out:
        bboxes.append(out['clf_target_bbox'])
    if 'clf_search_area' in out:
        bboxes.append(out['clf_search_area'])
    if 'segm_search_area' in out:
        bboxes.append(out['segm_search_area'])

    if self.visdom is not None:
        tracker.visdom_draw_tracking(image, bboxes, segmentation)
    elif tracker.params.visualization:
        self.visualize(image, bboxes, segmentation)

    for frame_num, frame_path in enumerate(seq.frames[1:], start=1):
        print("seq:{}, frame:{}/{}".format(seq.name, frame_num + 1, len(seq.frames)), end='\r')
        while True:
            if not self.pause_mode:
                break
            elif self.step:
                self.step = False
                break
            else:
                time.sleep(0.1)
        image = self._read_image(frame_path)

        start_time = time.time()

        info = seq.frame_info(frame_num)
        info['previous_output'] = prev_output

        out = tracker.track(image, info)
        prev_output = OrderedDict(out)
        _store_outputs(out, {'time': time.time() - start_time})

        segmentation = out['segmentation'] if 'segmentation' in out else None

        bboxes = [out['target_bbox']]
        if 'clf_target_bbox' in out:
            bboxes.append(out['clf_target_bbox'])
        if 'clf_search_area' in out:
            bboxes.append(out['clf_search_area'])
        if 'segm_search_area' in out:
            bboxes.append(out['segm_search_area'])

        if self.visdom is not None:
            tracker.visdom_draw_tracking(image, bboxes, segmentation)
        elif tracker.params.visualization:
            self.visualize(image, bboxes, segmentation)

    for key in ['target_bbox', 'segmentation']:
        if key in output and len(output[key]) <= 1:
            output.pop(key)
    return output


def save_tracker_output(seq, tracker, output: dict):
    """Saves the output of the tracker."""
    base_results_path = os.path.join(tracker.results_dir, seq.name)
    if not os.path.exists(os.path.join(tracker.results_dir, 'visible')):
        os.makedirs(os.path.join(tracker.results_dir, 'visible'))
    if not os.path.exists(os.path.join(tracker.results_dir, 'infrared')):
        os.makedirs(os.path.join(tracker.results_dir, 'infrared'))

    def save_bb(file, data):
        tracked_bb = np.array(data).astype(int)
        np.savetxt(file, tracked_bb, delimiter='\t', fmt='%d')

    def save_scores(file, data):
        scores = np.array(data).astype(float)
        np.savetxt(file, scores, delimiter='\t', fmt='%f')

    def save_time(file, data):
        exec_times = np.array(data).astype(float)
        np.savetxt(file, exec_times, delimiter='\t', fmt='%f')

    def _convert_dict(input_dict):
        data_dict = {}
        for elem in input_dict:
            for k, v in elem.items():
                if k in data_dict.keys():
                    data_dict[k].append(v)
                else:
                    data_dict[k] = [v, ]
        return data_dict

    for key, data in output.items():
        # If data is empty
        if not data:
            continue
        if key == 'target_bbox':
            base_results_path_inf = os.path.join(tracker.results_dir, 'infrared', seq.name)
            save_bb('{}.txt'.format(base_results_path_inf), [u[0] for u in data])
            base_results_path_vis = os.path.join(tracker.results_dir, 'visible', seq.name)
            save_bb('{}.txt'.format(base_results_path_vis), [u[1] for u in data])

        elif key == 'object_presence_score':
            if isinstance(data[0], (dict, OrderedDict)):
                data_dict = _convert_dict(data)

                for obj_id, d in data_dict.items():
                    scores_file = '{}_{}_object_presence_scores.txt'.format(base_results_path, obj_id)
                    save_scores(scores_file, d)
            else:
                scores_file = '{}_object_presence_scores.txt'.format(base_results_path)
                save_scores(scores_file, data)

        elif key == 'time':
            if isinstance(data[0], dict):
                data_dict = _convert_dict(data)
                for obj_id, d in data_dict.items():
                    timings_file = '{}_{}_time.txt'.format(base_results_path, obj_id)
                    save_time(timings_file, d)
            else:
                timings_file = '{}_time.txt'.format(base_results_path)
                save_time(timings_file, data)


def run_sequence(seq, tracker, debug=False, visdom_info=None):
    import pytracking.evaluation.running as running

    def _results_exist():
        if seq.dataset == 'oxuva':
            vid_id, obj_id = seq.name.split('_')[:2]
            pred_file = os.path.join(tracker.results_dir, '{}_{}.csv'.format(vid_id, obj_id))
            return os.path.isfile(pred_file)
        elif seq.object_ids is None:
            bbox_file = '{}/infrared/{}.txt'.format(tracker.results_dir, seq.name)
            return os.path.isfile(bbox_file)
        else:
            bbox_files = ['{}/infrared/{}_{}.txt'.format(tracker.results_dir, seq.name, obj_id) for obj_id in
                          seq.object_ids]
            missing = [not os.path.isfile(f) for f in bbox_files]
            return sum(missing) == 0

    visdom_info = {} if visdom_info is None else visdom_info

    if _results_exist() and not debug:
        print('FPS: {}'.format(-1))
        return
    
    print('Tracker: {} ({})(run id:{}) ,  Sequence: {}'.format(tracker.ext_parameter['name'],
                                                    tracker.ext_parameter['tracker_setting']['name'],
                                                    tracker.run_id, seq.name))

    if debug:
        output = tracker.run_sequence(seq, debug=debug, visdom_info=visdom_info)
    else:
        try:
            output = tracker.run_sequence(seq, debug=debug, visdom_info=visdom_info)
        except Exception as e:
            print(e)
            return

    sys.stdout.flush()

    if isinstance(output['time'][0], (dict, OrderedDict)):
        exec_time = sum([sum(times.values()) for times in output['time']])
        num_frames = len(output['time'])
    else:
        exec_time = sum(output['time'])
        num_frames = len(output['time'])

    print('FPS: {}'.format(num_frames / exec_time))

    if not debug:
        save_tracker_output(seq, tracker, output)


def run_dataset(dataset, trackers, debug=False, threads=0, visdom_info=None):
    print('Evaluating {:4d} trackers on {:5d} sequences'.format(len(trackers), len(dataset)))
    visdom_info = {} if visdom_info is None else visdom_info
    if threads == 0:
        mode = 'sequential'
    else:
        mode = 'parallel'
    if mode == 'sequential':
        for seq in dataset:
            for tracker_info in trackers:
                run_sequence(seq, tracker_info, debug=debug, visdom_info=visdom_info)
    elif mode == 'parallel':
        
        def run_one(pkl):
            arg_dict = {}
            arg_dict['run_module'] = 'pytracking.evaluation.running'
            arg_dict['run_function'] = 'run_sequence'
            arg_dict['run_argv'] = pkl
            arg_dict['config'] = __file__
            code = "import sys\n"
            code += f"sys.path.insert(0, {_project_root!r})\n"
            code += "import pyTrackBridge.rig as rig\n"
            code += "rig.main({},True)".format(repr(arg_dict))
            subprocess.run([sys.executable, "-c", code])


        param_cache = {}
        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args): pass
            def do_GET(self):
                data = param_cache.get(self.path.strip('/'))
                self.send_response(200 if data else 404)
                self.end_headers()
                if data: self.wfile.write(data)

        server = HTTPServer(('127.0.0.1', 0), Handler)
        http_port = server.server_address[1]
        threading.Thread(target=server.serve_forever, daemon=True).start()
        print(f"Local HTTP server running on http://127.0.0.1:{http_port}")
        
        param_list = [(seq, tracker_info, debug, visdom_info) for seq, tracker_info in product(dataset, trackers)]
        pkl_https = []
        for i, args in enumerate(param_list):
            param_cache[str(i)] = pickle.dumps(args)
            pkl_https.append(f"http://127.0.0.1:{http_port}/{i}")
        with ThreadPoolExecutor(max_workers=threads) as pool:
            pool.map(run_one, pkl_https)

    print('Done')
