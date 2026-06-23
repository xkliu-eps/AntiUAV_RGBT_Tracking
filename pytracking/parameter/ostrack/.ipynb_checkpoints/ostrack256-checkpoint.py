from pytracking.utils import TrackerParams
from pytracking.features.net_wrappers import NetWithBackbone
import os
from easydict import EasyDict as edict
from pytracking.evaluation.environment import env_settings


def parameters():
    params = TrackerParams()

    """
    Add default config for OSTrack.
    """
    cfg = edict()

    # MODEL
    cfg.MODEL = edict()
    cfg.MODEL.PRETRAIN_FILE = "mae_pretrain_vit_base.pth"
    cfg.MODEL.EXTRA_MERGER = False

    cfg.MODEL.RETURN_INTER = False
    cfg.MODEL.RETURN_STAGES = []

    # MODEL.BACKBONE
    cfg.MODEL.BACKBONE = edict()
    cfg.MODEL.BACKBONE.TYPE = "vit_base_patch16_224"
    cfg.MODEL.BACKBONE.STRIDE = 16
    cfg.MODEL.BACKBONE.MID_PE = False
    cfg.MODEL.BACKBONE.SEP_SEG = False
    cfg.MODEL.BACKBONE.CAT_MODE = 'direct'
    cfg.MODEL.BACKBONE.MERGE_LAYER = 0
    cfg.MODEL.BACKBONE.ADD_CLS_TOKEN = False
    cfg.MODEL.BACKBONE.CLS_TOKEN_USE_MODE = 'ignore'

    cfg.MODEL.BACKBONE.CE_LOC = []
    cfg.MODEL.BACKBONE.CE_KEEP_RATIO = []
    cfg.MODEL.BACKBONE.CE_TEMPLATE_RANGE = 'ALL'  # choose between ALL, CTR_POINT, CTR_REC, GT_BOX

    # MODEL.HEAD
    cfg.MODEL.HEAD = edict()
    cfg.MODEL.HEAD.TYPE = "CENTER"
    cfg.MODEL.HEAD.NUM_CHANNELS = 256

    # TRAIN
    cfg.TRAIN = edict()
    cfg.TRAIN.LR = 0.0001
    cfg.TRAIN.WEIGHT_DECAY = 0.0001
    cfg.TRAIN.EPOCH = 500
    cfg.TRAIN.LR_DROP_EPOCH = 400
    cfg.TRAIN.BATCH_SIZE = 16
    cfg.TRAIN.NUM_WORKER = 8
    cfg.TRAIN.OPTIMIZER = "ADAMW"
    cfg.TRAIN.BACKBONE_MULTIPLIER = 0.1
    cfg.TRAIN.GIOU_WEIGHT = 2.0
    cfg.TRAIN.L1_WEIGHT = 5.0
    cfg.TRAIN.FREEZE_LAYERS = [0, ]
    cfg.TRAIN.PRINT_INTERVAL = 50
    cfg.TRAIN.VAL_EPOCH_INTERVAL = 20
    cfg.TRAIN.GRAD_CLIP_NORM = 0.1
    cfg.TRAIN.AMP = False

    cfg.TRAIN.CE_START_EPOCH = 20  # candidate elimination start epoch
    cfg.TRAIN.CE_WARM_EPOCH = 80  # candidate elimination warm up epoch
    cfg.TRAIN.DROP_PATH_RATE = 0.1  # drop path rate for ViT backbone

    # TRAIN.SCHEDULER
    cfg.TRAIN.SCHEDULER = edict()
    cfg.TRAIN.SCHEDULER.TYPE = "step"
    cfg.TRAIN.SCHEDULER.DECAY_RATE = 0.1

    # DATA
    cfg.DATA = edict()
    cfg.DATA.SAMPLER_MODE = "causal"  # sampling methods
    cfg.DATA.MEAN = [0.485, 0.456, 0.406]
    cfg.DATA.STD = [0.229, 0.224, 0.225]
    cfg.DATA.MAX_SAMPLE_INTERVAL = 200
    # DATA.TRAIN
    cfg.DATA.TRAIN = edict()
    cfg.DATA.TRAIN.DATASETS_NAME = ["LASOT", "GOT10K_vottrain"]
    cfg.DATA.TRAIN.DATASETS_RATIO = [1, 1]
    cfg.DATA.TRAIN.SAMPLE_PER_EPOCH = 60000
    # DATA.VAL
    cfg.DATA.VAL = edict()
    cfg.DATA.VAL.DATASETS_NAME = ["GOT10K_votval"]
    cfg.DATA.VAL.DATASETS_RATIO = [1]
    cfg.DATA.VAL.SAMPLE_PER_EPOCH = 10000
    # DATA.SEARCH
    cfg.DATA.SEARCH = edict()
    cfg.DATA.SEARCH.SIZE = 320
    cfg.DATA.SEARCH.FACTOR = 5.0
    cfg.DATA.SEARCH.CENTER_JITTER = 4.5
    cfg.DATA.SEARCH.SCALE_JITTER = 0.5
    cfg.DATA.SEARCH.NUMBER = 1
    # DATA.TEMPLATE
    cfg.DATA.TEMPLATE = edict()
    cfg.DATA.TEMPLATE.NUMBER = 1
    cfg.DATA.TEMPLATE.SIZE = 128
    cfg.DATA.TEMPLATE.FACTOR = 2.0
    cfg.DATA.TEMPLATE.CENTER_JITTER = 0
    cfg.DATA.TEMPLATE.SCALE_JITTER = 0

    # TEST
    cfg.TEST = edict()
    cfg.TEST.TEMPLATE_FACTOR = 2.0
    cfg.TEST.TEMPLATE_SIZE = 128
    cfg.TEST.SEARCH_FACTOR = 5.0
    cfg.TEST.SEARCH_SIZE = 320
    cfg.TEST.EPOCH = 500

    # update default config from yaml file
    cfg.DATA.SEARCH.CENTER_JITTER = 3
    cfg.DATA.SEARCH.FACTOR = 4.0
    cfg.DATA.SEARCH.SCALE_JITTER = 0.25
    cfg.DATA.SEARCH.SIZE = 256
    cfg.DATA.TRAIN.DATASETS_NAME = ['uav300inf']
    cfg.DATA.TRAIN.DATASETS_RATIO = [1]
    cfg.DATA.VAL.DATASETS_NAME = ['uav300inf_val']
    cfg.TRAIN.BATCH_SIZE = 32
    cfg.TRAIN.EPOCH = 30
    cfg.TRAIN.LR = 0.0004
    cfg.TRAIN.LR_DROP_EPOCH = 240
    cfg.TRAIN.NUM_WORKER = 10
    cfg.TEST.EPOCH = 30
    cfg.TEST.SEARCH_FACTOR = 4.0
    cfg.TEST.SEARCH_SIZE = 256

    params.cfg = cfg
    # template and search region
    params.template_factor = cfg.TEST.TEMPLATE_FACTOR
    params.template_size = cfg.TEST.TEMPLATE_SIZE
    params.search_factor = cfg.TEST.SEARCH_FACTOR
    params.search_size = cfg.TEST.SEARCH_SIZE
    params.debug = 0
    params.visualization = False
    # Network checkpoint path
    params.checkpoint = os.path.join(env_settings().network_path, "OSTrack_ep0300.pth.tar")

    # whether to save boxes from all queries
    params.save_all_boxes = False

    return params
