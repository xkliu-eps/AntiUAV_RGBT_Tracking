# include "core.multiModeTrack"


def get_tracker_list():
    from pytracking.utils import TrackerParams
    import pyTrackBridge.tracker.OCTA_SOT_v_0_1 as OCTA_SOT_v_0_1

    def get_tracker():
        params = TrackerParams()
        params.infrared_params = 'dimp.dimp50infrared'
        params.visible_params = 'dimp.dimp50visible'
        params.visualization = False
        return OCTA_SOT_v_0_1.get_tracker_class()(params)

    return [
        {
            'name': 'multiTrack/OCTA_STO.0.1',
            'run_ids': None,
            'tracker_setting': {
                'name': 'test',
                'type': 'function',
                'function': get_tracker
            }
        }

    ]


def get_sequence_list():
    import pyTrackBridge.dataset.anti_uav300dataset as antiUav300
    return antiUav300.UAV300Dataset_BothConfig(mode='join', uav300_path='D:/ir_dataset/Anti-UAV300/').get_sequence_list()
