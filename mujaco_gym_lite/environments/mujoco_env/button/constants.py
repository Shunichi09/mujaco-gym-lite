import numpy as np

from mujaco_gym_lite.utils.views import surrounding_view

TASK_RELATED_OBJECT_NAMES = ["button_surface", "j2n6s300_hand", "j2n6s300_finger", "button_stop_top"]

_visibility_views, _visibility_views_dict = surrounding_view(point=np.array([0.1, 0.225, 0.25]), radius=0.75)

PREDEFINED_CAMERA_VIEWS = {
    "oracle": _visibility_views[:5],
    "vlm": _visibility_views_dict,
    "human": _visibility_views_dict,
    "visibility": _visibility_views,
}

PREDEFINED_CAMERA_VIEW_TEMPERATURE = [0.1, 3.0, 2.0, 1.0]
PREDEFINED_CAMERA_WRONG_RATES = [0.5, 0.4, 0.0, 0.3, 0.2]
