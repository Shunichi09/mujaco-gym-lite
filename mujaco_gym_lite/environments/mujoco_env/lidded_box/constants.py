import numpy as np

from mujaco_gym_lite.utils.views import surrounding_view

TASK_RELATED_OBJECT_NAMES = ["lidded_box_handle", "j2n6s300_hand", "j2n6s300_finger"]

_visibility_views, _visibility_views_dict = surrounding_view(point=np.array([0.0, 0.0, 0.1]), radius=0.75)

PREDEFINED_CAMERA_VIEWS = {
    "oracle": _visibility_views[:5],
    "vlm": _visibility_views_dict,
    "human": _visibility_views_dict,
    "visibility": _visibility_views,
}
