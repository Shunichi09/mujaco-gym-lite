import numpy as np

from mujaco_gym_lite.utils.views import surrounding_view

TASK_RELATED_OBJECT_NAMES = ["assembly_ring_main", "assembly_rod_main", "j2n6s300_hand", "j2n6s300_finger"]

_visibility_views, _visibility_views_dict = surrounding_view(point=np.array([-0.1, 0.1, 0.125]), radius=0.6)

PREDEFINED_CAMERA_VIEWS = {
    "oracle": _visibility_views[:5],
    "vlm": _visibility_views_dict,
    "human": _visibility_views_dict,
    "visibility": _visibility_views,
}
PREDEFINED_CAMERA_VIEW_TEMPERATURE = [0.1, 3.0, 2.0, 1.0]
PREDEFINED_CAMERA_WRONG_RATES = [0.5, 0.4, 0.0, 0.3, 0.2]
