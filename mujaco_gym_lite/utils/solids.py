import os
import pathlib
from typing import Optional, Union, cast

import numpy as np
import numpy.typing as npt
import open3d as o3d
from scipy.spatial import ConvexHull

from mujaco_gym_lite.utils.transforms import create_transformation_matrix


def fusion_mesh(
    meshes: Union[tuple[o3d.geometry.TriangleMesh, ...], list[o3d.geometry.TriangleMesh]],
) -> o3d.geometry.TriangleMesh:
    fusioned_mesh = o3d.geometry.TriangleMesh()
    for mesh in meshes:
        fusioned_mesh += mesh
    return fusioned_mesh


def create_coordinate_mesh(matrix: npt.NDArray, size: float) -> o3d.geometry.TriangleMesh:
    coordinate_mesh = o3d.geometry.TriangleMesh.create_coordinate_frame(size=size)
    coordinate_mesh.transform(matrix)
    return coordinate_mesh


def create_plane_mesh(matrix: npt.NDArray, width: float, height: float, depth: float) -> o3d.geometry.TriangleMesh:
    coordinate_mesh = o3d.geometry.TriangleMesh.create_box(width=width, height=height, depth=depth)
    # NOTE: create_box creates a box which center is not the origin
    coordinate_mesh.transform(
        create_transformation_matrix(translation=np.array([width * -0.5, height * -0.5, depth * -0.5]))
    )
    coordinate_mesh.transform(matrix)
    return coordinate_mesh


def create_point_mesh(
    matrix: npt.NDArray, radius: float, color: npt.NDArray = np.array([0.75, 0.0, 0.0])
) -> o3d.geometry.TriangleMesh:
    point_mesh = o3d.geometry.TriangleMesh.create_sphere(radius)
    point_mesh.transform(matrix)
    point_mesh.paint_uniform_color(color)
    return point_mesh


def load_mesh(stl_or_obj_path: pathlib.Path) -> o3d.geometry.TriangleMesh:
    if not os.path.exists(stl_or_obj_path):
        raise ValueError(f"Invalid path. Can not find {stl_or_obj_path}")
    stl_or_obj_data = o3d.io.read_triangle_model(str(stl_or_obj_path))
    assert len(stl_or_obj_data.meshes) == 1, "Invalid data. Contain more than two meshes."
    return stl_or_obj_data.meshes[0].mesh


def create_np_point_cloud_from_depth_image(
    depth: npt.NDArray, scale: float, K: npt.NDArray, organized: bool = False
) -> npt.NDArray:
    assert (3, 3) == K.shape
    height, width = depth.shape
    xmap = np.arange(width)
    ymap = np.arange(height)
    xmap, ymap = np.meshgrid(xmap, ymap)
    points_z = depth / scale
    fx = K[0, 0]
    fy = K[1, 1]
    cx = K[0, 2]
    cy = K[1, 2]
    points_x = (xmap - cx) * points_z / fx
    points_y = (ymap - cy) * points_z / fy
    cloud: npt.NDArray = np.stack([points_x, points_y, points_z], axis=-1)  # FIXME: Is this the correct way to cast ?
    if not organized:
        cloud = cloud.reshape([-1, 3])
    return cloud


def create_open3d_point_cloud_from_np_ndarray(
    position_array: npt.NDArray, rgb_array: Optional[npt.NDArray] = None
) -> o3d.geometry.PointCloud:
    o3d_point_cloud = o3d.geometry.PointCloud()
    o3d_point_cloud.points = o3d.utility.Vector3dVector(position_array)

    if rgb_array is not None:
        if np.max(rgb_array) > 1.0:
            raise ValueError("rgb array should be range [0 1]")
        o3d_point_cloud.colors = o3d.utility.Vector3dVector(rgb_array)

    return o3d_point_cloud


def crop_open3d_point_cloud(
    point_cloud: o3d.geometry.PointCloud, min_bound: npt.NDArray, max_bound: npt.NDArray
) -> o3d.geometry.PointCloud:
    bbox = o3d.geometry.AxisAlignedBoundingBox(min_bound, max_bound)
    point_cloud = point_cloud.crop(bbox)
    return point_cloud


def mesh_to_numpy_point_cloud(mesh: o3d.geometry.TriangleMesh, num_points: int, method: str = "poisson") -> npt.NDArray:
    if method == "poisson":
        o3d_pc = mesh.sample_points_poisson_disk(num_points)
    elif method == "uniform":
        o3d_pc = mesh.sample_points_uniformly(num_points)
    else:
        raise ValueError
    return np.array(o3d_pc.points)


def generate_sphere_points(radius: float, num_sphere_points: int, method: str = "uniform") -> npt.NDArray:
    if method == "uniform":
        theta = np.random.uniform(0, 2 * np.pi, num_sphere_points)
        phi = np.arccos(np.random.uniform(-1, 1, num_sphere_points))
        x = radius * np.sin(phi) * np.cos(theta)
        y = radius * np.sin(phi) * np.sin(theta)
        z = radius * np.cos(phi)
        return np.array([x, y, z]).T
    elif method == "poisson":
        raise NotImplementedError
    else:
        raise ValueError


def draw_geometries(
    mesh_or_point_cloud: Union[
        o3d.geometry.TriangleMesh, list[o3d.geometry.TriangleMesh], tuple[o3d.geometry.TriangleMesh, ...]
    ],
    mesh_show_wireframe: bool = False,
    with_coordinate_mesh: bool = False,
    coordinate_mesh_size: float = 0.05,
):
    if not (isinstance(mesh_or_point_cloud, tuple) or isinstance(mesh_or_point_cloud, list)):
        mesh_or_point_cloud = [mesh_or_point_cloud]

    if with_coordinate_mesh:
        o3d.visualization.draw_geometries(
            [
                *mesh_or_point_cloud,
                create_coordinate_mesh(np.eye(4), size=coordinate_mesh_size),
            ],
            mesh_show_wireframe=mesh_show_wireframe,
        )
    else:
        o3d.visualization.draw_geometries(mesh_or_point_cloud, mesh_show_wireframe=mesh_show_wireframe)


def are_points_inside_bbox(points: npt.NDArray, bbox_vertices: npt.NDArray) -> npt.NDArray:
    assert len(bbox_vertices.shape) == 2 and len(points.shape) == 2
    hull = ConvexHull(bbox_vertices)
    equations = hull.equations  # equation: [a, b, c, d]: a*x + b*y + c*z + d = 0
    points = np.atleast_2d(points)  # Ensure points is 2D array
    s = np.dot(points, equations[:, :3].T) + equations[:, 3]  # Dot product for all points
    return cast(npt.NDArray, np.all(s <= 1e-6, axis=1))  # list of bool for each points


def compute_cuboid_vertices(
    base_center: npt.ArrayLike, rotation_matrix: npt.NDArray, width: float, height: float, depth: float
) -> npt.NDArray:
    base_center = np.array(base_center)
    rotation_matrix = np.array(rotation_matrix)
    half_width = width / 2.0
    half_height = height / 2.0

    offsets = np.array(
        [
            [-half_width, -half_height, 0.0],
            [half_width, -half_height, 0.0],
            [-half_width, half_height, 0.0],
            [half_width, half_height, 0.0],
            [-half_width, -half_height, depth],
            [half_width, -half_height, depth],
            [-half_width, half_height, depth],
            [half_width, half_height, depth],
        ]
    )
    rotated_offsets = offsets @ rotation_matrix.T
    vertices = base_center + rotated_offsets
    return np.array(vertices)


def aligned_sphere_points(num_points: int, radius: float) -> npt.NDArray:
    phi = np.pi * (3.0 - np.sqrt(5.0))
    indices = np.arange(num_points)
    y = 1 - (2 * indices / (num_points - 1))
    r = np.sqrt(1 - y * y)
    theta = phi * indices
    x = r * np.cos(theta)
    z = r * np.sin(theta)
    points = np.stack([x, y, z], axis=-1) * radius
    return cast(npt.NDArray, points)
