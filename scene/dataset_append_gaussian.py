#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#
import pickle
import shutil
import os
import sys
from PIL import Image
from typing import NamedTuple
from scene.colmap_loader import read_extrinsics_text, read_intrinsics_text, qvec2rotmat, \
    read_extrinsics_binary, read_intrinsics_binary, read_points3D_binary, read_points3D_text
from utils.graphics_utils import getWorld2View2, focal2fov, fov2focal
import numpy as np
import json
from pathlib import Path
from plyfile import PlyData, PlyElement
from utils.sh_utils import SH2RGB
from scene.gaussian_model import BasicPointCloud
from sklearn.cluster import DBSCAN
import time


class CameraInfo(NamedTuple):
    uid: int
    global_id: int
    R: np.array
    T: np.array
    FovY: np.array
    FovX: np.array
    image_path: str
    image_name: str
    width: int
    height: int
    fx: float
    fy: float


class SceneInfo(NamedTuple):
    point_cloud: BasicPointCloud
    train_cameras: list
    test_cameras: list
    nerf_normalization: dict
    ply_path: str


def getNerfppNorm(cam_info):
    def get_center_and_diag(cam_centers):
        cam_centers = np.hstack(cam_centers)
        avg_cam_center = np.mean(cam_centers, axis=1, keepdims=True)
        center = avg_cam_center
        dist = np.linalg.norm(cam_centers - center, axis=0, keepdims=True)
        diagonal = np.max(dist)
        return center.flatten(), diagonal

    cam_centers = []

    for cam in cam_info:
        W2C = getWorld2View2(cam.R, cam.T)
        C2W = np.linalg.inv(W2C)
        cam_centers.append(C2W[:3, 3:4])

    center, diagonal = get_center_and_diag(cam_centers)
    radius = diagonal * 1.1

    translate = -center

    return {"translate": translate, "radius": radius}


def load_poses(pose_path, num):
    poses = []
    with open(pose_path, "r") as f:
        lines = f.readlines()
    for i in range(num):
        line = lines[i]
        c2w = np.array(list(map(float, line.split()))).reshape(4, 4)
        c2w[:3, 3] = c2w[:3, 3] * 10.0
        w2c = np.linalg.inv(c2w)
        w2c = w2c
        poses.append(w2c)
    poses = np.stack(poses, axis=0)
    return poses


def partition_cameras(cam_infos, output_folder, overlap_ratio=0.2, grid_size=(3, 3)):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Step 1: 获取所有相机的位移矩阵T的xy坐标
    positions = np.array([cam.T[:2] for cam in cam_infos])  # 只获取 T 的 xy 坐标

    # Step 2: 计算边界并创建网格
    min_xy = positions.min(axis=0)
    max_xy = positions.max(axis=0)
    step = (max_xy - min_xy) / grid_size  # 每个网格的大小（不含重叠）

    # 考虑重叠，将每个网格的范围扩展
    x_overlap = step[0] * overlap_ratio
    y_overlap = step[1] * overlap_ratio

    # Step 3: 遍历每个网格，找出在该区域内的相机
    for i in range(grid_size[0]):
        for j in range(grid_size[1]):
            x_min = min_xy[0] + i * step[0] - x_overlap
            x_max = min_xy[0] + (i + 1) * step[0] + x_overlap
            y_min = min_xy[1] + j * step[1] - y_overlap
            y_max = min_xy[1] + (j + 1) * step[1] + y_overlap

            # 筛选出在此网格内的相机
            selected_cameras = [
                cam.image_name for cam in cam_infos
                if x_min <= cam.T[0] <= x_max and y_min <= cam.T[1] <= y_max
            ]

            # Step 4: 将每个网格的相机名称保存到文件
            grid_filename = f"grid_{i}_{j}.txt"
            grid_filepath = os.path.join(output_folder, grid_filename)
            with open(grid_filepath, "w") as f:
                f.write("\n".join(selected_cameras))

    print(f"分区完成，每个网格的相机名称已保存到 {output_folder} 文件夹中。")


def readColmapCameras(cam_extrinsics, cam_intrinsics, images_folder):
    cam_infos = []
    for idx, key in enumerate(cam_extrinsics):
        sys.stdout.write('\r')
        # the exact output you're looking for:
        sys.stdout.write("Reading camera {}/{}".format(idx + 1, len(cam_extrinsics)))
        sys.stdout.flush()

        extr = cam_extrinsics[key]
        intr = cam_intrinsics[extr.camera_id]
        height = intr.height
        width = intr.width

        uid = intr.id
        R = np.transpose(qvec2rotmat(extr.qvec))
        T = np.array(extr.tvec)

        if intr.model == "SIMPLE_PINHOLE":
            focal_length_x = intr.params[0]
            FovY = focal2fov(focal_length_x, height)
            FovX = focal2fov(focal_length_x, width)
        elif intr.model == "PINHOLE":
            focal_length_x = intr.params[0]
            focal_length_y = intr.params[1]
            FovY = focal2fov(focal_length_y, height)
            FovX = focal2fov(focal_length_x, width)
        else:
            assert False, "Colmap camera model not handled: only undistorted datasets (PINHOLE or SIMPLE_PINHOLE cameras) supported!"

        image_path = os.path.join(images_folder, os.path.basename(extr.name))
        image_name = os.path.basename(image_path).split(".")[0]

        cam_info = CameraInfo(uid=uid, global_id=idx, R=R, T=T, FovY=FovY, FovX=FovX,
                              image_path=image_path, image_name=image_name,
                              width=width, height=height, fx=focal_length_x, fy=focal_length_y)
        cam_infos.append(cam_info)
    output_folder = "/data/wtf/cvpr/PGSR/data_GauUScene/Modern_Building"  # 输出文件夹路径
    partition_cameras(cam_infos, output_folder)

    sys.stdout.write('\n')
    return cam_infos


def fetchPly(path):
    plydata = PlyData.read(path)
    vertices = plydata['vertex']
    positions = np.vstack([vertices['x'], vertices['y'], vertices['z']]).T
    colors = np.vstack([vertices['red'], vertices['green'], vertices['blue']]).T / 255.0
    normals = np.vstack([vertices['nx'], vertices['ny'], vertices['nz']]).T
    return BasicPointCloud(points=positions, colors=colors, normals=normals)


def storePly(path, xyz, rgb):
    # Define the dtype for the structured array
    dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
             ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),
             ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]

    normals = np.zeros_like(xyz)

    elements = np.empty(xyz.shape[0], dtype=dtype)
    attributes = np.concatenate((xyz, normals, rgb), axis=1)
    elements[:] = list(map(tuple, attributes))

    # Create the PlyData object and write to file
    vertex_element = PlyElement.describe(elements, 'vertex')
    ply_data = PlyData([vertex_element])
    ply_data.write(path)


# ################################################################
import torch
import numpy as np


def fov2focal(fov, dim, device='cuda'):
    fov_tensor = torch.tensor(fov, dtype=torch.float16, device=device)
    dim_tensor = torch.tensor(dim, dtype=torch.float16, device=device)
    return dim_tensor / (2 * torch.tan(fov_tensor / 2))


def world_to_camera_point(points, R, T):
    R = torch.tensor(R, dtype=torch.float16, device=points.device)
    T = torch.tensor(T, dtype=torch.float16, device=points.device).view(3, 1)
    points = points.clone().detach()  # 使用 clone().detach() 复制 tensor
    points = points.float()  # 如果 points 不是 Float 类型，则转换为 Float
    R = R.float()  # 如果 R 不是 Float 类型，则转换为 Float
    T = T.float()
    points_camera = torch.matmul(R.t(), points.t()) + T
    points_camera = points_camera.t()
    return points_camera


def project_to_image(points, cam_info, scale):
    fx = fov2focal(cam_info.FovX, cam_info.width)
    fy = fov2focal(cam_info.FovY, cam_info.height)
    cx, cy = cam_info.width / 2, cam_info.height / 2

    points_camera = world_to_camera_point(points, cam_info.R, cam_info.T)

    x = points_camera[:, 0]
    y = points_camera[:, 1]
    z = points_camera[:, 2]

    valid = z > 0
    u = fx * x / z + cx
    v = fy * y / z + cy

    # Apply scale factor to account for mask downsampling
    u = u * scale
    v = v * scale

    # 将无效点（即在相机后面的点）的 u 和 v 设置为 -1
    u[~valid] = -1
    v[~valid] = -1

    return u, v


def remove_outliers(points, z_threshold=2):
    median = torch.median(points, dim=0).values
    abs_diff = torch.abs(points - median)
    mad = torch.median(abs_diff, dim=0).values
    modified_z_scores = 0.6745 * abs_diff / mad
    mask = modified_z_scores < z_threshold
    return points[mask.all(dim=1)]


# 假设我们已经定义了一个函数来计算相机的 Z 轴旋转角度
def get_z_axis_rotation(cam_info):
    R = cam_info.R  # 获取旋转矩阵
    camera_z_axis = R[:, 2]  # 获取相机 Z 轴方向（旋转矩阵的第三列）
    world_z_axis = np.array([0, 0, 1])  # 原始 Z 轴方向

    # 计算相机 Z 轴与世界 Z 轴之间的夹角
    cos_theta_z = np.dot(camera_z_axis, world_z_axis) / (np.linalg.norm(camera_z_axis) * np.linalg.norm(world_z_axis))
    theta_z_rad = np.arccos(cos_theta_z)  # 夹角（弧度）

    # 返回角度（度）
    return np.degrees(theta_z_rad)


from sklearn.neighbors import NearestNeighbors


def remove_outliers_statistical(cluster_points, k_neighbors=10, std_threshold=0.01):
    """
    基于统计的离群点移除（Statistical Outlier Removal）。

    :param cluster_points: 输入的点云数据
    :param k_neighbors: 用于计算的最近邻数量
    :param std_threshold: 离群点的阈值，基于标准差进行过滤
    :return: 去除离群点后的点云
    """
    # 使用 K 最近邻来计算每个点与其邻居的距离
    nbrs = NearestNeighbors(n_neighbors=k_neighbors).fit(cluster_points)
    distances, _ = nbrs.kneighbors(cluster_points)

    # 计算每个点到其 k 个邻居的平均距离
    avg_distances = np.mean(distances, axis=1)

    # 计算平均距离的标准差
    mean_distance = np.mean(avg_distances)
    std_distance = np.std(avg_distances)

    # 设定阈值，去除距离大于 mean + std_threshold * std_distance 的点
    filtered_points = cluster_points[avg_distances < mean_distance + std_threshold * std_distance]

    return filtered_points


def generate_dense_circle_points(cluster_points, min_z, max_z, radius, density=500):
    """
    在给定的平面上生成圆形密集点云。

    :param cluster_points: 聚类的点云，在 xy 平面的投影。
    :param min_z: 聚类中最低点的 z 值。
    :param max_z: 聚类中最高点的 z 值。
    :param radius: 外接圆的半径。
    :param density: 圆形密集点云的密集度。
    :return: 生成的圆形点云，形状为 (M, 3)。
    """
    # 先进行基于统计的离群点移除
    cluster_points = remove_outliers_statistical(cluster_points)

    # 计算圆心的z值
    center_z = max_z  # 圆心的 z 值等于最高点的 z 值

    # 计算外接圆的圆心坐标 (xy)，假设 cluster_points 已经是平面上的点云投影
    centroid_xy = np.mean(cluster_points, axis=0)  # 计算点云的外接圆的圆心

    # 计算新的半径
    # 半径等于外接圆的半径加上最高点到最低点的差值
    radius = np.max(np.linalg.norm(cluster_points - centroid_xy, axis=1)) + (max_z - min_z)
    radius = radius
    # 生成圆形点云
    theta = np.linspace(0, 2 * np.pi, int(density * radius))
    r_values = np.linspace(0, radius, int(density))
    circle_points = []

    for r in r_values:
        for angle in theta:
            x = centroid_xy[0] + r * np.cos(angle)
            y = centroid_xy[1] + r * np.sin(angle)
            circle_points.append([x, y, center_z])

    return np.array(circle_points)


# 计算点和相机中心的连线与 z 轴的夹角
def calculate_angles_with_z_axis(label_points_3d, camera_center):
    # 计算所有点与相机中心的连线与 z 轴的夹角
    vectors = label_points_3d - camera_center  # 计算每个点到相机中心的向量
    z_axis_vector = np.array([0, 0, 1])  # z 轴方向向量

    # 计算点与 z 轴的点积
    dot_products = np.dot(vectors, z_axis_vector)  # 点积
    vectors_magnitudes = np.linalg.norm(vectors, axis=1)  # 每个向量的模长
    z_axis_magnitude = np.linalg.norm(z_axis_vector)  # z 轴的模长

    # 计算夹角的余弦值
    cos_thetas = dot_products / (vectors_magnitudes * z_axis_magnitude)

    # 计算夹角 theta 的弧度值
    angles_rad = np.arccos(np.clip(cos_thetas, -1.0, 1.0))  # 防止计算出 NaN
    angles_deg = np.degrees(angles_rad)  # 转换为角度

    return angles_deg


# 设置旋转角度阈值
angle_threshold = 10


def apply_mask_to_point_cloud(points, colors, masks, camera_infos, scale, path):
    # 初始化张量
    points_tensor = torch.tensor(points, dtype=torch.float, device='cuda')
    print("处理前的点数：", points_tensor.shape[0])

    # 初始化有效点标签和投影计数
    labels_chunk = torch.zeros(points_tensor.shape[0], dtype=torch.uint8, device='cuda')
    false_proj_counts = torch.zeros(points_tensor.shape[0], dtype=torch.uint8, device='cuda')

    # 预处理步骤计时
    preproc_start_time = time.time()

    # 将掩码转换为 CUDA 张量（假设掩码列表较大，建议放在循环外部进行预处理）
    masks_cuda = [torch.tensor(mask, dtype=torch.uint8, device='cuda') for mask in masks]

    for cam_info, mask_tensor in zip(camera_infos, masks_cuda):
        u, v = project_to_image(points_tensor, cam_info, scale)
        u = u.long()
        v = v.long()

        # 筛选出在图像范围内的点
        valid_mask = (u >= 0) & (u < mask_tensor.shape[1]) & (v >= 0) & (v < mask_tensor.shape[0])
        valid_indices = valid_mask.nonzero(as_tuple=True)

        # 提取在范围内的点
        u_valid = u[valid_indices]
        v_valid = v[valid_indices]

        # 获取掩码值并应用有效性筛选
        valid_labels = mask_tensor[v_valid, u_valid]
        positive_mask = valid_labels > 0
        positive_indices = tuple(idx[positive_mask] for idx in valid_indices)

        # 更新 labels_chunk 只赋值有效标签
        labels_chunk[positive_indices] = valid_labels[positive_mask]

        # 统计投影低于128的情况
        false_proj = valid_labels < 128
        false_proj_counts[valid_indices] += false_proj.long()

    # 清理 CUDA 缓存
    torch.cuda.empty_cache()

    # 设置有效点条件
    false_tolerance = 20
    valid_points = (labels_chunk > 128) & (false_proj_counts <= false_tolerance)

    # 筛选符合条件的点云和颜色
    points_tensor = points_tensor[valid_points]
    colors = np.array(colors)[valid_points.cpu().numpy()]

    # 输出处理后的点数
    print("处理后的点数：", points_tensor.shape[0])

    # 存储处理后的点云数据到 .ply 文件
    output_path = path + f"{false_tolerance}_处理后_points.ply"
    storePly(output_path, points_tensor.cpu().numpy(), colors)

    # 打印预处理时间
    preproc_end_time = time.time()
    print("预处理步骤所用时间：", preproc_end_time - preproc_start_time, "秒")

    ###########################################点云聚类分割
    # 先去除绿色植被
    is_green = ((colors[:, 1] > colors[:, 0]) & (colors[:, 1] > colors[:, 2]) &
                (colors[:, 1] > colors[:, 0] + 15) | (colors[:, 1] > colors[:, 2] + 15))
    # # 筛选出非绿色的点云和颜色
    non_green_points = ~is_green
    points_tensor = points_tensor[non_green_points]
    colors = colors[non_green_points]
    #############################################分割

    # 输出处理后的点数
    print("处理后的点数（去除植被后）：", points_tensor.shape[0])

    # 存储处理后的点云数据到 .ply 文件
    output_path = path + f"{false_tolerance}_去除植被后_points.ply"
    storePly(output_path, points_tensor.cpu().numpy(), colors)
    ############################################################################
    # 假设 points_tensor 是包含点云坐标的张量，colors 是对应的颜色数组
    """points = points_tensor.cpu().numpy()

    # 选择投影平面，这里选择 XZ 平面，即只保留 X 和 Y 维度
    projected_points = points[:, [0, 1]]  # 选择第1列（X）和第2列（Y）

    # 设置 DBSCAN 参数
    eps = 0.05  # 距离阈值，根据点云的密集度调整
    min_samples = 100  # 每个簇的最小点数

    # 记录聚类开始时间
    start_time = time.time()

    # 执行 DBSCAN 聚类
    clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(projected_points)
    """
    # 设置 DBSCAN 参数并进行聚类
    eps = 0.05
    min_samples = 100
    # 记录聚类开始时间
    start_time = time.time()
    points = points_tensor.cpu().numpy()
    original_points = points_tensor.cpu().numpy()
    projected_points = points[:, [0, 1]]
    clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(projected_points)
    labels = torch.tensor(clustering.labels_, dtype=torch.int16, device='cuda') + 1  # DBSCAN 标签从 1 开始
    original_labels = labels
    # 记录扩增后的点和标签
    all_points = [points]
    all_labels = [labels]

    # 获取每个类的点云并生成密集圆形点云
    unique_labels = np.unique(clustering.labels_)
    for label in unique_labels:
        if label == -1:
            # 跳过噪声点
            continue

        # 获取该类的所有点
        class_mask = clustering.labels_ == label
        class_points = points[class_mask]

        # 计算外接圆的两倍半径
        class_projected = class_points[:, :2]  # 投影到 xy 平面
        center_xy = class_projected.mean(axis=0)
        max_distance = np.max(np.linalg.norm(class_projected - center_xy, axis=1))
        radius = max_distance * 2

        # 找到最低点
        min_z = np.min(class_points[:, 2])
        max_z = np.max(class_points[:, 2])

        # 生成密集的圆形点云
        dense_circle_points = generate_dense_circle_points(class_projected, min_z, max_z, radius)

        # 赋予与原类点相同的标签
        dense_circle_labels = torch.full((dense_circle_points.shape[0],), label + 1, dtype=torch.int16, device='cuda')

        # 将生成的圆形点云和标签添加到总点云和标签中
        all_points.append(dense_circle_points)
        all_labels.append(dense_circle_labels)

    # 合并所有点和标签
    all_points = np.vstack(all_points)
    all_labels = torch.cat(all_labels)

    # 将结果转换回张量格式
    points_tensor = torch.tensor(all_points, dtype=torch.float32, device='cuda')
    labels = all_labels
    # 计算聚类时间
    clustering_time = time.time() - start_time
    ################################################################################
    print(f"DBSCAN 聚类耗时：{clustering_time:.2f} 秒")

    ########################################获取每个标签数据
    from concurrent.futures import ThreadPoolExecutor

    start_time = time.time()
    # 假设 `camera_infos` 是相机参数列表，`masks_cuda` 是原始的 mask 列表，`points_tensor` 是聚类后的点云张量
    points = points_tensor.cpu().numpy()
    labels = torch.tensor(labels, dtype=torch.int16, device='cuda') + 1  # 获取 DBSCAN 的标签

    # 设置输出路径
    path_single = path + "_partition_final"
    unique_labels = set(labels.cpu().numpy())
    cam_info_lists = {label: [] for label in unique_labels if label != 0}  # 跳过噪声点标签 0
    ######################################################
    # 遍历每个簇并保存
    for label in unique_labels:
        if label == 0:
            continue  # 跳过噪声点（-1 标签）

        # 提取当前簇的点和对应的颜色
        cluster_points = points_tensor[labels.cpu().numpy() == label]
        # cluster_colors = colors[labels.cpu().numpy() == label]  # 使用原始颜色，不随机生成

        # 生成随机颜色（范围 0-255）并应用于整个簇
        random_color = np.random.randint(0, 256, size=(1, 3))  # 随机生成一个 RGB 颜色
        cluster_colors = np.tile(random_color, (cluster_points.shape[0], 1))  # 将颜色应用于整个簇

        # 使用 label 作为文件编号
        output_path = os.path.join(path_single, f"label_{label}_point3D.ply")
        os.makedirs(os.path.join(path_single), exist_ok=True)
        # 调用 storePly 保存每个簇的点云数据
        storePly(output_path, cluster_points.cpu().numpy(), cluster_colors)
    print("保存稀疏点云")
    ######################################################

    # 提前创建文件夹结构
    for label in unique_labels:
        if label == 0:
            continue  # 跳过噪声点
        os.makedirs(os.path.join(path_single, f"label_{label}", "mask_points"), exist_ok=True)
        os.makedirs(os.path.join(path_single, f"label_{label}", "images"), exist_ok=True)

    # 并行保存函数
    def save_image_and_data(label_id, cam_info, mask_merge_processed, gt_image_target_path):
        # 保存 mask 和 multi_mask
        mask_image_path = os.path.join(path_single, f"label_{label_id}", "mask_points", f"{cam_info.image_name}.JPG")
        Image.fromarray(mask_merge_processed).save(mask_image_path)
        # 原始路径
        original_path = cam_info.image_path

        # 替换路径中的 "_4"
        modified_path = original_path.replace("data/wtf/cvpr/PGSR/data_GauUScene_4",
                                              "data/wtf/cvpr/PGSR/data_GauUScene")

        # 复制原始图像文件
        shutil.copy(modified_path, gt_image_target_path)

    # 投影和处理每张影像
    with ThreadPoolExecutor(max_workers=40) as executor:
        futures = []
        # 在每一个相机中处理标签
        for cam_idx, cam_info in enumerate(camera_infos):
            mask_merge = torch.zeros_like(masks_cuda[0], dtype=torch.int16, device='cuda')

            # 投影所有点到当前影像
            u, v = project_to_image(points_tensor, cam_info, scale)
            u, v = u.long(), v.long()

            # 筛选出在图像范围内的点
            valid_mask = (u >= 0) & (u < mask_merge.shape[1]) & (v >= 0) & (v < mask_merge.shape[0])
            valid_u, valid_v = u[valid_mask], v[valid_mask]
            valid_labels = labels[valid_mask]

            # 将有效标签点投影到 mask_merge 中
            mask_merge[valid_v, valid_u] = valid_labels

            # 获取当前影像中存在的标签种类
            current_labels = torch.unique(valid_labels).cpu().numpy()

            # 处理每个标签的 mask
            for label_id in current_labels:
                if label_id == 0:
                    continue  # 跳过噪声点

                # 获取标签点的坐标
                label_points = original_points[original_labels.cpu().numpy() == label_id]  # 获取当前标签对应的所有点
                label_points_3d = label_points[:, :3]  # 仅使用3D坐标部分

                # 计算所有点与相机中心的夹角
                angles_deg = calculate_angles_with_z_axis(label_points_3d, cam_info.T)  # cam_info.T 是相机中心坐标

                # 检查是否有至少 100 个点的夹角在 45 到 135 度之间
                # if np.sum((angles_deg >= 75) & (angles_deg <= 135)) >= 100:
                # continue  # 如果有至少 100 个点满足条件，跳过该标签的后续运算

                # 以下部分是标签不满足条件时的处理
                # 创建灰度 mask 并准备保存路径
                mask_numpy = mask_merge.cpu().numpy()
                if np.sum(mask_numpy == label_id) > 1000:
                    # 创建灰度 mask，并准备保存路径
                    mask_merge_processed = np.where(mask_numpy == label_id, 255, 0).astype(np.uint8)
                    if 1 > 0:
                        gt_image_target_path = os.path.join(path_single, f"label_{label_id}", "images",
                                                            f"{cam_info.image_name}.JPG")

                        # 将保存操作提交到线程池
                        futures.append(
                            executor.submit(save_image_and_data, label_id, cam_info, mask_merge_processed,
                                            gt_image_target_path))

                        # 把符合条件的 cam_info 添加到对应标签的 cam_info_list 中
                        cam_info_copy = cam_info._replace(image_path=gt_image_target_path)
                        cam_info_lists[label_id].append(cam_info_copy)

        # 等待所有保存任务完成
        for future in futures:
            future.result()

    save_image_time = time.time() - start_time
    print(f"保存图像及点云耗时：{save_image_time:.2f} 秒")

    # 保存 cam_info_list 到 pkl 文件
    start_time = time.time()
    for label_id, cam_info_list in cam_info_lists.items():
        label_folder = os.path.join(path_single, f"label_{label_id}")
        cam_info_pkl_path = os.path.join(label_folder, 'cam_info_list.pkl')
        with open(cam_info_pkl_path, 'wb') as f:
            pickle.dump(cam_info_list, f)
        print(f"保存标签 {label_id} 的相机信息文件到：{cam_info_pkl_path}")

    save_cam_time = time.time() - start_time
    print(f"保存相机耗时：{save_cam_time:.2f} 秒")

    ###########################################保存点云

    # 遍历每个簇并保存
    for label in unique_labels:
        if label == 0:
            continue  # 跳过噪声点（-1 标签）

        # 提取当前簇的点和对应的颜色
        cluster_point = points[labels.cpu().numpy() == label]
        # 为每个簇的点设置白色（255, 255, 255）
        cluster_color = np.ones((cluster_point.shape[0], 3), dtype=np.uint8) * 255  # 所有颜色设置为白色
        # 使用 label 作为文件编号
        output_path = os.path.join(path_single, f"label_{label}", "point3D.ply")

        # 记录保存开始时间
        start_save_time = time.time()

        # 调用 storePly 保存每个簇的点云数据
        storePly(output_path, cluster_point, cluster_color)

        # 计算保存时间
        saving_time = time.time() - start_save_time
        print(f"保存簇 {label} 的点云数据耗时：{saving_time:.2f} 秒，保存路径：{output_path}")

    # 输出总耗时
    total_time = time.time() - start_time
    print(f"总耗时：{total_time:.2f} 秒")
    # 生成随机颜色（范围 0-255）并应用于整个簇
    random_color_all = np.random.randint(0, 256, size=(1, 3))  # 随机生成一个 RGB 颜色
    cluster_colors_all = np.tile(random_color_all, (points_tensor.cpu().numpy().shape[0], 1))  # 将颜色应用于整个簇

    return points_tensor.cpu().numpy(), cluster_colors_all


###############################################################


def readColmapSceneInfo(path, images, eval, llffhold=8):
    try:
        cameras_extrinsic_file = os.path.join(path, "sparse", "images.bin")
        cameras_intrinsic_file = os.path.join(path, "sparse", "cameras.bin")
        cam_extrinsics = read_extrinsics_binary(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_binary(cameras_intrinsic_file)
    except:
        cameras_extrinsic_file = os.path.join(path, "sparse", "images.txt")
        cameras_intrinsic_file = os.path.join(path, "sparse", "cameras.txt")
        cam_extrinsics = read_extrinsics_text(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_text(cameras_intrinsic_file)

    reading_dir = "images" if images == None else images
    cam_infos_unsorted = readColmapCameras(cam_extrinsics=cam_extrinsics, cam_intrinsics=cam_intrinsics,
                                           images_folder=os.path.join(path, reading_dir))
    cam_infos = sorted(cam_infos_unsorted.copy(), key=lambda x: x.image_name)

    mask_folder = os.path.join(path, "mask")
    masks = []
    determined_extension = None  # 用于存储第一次找到的有效扩展名

    for cam_info in cam_infos:
        if determined_extension is None:
            # 如果尚未确定扩展名，则尝试找到有效的扩展名
            possible_extensions = [".png", ".jpg", ".JPG"]
            for ext in possible_extensions:
                temp_path = os.path.join(mask_folder, f"{cam_info.image_name}{ext}")
                if os.path.exists(temp_path):
                    determined_extension = ext  # 确定扩展名
                    mask_path = temp_path
                    break
            if determined_extension is None:
                print(f"未找到 {cam_info.image_name} 的有效 mask 文件.")
                continue
        else:
            # 如果已经确定扩展名，直接使用
            mask_path = os.path.join(mask_folder, f"{cam_info.image_name}{determined_extension}")
            if not os.path.exists(mask_path):
                print(f"未找到 {cam_info.image_name} 的 mask 文件，预期扩展名为 {determined_extension}.")
                continue

        # 读取图像并转换为灰度图像
        mask = Image.open(mask_path).convert("L")
        masks.append(np.array(mask))

    # Assume all masks have the same scale relative to original images
    scale = masks[0].shape[1] / cam_infos[0].width
    scale = round(scale * 4) * 0.25
    print("scale", scale)

    if eval:
        train_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % llffhold != 0]
        test_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % llffhold == 0]
    else:
        train_cam_infos = cam_infos
        test_cam_infos = []

    nerf_normalization = getNerfppNorm(train_cam_infos)

    ply_path = os.path.join(path, "sparse/points3D.ply")
    bin_path = os.path.join(path, "sparse/points3D.bin")
    txt_path = os.path.join(path, "sparse/points3D.txt")

    if os.path.exists(ply_path):
        print(f"{ply_path} already exists. Skipping point cloud processing.")
        print(f"Reading point cloud from {bin_path}")
        """try:
            xyz, rgb, _ = read_points3D_binary(bin_path)
        except:
            xyz, rgb, _ = read_points3D_text(txt_path)"""
        plydata = PlyData.read(ply_path)
        vertices = plydata['vertex']
        xyz = np.vstack([vertices['x'], vertices['y'], vertices['z']]).T
        rgb = np.vstack([vertices['red'], vertices['green'], vertices['blue']]).T / 255.0

        print(f"Applying masks to point cloud")
        start_time = time.time()
        filtered_points, filtered_colors = apply_mask_to_point_cloud(xyz, rgb, masks, cam_infos, scale, path)
        end_time = time.time()
        print("the time of filter points is ", end_time - start_time)

        if len(filtered_points) == 0:
            print("Warning: No points left after applying masks")

        print(f"Filtered points: {filtered_points.shape}, Filtered colors: {filtered_colors.shape}")

        # Store filtered point cloud to PLY
        storePly(ply_path, filtered_points, filtered_colors)
    else:
        print(f"Reading point cloud from {bin_path}")
        try:
            xyz, rgb, _ = read_points3D_binary(bin_path)
        except:
            xyz, rgb, _ = read_points3D_text(txt_path)

        print(f"Applying masks to point cloud")
        start_time = time.time()
        filtered_points, filtered_colors = apply_mask_to_point_cloud(xyz, rgb, masks, cam_infos, scale, path)
        end_time = time.time()
        print("the time of filter points is ", end_time - start_time)

        if len(filtered_points) == 0:
            print("Warning: No points left after applying masks")

        print(f"Filtered points: {filtered_points.shape}, Filtered colors: {filtered_colors.shape}")

        # Store filtered point cloud to PLY
        storePly(ply_path, filtered_points, filtered_colors)

    try:
        pcd = fetchPly(ply_path)
    except Exception as e:
        print(f"Error fetching PLY file: {e}")
        pcd = None

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path)
    return scene_info


def readSingleSceneInfo(path, images, eval, llffhold=5):
    try:
        cameras_pkl_file = os.path.join(path, "cam_info_list.pkl")
        points_file = os.path.join(path, "point3D.ply")
    except:
        cameras_pkl_file = os.path.join(path, "cam_info_list.txt")
        points_file = os.path.join(path, "point3D.txt")

    with open(cameras_pkl_file, 'rb') as file:
        cam_infos = pickle.load(file)
    # 补全cam_infos缺少的image和merge_mask
    for i, cam_info in enumerate(cam_infos):
        image_path = os.path.join(path, "images", cam_info.image_name + ".JPG")
        gt_image = Image.open(image_path)
        zero_mask_merge = np.zeros((gt_image.height, gt_image.width), dtype=np.int8)  # 在这里暂时不使用
        # 使用 _replace 返回新的对象，并将其更新到 cam_infos 列表中
        cam_infos[i] = cam_info

    if eval:
        train_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % llffhold != 0]
        test_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % llffhold == 0]
    else:
        train_cam_infos = cam_infos
        test_cam_infos = []

    nerf_normalization = getNerfppNorm(train_cam_infos)
    pcd = fetchPly(points_file)

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=points_file)
    return scene_info


def readCamerasFromTransforms(path, transformsfile, white_background, extension=".png"):
    cam_infos = []

    with open(os.path.join(path, transformsfile)) as json_file:
        contents = json.load(json_file)
        fovx = contents["camera_angle_x"]

        frames = contents["frames"]
        for idx, frame in enumerate(frames):
            cam_name = os.path.join(path, frame["file_path"] + extension)

            # NeRF 'transform_matrix' is a camera-to-world transform
            c2w = np.array(frame["transform_matrix"])
            # change from OpenGL/Blender camera axes (Y up, Z back) to COLMAP (Y down, Z forward)
            c2w[:3, 1:3] *= -1

            # get the world-to-camera transform and set R, T
            w2c = np.linalg.inv(c2w)
            R = np.transpose(w2c[:3, :3])  # R is stored transposed due to 'glm' in CUDA code
            T = w2c[:3, 3]

            image_path = os.path.join(path, cam_name)
            image_name = Path(cam_name).stem
            image = Image.open(image_path)

            im_data = np.array(image.convert("RGBA"))

            bg = np.array([1, 1, 1]) if white_background else np.array([0, 0, 0])

            norm_data = im_data / 255.0
            arr = norm_data[:, :, :3] * norm_data[:, :, 3:4] + bg * (1 - norm_data[:, :, 3:4])
            image = Image.fromarray(np.array(arr * 255.0, dtype=np.byte), "RGB")

            fovy = focal2fov(fov2focal(fovx, image.size[0]), image.size[1])
            FovY = fovy
            FovX = fovx

            cam_infos.append(CameraInfo(uid=idx, global_id=idx, R=R, T=T, FovY=FovY, FovX=FovX, image=image,
                                        image_path=image_path, image_name=image_name, width=image.size[0],
                                        height=image.size[1]))

    return cam_infos


def readNerfSyntheticInfo(path, white_background, eval, extension=".png"):
    print("Reading Training Transforms")
    train_cam_infos = readCamerasFromTransforms(path, "transforms_train.json", white_background, extension)
    print("Reading Test Transforms")
    test_cam_infos = readCamerasFromTransforms(path, "transforms_test.json", white_background, extension)

    if not eval:
        train_cam_infos.extend(test_cam_infos)
        test_cam_infos = []

    nerf_normalization = getNerfppNorm(train_cam_infos)

    ply_path = os.path.join(path, "points3d.ply")
    if not os.path.exists(ply_path):
        # Since this data set has no colmap data, we start with random points
        num_pts = 100_000
        print(f"Generating random point cloud ({num_pts})...")

        # We create random points inside the bounds of the synthetic Blender scenes
        xyz = np.random.random((num_pts, 3)) * 2.6 - 1.3
        shs = np.random.random((num_pts, 3)) / 255.0
        pcd = BasicPointCloud(points=xyz, colors=SH2RGB(shs), normals=np.zeros((num_pts, 3)))

        storePly(ply_path, xyz, SH2RGB(shs) * 255)
    try:
        pcd = fetchPly(ply_path)
    except:
        pcd = None

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path)
    return scene_info


sceneLoadTypeCallbacks = {
    "Single": readSingleSceneInfo,
    "Colmap": readColmapSceneInfo,
    "Blender": readNerfSyntheticInfo
}