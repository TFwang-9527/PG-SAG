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
        c2w[:3,3] = c2w[:3,3] * 10.0
        w2c = np.linalg.inv(c2w)
        w2c = w2c
        poses.append(w2c)
    poses = np.stack(poses, axis=0)
    return poses
from sklearn.decomposition import PCA

def readColmapCameras(cam_extrinsics, cam_intrinsics, images_folder):
    cam_infos = []
    for idx, key in enumerate(cam_extrinsics):
        sys.stdout.write('\r')
        # the exact output you're looking for:
        sys.stdout.write("Reading camera {}/{}".format(idx+1, len(cam_extrinsics)))
        sys.stdout.flush()

        extr = cam_extrinsics[key]
        intr = cam_intrinsics[extr.camera_id]
        height = intr.height
        width = intr.width

        uid = intr.id
        R = np.transpose(qvec2rotmat(extr.qvec))
        T = np.array(extr.tvec)

        if intr.model=="SIMPLE_PINHOLE":
            focal_length_x = intr.params[0]
            FovY = focal2fov(focal_length_x, height)
            FovX = focal2fov(focal_length_x, width)
        elif intr.model=="PINHOLE":
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

import torch
import numpy as np
def fov2focal(fov, dim, device='cuda'):
    fov_tensor = torch.tensor(fov, dtype=torch.float16, device=device)
    dim_tensor = torch.tensor(dim, dtype=torch.float16, device=device)
    return dim_tensor / (2 * torch.tan(fov_tensor / 2))


def world_to_camera_point(points, R, T):
    R = torch.tensor(R, dtype=torch.float16, device=points.device)
    T = torch.tensor(T, dtype=torch.float16, device=points.device).view(3, 1)
    points = points.clone().detach()
    points = points.float()
    R = R.float()
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

    u = u * scale
    v = v * scale

    u[~valid] = -1
    v[~valid] = -1

    return u, v

def get_z_axis_rotation(cam_info):
    R = cam_info.R
    camera_z_axis = R[:, 2]
    world_z_axis = np.array([0, 0, 1])
    cos_theta_z = np.dot(camera_z_axis, world_z_axis) / (np.linalg.norm(camera_z_axis) * np.linalg.norm(world_z_axis))
    theta_z_rad = np.arccos(cos_theta_z)
    return np.degrees(theta_z_rad)

def apply_mask_to_point_cloud(points, colors, masks, camera_infos, scale, path):

    points_tensor = torch.tensor(points, dtype=torch.float, device='cuda')
    labels_chunk = torch.zeros(points_tensor.shape[0], dtype=torch.uint8, device='cuda')
    false_proj_counts = torch.zeros(points_tensor.shape[0], dtype=torch.uint8, device='cuda')
    masks_cuda = [torch.tensor(mask, dtype=torch.uint8, device='cuda') for mask in masks]

    for cam_info, mask_tensor in zip(camera_infos, masks_cuda):
        u, v = project_to_image(points_tensor, cam_info, scale)
        u = u.long()
        v = v.long()
        valid_mask = (u >= 0) & (u < mask_tensor.shape[1]) & (v >= 0) & (v < mask_tensor.shape[0])
        valid_indices = valid_mask.nonzero(as_tuple=True)
        u_valid = u[valid_indices]
        v_valid = v[valid_indices]
        valid_labels = mask_tensor[v_valid, u_valid]
        positive_mask = valid_labels > 0
        positive_indices = tuple(idx[positive_mask] for idx in valid_indices)
        labels_chunk[positive_indices] = valid_labels[positive_mask]
        false_proj = valid_labels < 128
        false_proj_counts[valid_indices] += false_proj.long()
    torch.cuda.empty_cache()

    false_tolerance = 20
    valid_points = (labels_chunk > 128) & (false_proj_counts <= false_tolerance)
    points_tensor = points_tensor[valid_points]
    colors = np.array(colors)[valid_points.cpu().numpy()]
    points = points_tensor.cpu().numpy()
    projected_points = points[:, [0, 1]]
    eps = 0.1
    min_samples = 100
    clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(projected_points)

    from concurrent.futures import ThreadPoolExecutor
    points = points_tensor.cpu().numpy()
    labels = torch.tensor(clustering.labels_, dtype=torch.int16, device='cuda') + 3
    path_single = path+"_multi"
    unique_labels = set(labels.cpu().numpy())
    cam_info_lists = {label: [] for label in unique_labels if label != 0}
    for label in unique_labels:
        if label == 0:
            continue
        os.makedirs(os.path.join(path_single, f"label_{label}", "mask_points"), exist_ok=True)
        os.makedirs(os.path.join(path_single, f"label_{label}", "images"), exist_ok=True)

    def save_image_and_data(label_id, cam_info, mask_merge_processed, gt_image_target_path):
        mask_image_path = os.path.join(path_single, f"label_{label_id}", "mask_points", f"{cam_info.image_name}.JPG")
        Image.fromarray(mask_merge_processed).save(mask_image_path)
        shutil.copy(cam_info.image_path, gt_image_target_path)

    with ThreadPoolExecutor(max_workers=40) as executor:
        futures = []
        for cam_idx, cam_info in enumerate(camera_infos):
            mask_merge = torch.zeros_like(masks_cuda[0], dtype=torch.int16, device='cuda')
            u, v = project_to_image(points_tensor, cam_info, scale)
            u, v = u.long(), v.long()
            valid_mask = (u >= 0) & (u < mask_merge.shape[1]) & (v >= 0) & (v < mask_merge.shape[0])
            valid_u, valid_v = u[valid_mask], v[valid_mask]
            valid_labels = labels[valid_mask]
            mask_merge[valid_v, valid_u] = valid_labels
            current_labels = torch.unique(valid_labels).cpu().numpy()
            for label_id in current_labels:
                if label_id == 0:
                    continue
                mask_numpy = mask_merge.cpu().numpy()
                if np.sum(mask_numpy == label_id) > 100:
                    mask_merge_processed = np.where(mask_numpy == label_id, 255, 0).astype(np.uint8)
                    gt_image_target_path = os.path.join(path_single, f"label_{label_id}", "images",
                                                            f"{cam_info.image_name}.JPG")
                    futures.append(
                        executor.submit(save_image_and_data, label_id, cam_info, mask_merge_processed,
                                            gt_image_target_path))
                    cam_info_copy = cam_info._replace(image_path=gt_image_target_path)
                    cam_info_lists[label_id].append(cam_info_copy)
        for future in futures:
            future.result()

    for label_id, cam_info_list in cam_info_lists.items():
        label_folder = os.path.join(path_single, f"label_{label_id}")
        cam_info_pkl_path = os.path.join(label_folder, 'cam_info_list.pkl')
        with open(cam_info_pkl_path, 'wb') as f:
            pickle.dump(cam_info_list, f)
    for label in unique_labels:
        if label == 0:
            continue
        cluster_points = points[labels.cpu().numpy() == label]
        cluster_colors = np.ones((cluster_points.shape[0], 3), dtype=np.uint8) * 255
        output_path = os.path.join(path_single, f"label_{label}", "point3D.ply")
        storePly(output_path, cluster_points, cluster_colors)
    return points_tensor.cpu().numpy(), colors


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
    determined_extension = None

    for cam_info in cam_infos:
        if determined_extension is None:
            possible_extensions = [".png", ".jpg", ".JPG"]
            for ext in possible_extensions:
                temp_path = os.path.join(mask_folder, f"{cam_info.image_name}{ext}")
                if os.path.exists(temp_path):
                    determined_extension = ext
                    mask_path = temp_path
                    break
            if determined_extension is None:
                print(f"No valid mask file found for {cam_info.image_name}.")
                continue
        else:
            mask_path = os.path.join(mask_folder, f"{cam_info.image_name}{determined_extension}")
            if not os.path.exists(mask_path):
                print(f"No mask file found for {cam_info.image_name}, expected extension: {determined_extension}.")
                continue

        # 读取图像并转换为灰度图像
        mask = Image.open(mask_path).convert("L")
        masks.append(np.array(mask))

    # Assume all masks have the same scale relative to original images
    a = masks[0].shape[1]
    b = cam_infos[0].width
    scale = masks[0].shape[1] / cam_infos[0].width
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
        plydata = PlyData.read(ply_path)
        vertices = plydata['vertex']
        xyz = np.vstack([vertices['x'], vertices['y'], vertices['z']]).T
        rgb = np.vstack([vertices['red'], vertices['green'], vertices['blue']]).T / 255.0
        filtered_points, filtered_colors = apply_mask_to_point_cloud(xyz, rgb, masks, cam_infos, scale, path)
        if len(filtered_points) == 0:
            print("Warning: No points left after applying masks")
        storePly(ply_path, filtered_points, filtered_colors)
    else:
        print(f"Reading point cloud from {bin_path}")
        try:
            xyz, rgb, _ = read_points3D_binary(bin_path)
        except:
            xyz, rgb, _ = read_points3D_text(txt_path)
        filtered_points, filtered_colors = apply_mask_to_point_cloud(xyz, rgb, masks, cam_infos, scale, path)
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
    for i, cam_info in enumerate(cam_infos):
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
            R = np.transpose(w2c[:3,:3])  # R is stored transposed due to 'glm' in CUDA code
            T = w2c[:3, 3]

            image_path = os.path.join(path, cam_name)
            image_name = Path(cam_name).stem
            image = Image.open(image_path)

            im_data = np.array(image.convert("RGBA"))

            bg = np.array([1,1,1]) if white_background else np.array([0, 0, 0])

            norm_data = im_data / 255.0
            arr = norm_data[:,:,:3] * norm_data[:, :, 3:4] + bg * (1 - norm_data[:, :, 3:4])
            image = Image.fromarray(np.array(arr*255.0, dtype=np.byte), "RGB")

            fovy = focal2fov(fov2focal(fovx, image.size[0]), image.size[1])
            FovY = fovy 
            FovX = fovx

            cam_infos.append(CameraInfo(uid=idx, global_id=idx, R=R, T=T, FovY=FovY, FovX=FovX, image=image,
                            image_path=image_path, image_name=image_name, width=image.size[0], height=image.size[1]))
            
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
    "Blender" : readNerfSyntheticInfo
}
