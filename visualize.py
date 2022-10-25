import argparse
from pathlib import Path

import open3d as o3d
import numpy as np
from scipy.spatial.transform import Rotation as R

from carla_data_utils import load_raw_data_infos


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--id", type=int, default=0, required=False)
    args = parser.parse_args()

    root_path = Path("data/carla_raw")
    # 加载原始数据
    data_infos = load_raw_data_infos(root_path, args.id)
    data_info = data_infos[0]

    print("scene_id", data_info.scene_id)

    points_list = []
    for lidar_info in data_info.lidars:
        lidar_info.pc_path

        pc = o3d.io.read_point_cloud(
            str(root_path / lidar_info.pc_path)
        )
        points = np.asarray(pc.points).astype(np.float32)

        # 统一到世界坐标系
        pc.rotate(lidar_info.sensor_rot, center=[0., 0., 0.])
        pc.translate(lidar_info.sensor_trans)

        points = np.asarray(pc.points).astype(np.float32)
        points_list.append(points)

    # 融合四个lidar的点云
    points = np.concatenate(points_list, axis=0)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)

    bboxes = []
    for vehicle in data_info.vehicles:
        bbox = vehicle.get_bbox()

        # 汽车底部中心的的坐标
        bottom_center = bbox[0:3]
        # 汽车的大小（已经在get_bbox里乘2，这里不需要乘2）
        extent = bbox[3:6]
        yaw = bbox[6]

        # 汽车中心点坐标，z轴加上高度的一半
        center = bottom_center[:2] + [bottom_center[2] + extent[2] / 2.]

        # 获取旋转矩阵
        r = R.from_euler("z", yaw).as_matrix()
        bbox = o3d.geometry.OrientedBoundingBox(
            center=center,
            R=r,
            extent=extent,
        )
        bbox.color = [1, 0, 0]
        bboxes.append(bbox)

    # 将所有元素，从视觉坐标系转换到lidar0坐标系下
    world_to_lidar0_rot = np.linalg.inv(data_info.lidars[0].sensor_rot)
    world_to_lidar0_trans = -data_info.lidars[0].sensor_trans
    for geom in [pcd] + bboxes:
        geom.translate(world_to_lidar0_trans)
        geom.rotate(world_to_lidar0_rot, center=[0., 0., 0.])

    # 左手坐标系到右手坐标系
    geoms = []
    for geom in [pcd] + bboxes:
        if isinstance(geom, o3d.geometry.OrientedBoundingBox):
            center = geom.center.copy()
            # y -> -y
            center[1] *= -1
            bbox = o3d.geometry.OrientedBoundingBox(
                center=center,
                R=geom.R,
                extent=geom.extent,
            )
            bbox.color = [1, 0, 0]
            geoms.append(bbox)
        elif isinstance(geom, o3d.geometry.PointCloud):
            points = np.asarray(geom.points).astype(np.float32)
            # y -> -y
            points[:, 1] *= -1
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(points)
            geoms.append(pcd)
        else:
            raise NotImplementedError(geom)

    o3d.visualization.draw(geoms)


if __name__ == "__main__":
    main()
