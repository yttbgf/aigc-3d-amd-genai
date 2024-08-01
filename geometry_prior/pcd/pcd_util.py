import os
import math
import numpy as np
import open3d as o3d


def save_pcd(basic_pcd, path="./basic_pcd.pcd"):
    os.makedirs(os.path.dirname(path), exist_ok=True)

    pcd_o3d = o3d.geometry.PointCloud()
    pcd_o3d.points = o3d.utility.Vector3dVector(basic_pcd.points)

    # 如果有颜色和法线数据，也可以添加到 PointCloud 对象中
    if basic_pcd.colors is not None:
        pcd_o3d.colors = o3d.utility.Vector3dVector(basic_pcd.colors)

    if basic_pcd.normals is not None:
        pcd_o3d.normals = o3d.utility.Vector3dVector(basic_pcd.normals)

    # 保存 PointCloud 对象为 PCD 文件
    o3d.io.write_point_cloud(path, pcd_o3d)


def load_pcd(path="./basic_pcd.pcd"):
    # 读取 PCD 文件
    pcd_o3d = o3d.io.read_point_cloud(path)

    # 将 open3d PointCloud 对象转换为 BasicPointCloud 实例
    def convert_to_basic_point_cloud(pcd_o3d):
        from gs_renderer import BasicPointCloud
        points = np.asarray(pcd_o3d.points)
        colors = np.asarray(pcd_o3d.colors) if pcd_o3d.has_colors() else None
        normals = np.asarray(pcd_o3d.normals) if pcd_o3d.has_normals() else None
        return BasicPointCloud(points, colors, normals)

    # 加载 PCD 文件并转换为 BasicPointCloud 实例
    basic_pcd = convert_to_basic_point_cloud(pcd_o3d)

    # 打印结果
    '''
    print("Points:", basic_pcd.points)
    if basic_pcd.colors is not None:
        print("Colors:", basic_pcd.colors)
    if basic_pcd.normals is not None:
        print("Normals:", basic_pcd.normals)
    '''
    return basic_pcd
