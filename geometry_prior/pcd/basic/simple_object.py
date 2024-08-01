import open3d as o3d
import numpy as np
print(o3d.core.cuda.is_available())


def create_sphere_pcd(radius=1.0, resolution=100):
    """
    创建一个球体的点云数据(PCD)。
    :param radius: 球体的半径
    :param resolution: 球体的分辨率，即分割的精度
    :return: open3d.geometry.PointCloud
    """
    # 创建一个空的点云对象
    sphere_pcd = o3d.geometry.PointCloud()
    sphere_pcd_points = []
    # 根据球体的参数生成点云数据
    for i in range(resolution):
        for j in range(resolution):
            theta = np.pi * i / resolution
            phi = 2 * np.pi * j / resolution
            x = radius * np.cos(phi) * np.sin(theta)
            y = radius * np.sin(phi) * np.sin(theta)
            z = radius * np.cos(theta)
            point = [x, y, z]
            # 将点添加到点云中
            sphere_pcd_points.append(point)

    # From numpy to Open3D
    sphere_pcd.points = o3d.utility.Vector3dVector(sphere_pcd_points)

    # From Open3D to numpy
    #sphere_pcd_points = np.asarray(sphere_pcd.points)

    return sphere_pcd


def create_cylinder_pcd(radius, height, resolution):
    """
    创建一个圆柱体的点云数据(PCD)。
    :param radius: 圆柱体的半径
    :param height: 圆柱体的高度
    :param resolution: 圆柱体的分辨率，即分割的精度
    :return: open3d.geometry.PointCloud
    """
    # 创建一个空的点云对象
    cylinder_pcd = o3d.geometry.PointCloud()
    cylinder_pcd_points = []

    # 根据圆柱体的参数生成点云数据
    for i in range(resolution):
        theta = 2 * np.pi * i / resolution
        for j in range(2):  # 圆柱体上下两个圆面
            for k in range(resolution):
                if j == 0:  # 底面
                    z = 0
                else:  # 顶面
                    z = height
                x = radius * np.cos(theta)
                y = radius * np.sin(theta)

                # 将点添加到点云中
                cylinder_pcd_points.append([x, y, z])

    # 转换点云列表为numpy数组
    cylinder_pcd.points = o3d.utility.Vector3dVector(cylinder_pcd_points)

    return cylinder_pcd


def create_box_pcd(width, height, depth, resolution):
    """
    创建一个长方体的点云数据(PCD)。
    :param width: 长方体的宽度
    :param height: 长方体的高度
    :param depth: 长方体的深度
    :param resolution: 长方体的分辨率，即分割的精度
    :return: open3d.geometry.PointCloud
    """
    # 创建一个空的点云对象
    box_pcd = o3d.geometry.PointCloud()
    box_pcd_points = []
    # 长方体的8个角点
    points = [
        [0, 0, 0],  # 点1
        [width, 0, 0],  # 点2
        [width, height, 0],  # 点3
        [0, height, 0],  # 点4
        [0, 0, depth],  # 点5
        [width, 0, depth],  # 点6
        [width, height, depth],  # 点7
        [0, height, depth]  # 点8
    ]

    # 将点添加到点云中
    for point in points:
        box_pcd_points.append(point)

    # 根据分辨率细分点云
    if resolution > 1:
        # 细分长方体的每个面
        # 这里只是一个示意，实际的细分可能需要更复杂的算法
        for i in range(resolution - 1):
            for j in range(resolution - 1):
                # 计算细分点的坐标
                u = i / (resolution - 1)
                v = j / (resolution - 1)
                # 这里只是示意，需要根据实际情况计算细分点
                new_point = [u * width, v * height, depth]  # 以深度面为例
                box_pcd_points.append(new_point)

    # 转换点云列表为numpy数组
    box_pcd.points = o3d.utility.Vector3dVector(box_pcd_points)
    return box_pcd


def create_cube_pcd(side_length, resolution):
    """
    创建一个正方体的点云数据(PCD)。
    :param side_length: 正方体的边长
    :param resolution: 正方体的分辨率，即每个边等分的点数
    :return: open3d.geometry.PointCloud
    """
    # 创建一个空的点云对象
    cube_pcd = o3d.geometry.PointCloud()
    cube_pcd_points = []
    # 正方体的8个角点
    points = [
        [0, 0, 0],  # 点1
        [side_length, 0, 0],  # 点2
        [side_length, side_length, 0],  # 点3
        [0, side_length, 0],  # 点4
        [0, 0, side_length],  # 点5
        [side_length, 0, side_length],  # 点6
        [side_length, side_length, side_length],  # 点7
        [0, side_length, side_length]  # 点8
    ]

    # 将点添加到点云中
    for point in points:
        cube_pcd_points.append(point)

    # 根据分辨率细分正方体的每个面
    for i in range(resolution + 1):
        for j in range(resolution + 1):
            # 计算细分点的坐标
            u = i / resolution
            v = j / resolution
            for k in range(2):  # 正方体的每个面
                if k == 0:  # x-y平面
                    z = 0
                else:  # x-z平面
                    y = 0
                # 计算细分点的x和y坐标
                x = u * side_length
                if k == 1:  # x-z平面
                    y = v * side_length
                else:  # x-y平面
                    y = side_length if u == resolution else (u + 1) * side_length
                # 将细分点添加到点云中
                cube_pcd_points.append([x, y, z])

    # 转换点云列表为numpy数组
    cube_pcd.points = o3d.utility.Vector3dVector(cube_pcd_points)
    return cube_pcd


def create_cone_pcd(radius, height, resolution):
    """
    创建一个圆锥体的点云数据(PCD)。
    :param radius: 圆锥体底面半径
    :param height: 圆锥体高度
    :param resolution: 圆锥体的分辨率，即底面圆周等分的点数
    :return: open3d.geometry.PointCloud
    """
    # 创建一个空的点云对象
    cone_pcd = o3d.geometry.PointCloud()
    cone_pcd_points = []
    # 圆锥体底面圆周上的点
    for i in range(resolution):
        theta = 2 * np.pi * i / resolution
        x = radius * np.cos(theta)
        y = radius * np.sin(theta)
        z = 0  # 底面在z=0平面
        cone_pcd_points.append([x, y, z])

    # 圆锥体侧面的点
    for i in range(1, resolution):
        theta = 2 * np.pi * i / resolution
        for t in np.linspace(0, 1, num=resolution):
            x = (1 - t) * radius * np.cos(theta)
            y = (1 - t) * radius * np.sin(theta)
            z = t * height
            cone_pcd_points.append([x, y, z])

    # 圆锥体顶点
    cone_pcd_points.append([0, 0, height])
    # TODO: 圆锥体底面

    # 转换点云列表为numpy数组
    cone_pcd.points = o3d.utility.Vector3dVector(cone_pcd_points)
    return cone_pcd


def create_frustum_pcd(bottom_radius, top_radius, height, resolution):
    """
    创建一个圆台的点云数据(PCD)。
    :param bottom_radius: 圆台底面半径
    :param top_radius: 圆台顶面半径
    :param height: 圆台高度
    :param resolution: 圆台的分辨率，即底面和顶面圆周等分的点数
    :return: open3d.geometry.PointCloud
    """
    # 创建一个空的点云对象
    frustum_pcd = o3d.geometry.PointCloud()
    frustum_pcd_points = []
    # 圆台底面圆周上的点
    for i in range(resolution):
        theta = 2 * np.pi * i / resolution
        x = bottom_radius * np.cos(theta)
        y = bottom_radius * np.sin(theta)
        z = 0  # 底面在z=0平面
        frustum_pcd_points.append([x, y, z])

    # 圆台顶面圆周上的点
    for i in range(resolution):
        theta = 2 * np.pi * i / resolution
        x = top_radius * np.cos(theta)
        y = top_radius * np.sin(theta)
        z = height  # 顶面在z=height平面
        frustum_pcd_points.append([x, y, z])

    # 圆台侧面的点
    for i in range(resolution):
        theta = 2 * np.pi * i / resolution
        for t in np.linspace(0, 1, num=resolution):
            x = (1 - t) * bottom_radius * np.cos(theta) + t * top_radius * np.cos(theta)
            y = (1 - t) * bottom_radius * np.sin(theta) + t * top_radius * np.sin(theta)
            z = t * height
            frustum_pcd_points.append([x, y, z])

    # 转换点云列表为numpy数组
    frustum_pcd.points = o3d.utility.Vector3dVector(frustum_pcd_points)
    return frustum_pcd


if __name__ == "__main__":
    from gs_renderer import BasicPointCloud
    from geometry_prior.pcd.pcd_util import save_pcd, load_pcd

    #basic_pcd = load_pcd()
    basic_pcd = load_pcd("../../../basic_pcd_raw.pcd")
    basic_pcd = load_pcd("../../../basic_pcd_self.pcd")
    exit(0)
    # 使用默认参数创建球体点云
    sphere_pcd = create_sphere_pcd()
    # 可视化球体点云
    o3d.visualization.draw_geometries([sphere_pcd])

    # 使用默认参数创建圆柱体点云
    #cylinder_pcd = create_cylinder_pcd(radius=1.0, height=2.0, resolution=100)
    # 可视化圆柱体点云
    #o3d.visualization.draw_geometries([cylinder_pcd])

    # 使用默认参数创建圆锥体点云
    #cone_pcd = create_cone_pcd(radius=1.0, height=2.0, resolution=50)
    # 可视化圆锥体点云
    #o3d.visualization.draw_geometries([cone_pcd])

    # 使用默认参数创建圆台点云
    #frustum_pcd = create_frustum_pcd(bottom_radius=1.0, top_radius=0.5, height=2.0, resolution=50)
    # 可视化圆台点云
    #o3d.visualization.draw_geometries([frustum_pcd])

    # 使用默认参数创建长方体点云
    #box_pcd = create_box_pcd(width=2.0, height=1.5, depth=3.0, resolution=20)
    # 可视化长方体点云
    #o3d.visualization.draw_geometries([box_pcd])
    #exit(0)
    # 使用默认参数创建正方体点云
    #cube_pcd = create_cube_pcd(side_length=1.0, resolution=10)
    # 可视化正方体点云
    #o3d.visualization.draw_geometries([cube_pcd])

    # 假设你已经有了一个 BasicPointCloud 实例
    # 这里我们创建一个示例实例
    n = 10
    n = len(sphere_pcd.points)
    print(n)
    #points = np.random.rand(n, 3)  # n个随机点，每个点3个坐标
    colors = np.random.rand(n, 3)  # n个随机颜色，每个颜色3个通道
    normals = np.random.rand(n, 3)  # n个随机法线

    basic_pcd = BasicPointCloud(sphere_pcd.points, colors, normals)

    # 将 BasicPointCloud 实例转换为 open3d 的 PointCloud 对象
    save_pcd(basic_pcd)
    basic_pcd = load_pcd()
