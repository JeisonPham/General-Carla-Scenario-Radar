import matplotlib.pyplot as plt
import json
import numpy as np
import os
import open3d as o3d
import mayavi.mlab as mlab
import ply

def draw_lidar(pc,
               color=None,
               fig=None,
               bgcolor=(0, 0, 0),
               pts_scale=0.3,
               pts_mode='sphere',
               pts_color=None,
               color_by_intensity=False,
               pc_label=False):
    ''' Draw lidar points
    Args:
        pc: numpy array (n,3) of XYZ
        color: numpy array (n) of intensity or whatever
        fig: mayavi figure handler, if None create new one otherwise will use it
    Returns:
        fig: created or used fig
    '''
    pts_mode = 'point'
    if fig is None:
        fig = mlab.figure(figure=None, bgcolor=bgcolor,
                          fgcolor=None, engine=None, size=(1600, 1000))
    if color is None:
        color = pc[:, 2]
    if pc_label:
        color = pc[:, 4]
    if color_by_intensity:
        color = pc[:, 2]

    x = pc[:, 0]
    y = pc[:, 1]
    z = pc[:, 2]
    d = pc[:, -1]
    mlab.points3d(x, y, z, d,
                  mode="point", colormap='spectral', scale_factor=pts_scale, figure=fig)

    return fig

def draw_gt_boxes3d(gt_boxes3d, fig, color=(1, 1, 1), line_width=1, draw_text=False, text_scale=(1, 1, 1), color_list=None):
    ''' Draw 3D bounding boxes
    Args:
        gt_boxes3d: numpy array (n,8,3) for XYZs of the box corners
        fig: mayavi figure handler
        color: RGB value tuple in range (0,1), box line color
        line_width: box line width
        draw_text: boolean, if true, write box indices beside boxes
        text_scale: three number tuple
        color_list: a list of RGB tuple, if not None, overwrite color.
    Returns:
        fig: updated fig
    '''
    num = len(gt_boxes3d)
    for n in range(num):
        b = gt_boxes3d[n]
        if color_list is not None:
            color = color_list[n]
        if draw_text:
            mlab.text3d(b[4, 0], b[4, 1], b[4, 2], '%d' %
                        n, scale=text_scale, color=color, figure=fig)
        # colors = [(0, 0.5, 0.5), (0, 0, 1), (0, 1, 0), (0, 1, 1), (1, 0, 0), (1, 0, 1), (1, 1, 0), (1, 1, 1)]
        # print(b)
        # for i, point in enumerate(b):
        #     x, y, z = point
        #     mlab.points3d(x, y, z, color=colors[i], scale_factor=1, figure=fig)
        mlab.plot3d([b[0, 0], b[1, 0]], [b[0, 1], b[1, 1]], [b[0, 2], b[1, 2]],
                    color=color, tube_radius=None, line_width=line_width, figure=fig)

        mlab.plot3d([b[1, 0], b[3, 0]], [b[1, 1], b[3, 1]], [b[1, 2], b[3, 2]],
                    color=color, tube_radius=None, line_width=line_width, figure=fig)

        mlab.plot3d([b[2, 0], b[3, 0]], [b[2, 1], b[3, 1]], [b[2, 2], b[3, 2]],
                    color=color, tube_radius=None, line_width=line_width, figure=fig)

        mlab.plot3d([b[0, 0], b[2, 0]], [b[0, 1], b[2, 1]], [b[0, 2], b[2, 2]],
                    color=color, tube_radius=None, line_width=line_width, figure=fig)



        mlab.plot3d([b[4, 0], b[5, 0]], [b[4, 1], b[5, 1]], [b[4, 2], b[5, 2]],
                    color=color, tube_radius=None, line_width=line_width, figure=fig)

        mlab.plot3d([b[4, 0], b[6, 0]], [b[4, 1], b[6, 1]], [b[4, 2], b[6, 2]],
                    color=color, tube_radius=None, line_width=line_width, figure=fig)

        mlab.plot3d([b[5, 0], b[7, 0]], [b[5, 1], b[7, 1]], [b[5, 2], b[7, 2]],
                    color=color, tube_radius=None, line_width=line_width, figure=fig)

        mlab.plot3d([b[6, 0], b[7, 0]], [b[6, 1], b[7, 1]], [b[6, 2], b[7, 2]],
                    color=color, tube_radius=None, line_width=line_width, figure=fig)

        mlab.plot3d([b[0, 0], b[4, 0]], [b[0, 1], b[4, 1]], [b[0, 2], b[4, 2]],
                    color=color, tube_radius=None, line_width=line_width, figure=fig)

        mlab.plot3d([b[1, 0], b[5, 0]], [b[1, 1], b[5, 1]], [b[1, 2], b[5, 2]],
                    color=color, tube_radius=None, line_width=line_width, figure=fig)

        mlab.plot3d([b[2, 0], b[6, 0]], [b[2, 1], b[6, 1]], [b[2, 2], b[6, 2]],
                    color=color, tube_radius=None, line_width=line_width, figure=fig)

        mlab.plot3d([b[3, 0], b[7, 0]], [b[3, 1], b[7, 1]], [b[3, 2], b[7, 2]],
                    color=color, tube_radius=None, line_width=line_width, figure=fig)
    # mlab.show(1)
    #mlab.view(azimuth=180, elevation=70, focalpoint=[ 12.0909996 , -1.04700089, -2.03249991], distance=62.0, figure=fig)
    return fig
def yaw(r):
    return np.array([[np.cos(r), -np.sin(r), 0],
                     [np.sin(r), np.cos(r), 0],
                     [0, 0, 1]])


root_folder = "data/nss_0702/TEST_0_07_10_15_02_45"

timestamp = 18
#
# bounding_box = np.load(os.path.join(root_folder, "bb3d", f"{timestamp:04d}.npy"), allow_pickle=True).item()
# vehicles = np.array(bounding_box['vehicles'])[:, 0, :]
#
# with open(os.path.join(root_folder, "measurements", f"{timestamp:04d}.json"), 'r') as file:
#     measurement = json.load(file)
# transform = np.array(measurement['transform'])
# lidar_front = np.load(os.path.join(root_folder, "semantic_lidar_front", f"{timestamp:04d}.npy"), allow_pickle=True)
# lidar_back = np.load(os.path.join(root_folder, "semantic_lidar_back", f"{timestamp:04d}.npy"), allow_pickle=True)
lidar = np.load(os.path.join(root_folder, "semantic_lidar", f"{timestamp:04d}.npy"), allow_pickle=True)
# lidar_back[:, 2] *= -1
# lidar_back[:, -1] += 10

# lidar = np.concatenate([lidar_front, lidar_back], axis=0)
# lidar = np.load('/home/jeison/Desktop/mmfn/data/nss_0702/TEST_0_06_20_20_26_32/frames/frame_0170.ply.npy')
# lidar = np.hstack([lidar, np.ones((lidar.shape[0], 1))]).T
# lidar = np.linalg.inv(transform) @ lidar
# lidar = lidar.T[:, :3]

# figure = plt.figure(figsize=(10, 10))
# ax = figure.add_subplot(projection='3d')
# ax.scatter(lidar[:, 0], lidar[:, 1], lidar[:, 2], s=1)
# for item in vehicles:
#     ax.scatter(item[0], item[1], item[2], color='r', s=5)
# plt.show()





fig = mlab.figure(figure=None, bgcolor=(0,0,0),
                      fgcolor=None, engine=None, size=(1000, 500))
fig = draw_lidar(lidar, fig)
# colors = [(1, 1, 1), (0, 0, 1), (0, 1, 0), (1, 0, 0), (1, 0, 1)]
# for i, items in enumerate(bounding_box.values()):
#     for item in items:
#         item = item[0]
#         center = np.array([-item[1], item[0], item[2] - 0.7])
#         l, w, h = item[4], item[3], item[5]
#         rotation = yaw(np.deg2rad(item[-1] - 90))
#         box = []
#
#         box = np.array([[l, w, h],
#                         [l, w, -h],
#                         [l, -w, h],
#                         [l, -w, -h],
#                         [-l, w, h],
#                         [-l, w, -h],
#                         [-l, -w, h],
#                         [-l, -w, -h]
#                         ])
#         box = rotation @ np.array(box).T
#         points = np.array([box.T + center])
#         fig = draw_gt_boxes3d(points, fig, color=colors[i])

mlab.show()

# pt = o3d.geometry.PointCloud()
# pt.points = o3d.utility.Vector3dVector(lidar)
#
# objects = [pt]
# for item in vehicles:
#     center = [-item[1], item[0], item[2] - 0.7]
#     r = yaw(np.deg2rad(item[-1] - 90))
#     # r = np.eye(3)
#     extent = np.array([item[4], item[3], item[5]]) * 2
#     print(extent)
#     sample = o3d.geometry.OrientedBoundingBox(center, r, extent)
#     objects.append(sample)
#
# # o3d.visualization.draw_geometries(objects)
