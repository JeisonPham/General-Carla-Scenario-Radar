import os
import json
from PIL import Image

import numpy as np
import torch 
from torch.utils.data import Dataset
from tqdm import tqdm
import sys
import pickle
class CARLA_Data(Dataset):

    def __init__(self, root, config):
        
        self.seq_len = config.seq_len
        self.pred_len = config.pred_len
        self.ignore_sides = config.ignore_sides
        self.ignore_rear = config.ignore_rear

        self.input_resolution = config.input_resolution
        self.scale = config.scale

        self.lidar = []
        self.front = []
        self.left = []
        self.right = []
        self.rear = []
        self.maps = []
        self.vectormap = []
        self.radar = []
        self.semantic_lidar = []
        self.x = []
        self.y = []
        self.x_command = []
        self.y_command = []
        self.theta = []
        self.steer = []
        self.throttle = []
        self.brake = []
        self.command = []
        self.velocity = []
        
        for sub_root in tqdm(root, file=sys.stdout):
            
            preload_file = os.path.join(sub_root, 'rg_lidar_mmfn_diag_pl_'+str(self.seq_len)+'_'+str(self.pred_len)+'.npy')
            # dump to npy if no preload
            
            # if not os.path.exists(preload_file):
            # if False:
            if True:
                preload_front = []
                preload_lidar = []
                preload_x = []
                preload_y = []
                preload_x_command = []
                preload_y_command = []
                preload_theta = []
                preload_steer = []
                preload_throttle = []
                preload_brake = []
                preload_command = []
                preload_velocity = []
                preload_maps = []
                preload_vectormap = []
                preload_radar = []
                preload_semantic_lidar = []
                # list sub-directories in root 
                root_files = os.listdir(sub_root)
                routes = [folder for folder in root_files if not os.path.isfile(os.path.join(sub_root,folder))]
                for route in routes:
                    route_dir = os.path.join(sub_root, route)
                    # print(route_dir)
                    # subtract final frames (pred_len) since there are no future waypoints
                    # first frame of sequence not used
                    
                    num_seq = (len(os.listdir(route_dir+"/rgb_front/"))-self.pred_len-2)//self.seq_len
                    
                    for seq in range(num_seq):
                        fronts = []
                        lidars = []
                        semantic_lidar = []
                        xs = []
                        ys = []
                        thetas = []
                        maps = []
                        vector_maps = []
                        radar = []
                        # read files sequentially (past and current frames)
                        for i in range(self.seq_len):
                            # images
                            filename = f"{str(seq*self.seq_len+1+i).zfill(4)}.png"
                            fronts.append(route_dir+"/rgb_front/"+filename)
                            vector_maps.append(route_dir+"/vectormap/"+f"{str(seq*self.seq_len+1+i).zfill(4)}.npy")
                            maps.append(route_dir+"/maps/"+filename)

                            # point cloud
                            lidars.append(route_dir + f"/lidar/{str(seq*self.seq_len+1+i).zfill(4)}.npy")

                            # semantic lidar
                            
                            # radar
                            radar.append(route_dir + f"/radar/{str(seq*self.seq_len+1+i).zfill(4)}.npy")

                            # position
                            with open(route_dir + f"/measurements/{str(seq*self.seq_len+1+i).zfill(4)}.json", "r") as read_file:
                                data = json.load(read_file)
                            xs.append(data['x'])
                            ys.append(data['y'])
                            thetas.append(data['theta'])

                        # get control value of final frame in sequence
                        preload_x_command.append(data['x_command'])
                        preload_y_command.append(data['y_command'])
                        preload_steer.append(data['steer'])
                        preload_throttle.append(data['throttle'])
                        preload_brake.append(data['brake'])
                        preload_command.append(data['command'])
                        preload_velocity.append(data['speed'])
                        
                        # read files sequentially (future frames)
                        for i in range(self.seq_len, self.seq_len + self.pred_len):
                            # point cloud
                            lidars.append(route_dir + f"/lidar/{str(seq*self.seq_len+1+i).zfill(4)}.npy")
                            radar.append(route_dir + f"/radar/{str(seq*self.seq_len+1+i).zfill(4)}.npy")
                            
                            # position
                            with open(route_dir + f"/measurements/{str(seq*self.seq_len+1+i).zfill(4)}.json", "r") as read_file:
                                data = json.load(read_file)
                            xs.append(data['x'])
                            ys.append(data['y'])

                            # fix for theta=nan in some measurements
                            if np.isnan(data['theta']):
                                thetas.append(0)
                            else:
                                thetas.append(data['theta'])

                        preload_front.append(fronts)
                        preload_lidar.append(lidars)
                        preload_x.append(xs)
                        preload_y.append(ys)
                        preload_theta.append(thetas)
                        preload_maps.append(maps)
                        preload_vectormap.append(vector_maps)
                        preload_radar.append(radar)
                # dump to npy
                preload_dict = {}
                preload_dict['front'] = preload_front
                preload_dict['lidar'] = preload_lidar
                preload_dict['maps'] = preload_maps
                preload_dict['vectormap'] = preload_vectormap
                preload_dict['radar'] = preload_radar
                preload_dict['x'] = preload_x
                preload_dict['y'] = preload_y
                preload_dict['x_command'] = preload_x_command
                preload_dict['y_command'] = preload_y_command
                preload_dict['theta'] = preload_theta
                preload_dict['steer'] = preload_steer
                preload_dict['throttle'] = preload_throttle
                preload_dict['brake'] = preload_brake
                preload_dict['command'] = preload_command
                preload_dict['velocity'] = preload_velocity
                np.save(preload_file, preload_dict)

            # load from npy if available
            preload_dict = np.load(preload_file, allow_pickle=True)
            self.front += preload_dict.item()['front']
            self.lidar += preload_dict.item()['lidar']
            self.maps += preload_dict.item()['maps']
            self.vectormap += preload_dict.item()['vectormap']
            self.radar += preload_dict.item()['radar']
            self.x += preload_dict.item()['x']
            self.y += preload_dict.item()['y']
            self.x_command += preload_dict.item()['x_command']
            self.y_command += preload_dict.item()['y_command']
            self.theta += preload_dict.item()['theta']
            self.steer += preload_dict.item()['steer']
            self.throttle += preload_dict.item()['throttle']
            self.brake += preload_dict.item()['brake']
            self.command += preload_dict.item()['command']
            self.velocity += preload_dict.item()['velocity']
            tqdm.write("Preloading " + str(len(preload_dict.item()['front'])) + " sequences from " + preload_file)

    def __len__(self):
        """Returns the length of the dataset. """
        return len(self.front)

    def __getitem__(self, index):
        """Returns the item at index idx. """
        data = dict()
        data['fronts'] = []
        data['lidars'] = []
        data['vectormaps'] = []
        data['radar'] = []
        data['maps'] = []
        seq_fronts = self.front[index]
        seq_lidars = self.lidar[index]
        seq_maps = self.maps[index]
        seq_vectormaps = self.vectormap[index]
        seq_radar = self.radar[index]
        seq_x = self.x[index]
        seq_y = self.y[index]
        seq_theta = self.theta[index]

        full_lidar = []
        
        for i in range(self.seq_len):
            data['fronts'].append(torch.from_numpy(np.array(scale_and_crop_image(Image.open(seq_fronts[i]), scale=self.scale, crop=self.input_resolution))))
            
            # for vectormap didn't create
            reindex = index
            # while not os.path.exists(seq_vectormaps[i]):
            #     if reindex - 1 >= 0:
            #         reindex -= 1
            #     else:
            #         reindex += 1
            #     print('there is not vectormap on', index, 'reindex at ', reindex)
            #     seq_vectormaps[i] = self.vectormap[reindex]
            #
            # data['vectormaps'].append(torch.from_numpy(np.load(seq_vectormaps[i])))
            
            data['maps'].append(torch.from_numpy(np.transpose(Image.open(seq_maps[i]), (2,0,1))))
            data['radar'].append(radar_to_size(np.load(seq_radar[i]), (81,5)))

            lidar_unprocessed = np.load(seq_lidars[i])[...,:3] # lidar: XYZI
            full_lidar.append(lidar_unprocessed)

            # fix for theta=nan in some measurements
            if np.isnan(seq_theta[i]):
                seq_theta[i] = 0.
        ego_x = seq_x[i]
        ego_y = seq_y[i]
        ego_theta = seq_theta[i]

        # process only past lidar point clouds
        if i < self.seq_len:
            # convert coordinate frame of point cloud
            full_lidar[i][:,1] *= -1 # inverts x, y
            full_lidar[i] = transform_2d_points(full_lidar[i], 
                np.pi/2-seq_theta[i], -seq_x[i], -seq_y[i], np.pi/2-ego_theta, -ego_x, -ego_y)
            lidar_processed = lidar_to_histogram_features(full_lidar[i], crop=self.input_resolution)
            data['lidars'].append(lidar_processed)

        # waypoint processing to local coordinates
        waypoints = []
        for i in range(self.seq_len + self.pred_len):
            # waypoint is the transformed version of the origin in local coordinates
            # we use 90-theta instead of theta
            # LBC code uses 90+theta, but x is to the right and y is downwards here
            local_waypoint = transform_2d_points(np.zeros((1,3)), 
                np.pi/2-seq_theta[i], -seq_x[i], -seq_y[i], np.pi/2-ego_theta, -ego_x, -ego_y)
            waypoints.append(tuple(local_waypoint[0,:2]))
        data['waypoints'] = waypoints

        # convert x_command, y_command to local coordinates
        # taken from LBC code (uses 90+theta instead of theta)
        R = np.array([
            [np.cos(np.pi/2+ego_theta), -np.sin(np.pi/2+ego_theta)],
            [np.sin(np.pi/2+ego_theta),  np.cos(np.pi/2+ego_theta)]
            ])
    
        # local_command_point = np.array([-self.y_command[index]-ego_x, self.x_command[index]-ego_y])
        local_command_point = np.array([self.x_command[index]-ego_x, self.y_command[index]-ego_y])
        
        local_command_point = R.T.dot(local_command_point)
        data['target_point'] = tuple(local_command_point)
        data['steer'] = self.steer[index]
        data['throttle'] = self.throttle[index]
        data['brake'] = self.brake[index]
        data['command'] = self.command[index]
        data['velocity'] = self.velocity[index]

        return data


def lidar_to_histogram_features(lidar, crop=256):
    """
    Convert LiDAR point cloud into 2-bin histogram over 256x256 grid
    """
    def splat_points(point_cloud):
        # 256 x 256 grid
        pixels_per_meter = 8
        hist_max_per_pixel = 5
        meters_max = 32
        xbins = np.linspace(-16, 16, meters_max*pixels_per_meter+1)
        ybins = np.linspace(-24, 8,  meters_max*pixels_per_meter+1)
        hist = np.histogramdd(point_cloud[...,:2], bins=(xbins, ybins))[0]
        hist[hist>hist_max_per_pixel] = hist_max_per_pixel
        overhead_splat = hist/hist_max_per_pixel
        return overhead_splat

    below = lidar[lidar[...,2]<=-2.0]
    above = lidar[lidar[...,2]>-2.0]
    below_features = splat_points(below)
    above_features = splat_points(above)
    features = np.stack([below_features, above_features], axis=-1)
    features = np.transpose(features, (2, 0, 1)).astype(np.float32)
    return features


def scale_and_crop_image(image, scale=1, crop=256):
    """
    Scale and crop a PIL image, returning a channels-first numpy array.
    """
    # image = Image.open(filename)
    (width, height) = (int(image.width // scale), int(image.height // scale))
    im_resized = image.resize((width, height))
    image = np.asarray(im_resized)
    start_x = height//2 - crop//2
    start_y = width//2 - crop//2
    cropped_image = image[start_x:start_x+crop, start_y:start_y+crop]
    cropped_image = np.transpose(cropped_image, (2,0,1))
    return cropped_image


def transform_2d_points(xyz, r1, t1_x, t1_y, r2, t2_x, t2_y):
    """
    Build a rotation matrix and take the dot product.
    """
    # z value to 1 for rotation
    xy1 = xyz.copy()
    xy1[:,2] = 1

    c, s = np.cos(r1), np.sin(r1)
    r1_to_world = np.matrix([[c, s, t1_x], [-s, c, t1_y], [0, 0, 1]])

    # np.dot converts to a matrix, so we explicitly change it back to an array
    world = np.asarray(r1_to_world @ xy1.T)

    c, s = np.cos(r2), np.sin(r2)
    r2_to_world = np.matrix([[c, s, t2_x], [-s, c, t2_y], [0, 0, 1]])
    world_to_r2 = np.linalg.inv(r2_to_world)

    out = np.asarray(world_to_r2 @ world).T
    
    # reset z-coordinate
    out[:,2] = xyz[:,2]

    return out

def radar_to_size(data, target_size):
    data        = np.asarray(data)
    target_data = np.zeros(target_size)
    if data.shape[0] >= target_data.shape[0]:
        # remove bigger ttc, depth/velocity = ttc
        n = data.shape[0] - target_data.shape[0]
        target_data = np.delete(data, (-abs(data[:,0]/data[:,3])).argsort()[:n], 0)
    else:
        m = data.shape[0]
        target_data[:m,:] = data[:m,:]
    return target_data


class PRE_Data(Dataset):
    def __init__(self, root, config, data_use='train'):
        self.seq_len = config.seq_len
        self.pred_len = config.pred_len
        self.preload_dict = []
        
        preload_file = os.path.join(root, 'rg_vec_mmfn_diag_pl_'+str(self.seq_len)+'_'+str(self.pred_len)+ '_' + data_use +'.npy')
        preload_dict = []

        if not os.path.exists(preload_file):
            # list sub-directories in root
            for pkl_file in os.listdir(root):
                if pkl_file.split('.')[-1]=='pkl':
                    pkl_file = str(root) + '/' + pkl_file
                    preload_dict.append(pkl_file)
            np.save(preload_file, preload_dict)
            print("Saving preloading file for ",data_use)

        # load from npy if available
        preload_dict = np.load(preload_file) #, allow_pickle=True
        self.preload_dict = preload_dict
        print("Preloading sequences from " + preload_file)

    def __len__(self):
        """Returns the length of the dataset. """
        return len(self.preload_dict)

    def __getitem__(self, index):
        """Returns the item at index idx. """

        with open(self.preload_dict[index], 'rb') as fd:
            data = pickle.load(fd)
        data_test = []
        for i in range(81):
            data_test.append(data['radar'][0][:,1] - data['radar'][0][i,1])
        data['radar_adj'] = np.array(data_test)
        return data