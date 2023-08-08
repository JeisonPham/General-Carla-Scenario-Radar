import time, random, os
import datetime, json, pathlib
from pathlib import Path
import cv2
import carla

from leaderboard.autoagents import autonomous_agent, gen
from .planner_controller import RoutePlanner
from .utils import imu_msg, LocalizationOperator, from_imu, from_gps, build_rmap, RoughMap, bc
from .carla_birdeye_view import BirdViewProducer, BirdViewCropType, PixelDimensions
import numpy as np
from PIL import Image

WEATHERS = {
    'ClearNoon': carla.WeatherParameters.ClearNoon,
    'ClearSunset': carla.WeatherParameters.ClearSunset,

    'CloudyNoon': carla.WeatherParameters.CloudyNoon,
    'CloudySunset': carla.WeatherParameters.CloudySunset,

    'WetNoon': carla.WeatherParameters.WetNoon,
    'WetSunset': carla.WeatherParameters.WetSunset,

    'MidRainyNoon': carla.WeatherParameters.MidRainyNoon,
    'MidRainSunset': carla.WeatherParameters.MidRainSunset,

    'WetCloudyNoon': carla.WeatherParameters.WetCloudyNoon,
    'WetCloudySunset': carla.WeatherParameters.WetCloudySunset,

    'HardRainNoon': carla.WeatherParameters.HardRainNoon,
    'HardRainSunset': carla.WeatherParameters.HardRainSunset,

    'SoftRainNoon': carla.WeatherParameters.SoftRainNoon,
    'SoftRainSunset': carla.WeatherParameters.SoftRainSunset,
}
WEATHERS_IDS = list(WEATHERS)


class BaseAgent(autonomous_agent.AutonomousAgent):
    def setup(self, path_to_conf_file):
        self.track = autonomous_agent.Track.MAP
        self.config = path_to_conf_file

        self.rough_map = RoughMap(self.config.up, self.config.down, self.config.left, self.config.right,
                                  self.config.lane_node_num, self.config.feature_num)
        self.rough_map_have_load = False

        self.step = -1
        self.wall_start = time.time()
        self.initialized = False



        self._sensor_data = {
            'width': self.config.camera_w,
            'height': self.config.camera_h,
            'fov': self.config.camera_fov
        }

        self.save_path, self.weather_id = None, None

        if self.config.save_data:
            now = datetime.datetime.now()
            if self.config.town is None:
                self.config.town = "TEST"
            string = pathlib.Path(self.config.town).stem + '_' + str(self.config.route_id) + '_'
            string += '_'.join(map(lambda x: '%02d' % x, (now.month, now.day, now.hour, now.minute, now.second)))
            self.save_path = pathlib.Path(self.config.data_save) / string
            print("Data save path", self.save_path)
            self.save_path.mkdir(parents=True, exist_ok=False)

            (self.save_path / 'measurements').mkdir(parents=True, exist_ok=True)
            (self.save_path / 'vectormap').mkdir(parents=True, exist_ok=True)
            (self.save_path / 'lidar').mkdir(parents=True, exist_ok=True)
            (self.save_path / 'maps').mkdir(parents=True, exist_ok=True)
            (self.save_path / 'opendrive').mkdir(parents=True, exist_ok=True)
            (self.save_path / 'radar').mkdir(parents=True, exist_ok=True)
            (self.save_path / 'rgb_front').mkdir()
            (self.save_path / 'bb3d').mkdir(parents=True, exist_ok=True)
            (self.save_path / 'semantic_lidar').mkdir(parents=True, exist_ok=True)



        self.agent_loc, self.agent_rot = None, None
        self.imu_data = imu_msg()
        self.agent_vel, self.yaw = None, None
        self.agent_pose = LocalizationOperator()
        self.prev_lidar = None

    def _init(self):
        self._command_planner = RoutePlanner(7.5, 25.0, 257)
        self._command_planner.set_route(self._global_plan, True)

        self.initialized = True

        self._sensor_data['calibration'] = self._get_camera_to_car_calibration(self._sensor_data)

        self._sensors = self.sensor_interface._sensors_objects



    def _get_position(self, tick_data):
        gps = tick_data['gps']
        gps = (gps - self._command_planner.mean) * self._command_planner.scale

        return gps

    def sensors(self):
        return [
            {
                'type': 'sensor.camera.rgb',
                'x': self.config.camera_x, 'y': 0.0, 'z': self.config.camera_z,
                'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0,
                'width': self._sensor_data['width'], 'height': self._sensor_data['height'],
                'fov': self._sensor_data['fov'],
                'id': 'rgb_front'
            },
            {
                'type': 'sensor.lidar.ray_cast',
                'x': 0.7, 'y': 0.0, 'z': 1.60,
                'roll': 0.0, 'pitch': 0.0, 'yaw': -90.0,
                'range': 100, "channels": 64, 'upper_fov': 15, 'lower_fov': -15,
                'id': 'lidar'
            },
            {
                'type': 'sensor.lidar.ray_cast_semantic',
                'x': 0.7, 'y': 0.0, 'z': 1.60,
                'roll': 0.0, 'pitch': 0.0, 'yaw': -90.0,
                'range': 100, "channels": 64, 'upper_fov': 15, 'lower_fov': -15,
                'id': 'semantic_lidar_front'
            },
            {
                'type': 'sensor.lidar.ray_cast_semantic',
                'x': 0.7, 'y': 0.0, 'z': 1.60,
                'roll': 0.0, 'pitch': 0.0, 'yaw': -270.0,
                'range': 100, "channels": 64, 'upper_fov': 15, 'lower_fov': -15,
                'id': 'semantic_lidar_back'
            },
            {
                'type': 'sensor.other.imu',
                'x': 0.0, 'y': 0.0, 'z': 0.0,
                'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0,
                'sensor_tick': 0.05,
                'id': 'imu'
            },
            {
                'type': 'sensor.other.gnss',
                'x': 0.0, 'y': 0.0, 'z': 0.0,
                'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0,
                'sensor_tick': 0.01,
                'id': 'gps'
            },
            {
                'type': 'sensor.speedometer',
                'reading_frequency': 20,
                'id': 'speed'
            },
            {
                'type': 'sensor.opendrive_map',
                'reading_frequency': 1,
                'id': 'opendrive'
            },
            {
                'type': 'sensor.other.radar',
                'x': 2.8, 'y': 0.0, 'z': 1.00,
                'roll': 0.0, 'pitch': 5.0, 'yaw': 0.0,
                'fov': 35, 'id': 'radar_front'
            },
            {
                'type': 'sensor.other.radar',
                'x': -2.8, 'y': 0.0, 'z': 1.00,
                'roll': 0.0, 'pitch': 5.0, 'yaw': -180,
                'fov': 35, 'id': 'radar_rear'
            },
        ]

    # def _get_3d_bbs(self, max_distance=50):

    #         bounding_boxes = {
    #             "traffic_lights": [],
    #             "stop_signs": [],
    #             "vehicles": [],
    #             "pedestrians": [],
    #         }

    #         bounding_boxes["traffic_lights"] = self._find_obstacle_3dbb(
    #             carla.CityObjectLabel.TrafficLight, max_distance
    #         )
    #         bounding_boxes["stop_signs"] = self._find_obstacle_3dbb(carla.CityObjectLabel.TrafficSigns, max_distance)
    #         bounding_boxes["vehicles"] = self._find_obstacle_3dbb(carla.CityObjectLabel.Vehicles, max_distance)
    #         bounding_boxes["pedestrians"] = self._find_obstacle_3dbb(
    #             carla.CityObjectLabel.Pedestrians, max_distance
    #         )

    #         return bounding_boxes
    def get_matrix(self, transform):
        """
        Creates matrix from carla transform.
        """

        rotation = transform.rotation
        location = transform.location
        c_y = np.cos(np.radians(rotation.yaw))
        s_y = np.sin(np.radians(rotation.yaw))
        c_r = np.cos(np.radians(rotation.roll))
        s_r = np.sin(np.radians(rotation.roll))
        c_p = np.cos(np.radians(rotation.pitch))
        s_p = np.sin(np.radians(rotation.pitch))
        matrix = np.matrix(np.identity(4))
        matrix[0, 3] = location.x
        matrix[1, 3] = location.y
        matrix[2, 3] = location.z
        matrix[0, 0] = c_p * c_y
        matrix[0, 1] = c_y * s_p * s_r - s_y * c_r
        matrix[0, 2] = -c_y * s_p * c_r - s_y * s_r
        matrix[1, 0] = s_y * c_p
        matrix[1, 1] = s_y * s_p * s_r + c_y * c_r
        matrix[1, 2] = -s_y * s_p * c_r + c_y * s_r
        matrix[2, 0] = s_p
        matrix[2, 1] = -c_p * s_r
        matrix[2, 2] = c_p * c_r
        return matrix

    def save_map(self, input_data):
        self.CarlaMap = carla.Map('map', input_data['opendrive'][1]['opendrive'])
        self.birdview_producer = BirdViewProducer(
            self.CarlaMap,
            target_size=PixelDimensions(width=256, height=256),
            render_lanes_on_junctions=True,
            pixels_per_meter=8,
            crop_type=BirdViewCropType.FRONT_AND_REAR_AREA,
        )
        if self.save_path is not None:
            print("Loading success for map, save path is:", self.save_path)
            open_drive_folder = os.path.join(self.save_path, "opendrive")
            with open(os.path.join(open_drive_folder, "opstr.txt"), "w") as text_file:
                text_file.write(input_data['opendrive'][1]['opendrive'])

            lib_path = os.path.abspath('../../../assets/package')
            if os.path.exists(lib_path):
                # vector representation save
                tmp_dir = open_drive_folder
                is_error = build_rmap([tmp_dir], lib_path)
                if not is_error:
                    self.rough_map.read(os.path.join(open_drive_folder, "a.rmap"))
                    print("load rough_map which lane_num = ", len(self.rough_map.lanes))
                    self.rough_map_have_load = True

    def _find_obstacle_3dbb(self, obstacle_type, max_distance=50):
        """Returns a list of 3d bounding boxes of type obstacle_type.
        If the object does have a bounding box, this is returned. Otherwise a bb
        of size 0.5,0.5,2 is returned at the origin of the object.
        Args:
            obstacle_type (String): Regular expression
            max_distance (int, optional): max search distance. Returns all bbs in this radius. Defaults to 50.
        Returns:
            List: List of Boundingboxes
        """
        obst = list()

        _actors = self._world.get_actors()
        _obstacles = _actors.filter(obstacle_type)

        ego_transform = self._vehicle.get_transform()
        inv_matrix = np.array(ego_transform.get_inverse_matrix())
        _obstacle = self._vehicle

        transform = _obstacle.get_transform()
        bounding_box = _obstacle.bounding_box
        bounding_box.location += transform.location

        extent = bounding_box.extent
        rotation = transform.rotation

        bb = np.array(
            [
                [0, 0, 0, extent.x, extent.y, extent.z, rotation.yaw],
            ]
        )
        obst.append(bb)

        for _obstacle in _obstacles:
            distance_to_car = _obstacle.get_transform().location.distance(
                self._vehicle.get_location()
            )

            if 0 < distance_to_car <= max_distance:

                if hasattr(_obstacle, "bounding_box"):
                    transform = _obstacle.get_transform()
                    bounding_box = _obstacle.bounding_box
                    bounding_box.location += transform.location
                    # self._world.debug.draw_box(bounding_box, transform.rotation)


                    loc = bounding_box.location
                    loc = inv_matrix @ np.array([[loc.x, loc.y, loc.z, 1]]).T.flatten()
                    # loc = _obstacle.get_transform().location - self._vehicle.get_transform().location
                    # print(loc)
                    # assert False
                    # _obstacle.get_transform().transform(loc)

                    extent = bounding_box.extent
                    rotation = transform.rotation

                    bb = np.array(
                        [
                            [loc[0], loc[1], loc[2], extent.x, extent.y, extent.z, rotation.yaw],
                        ]
                    )

                else:
                    loc = _obstacle.get_transform().location
                    loc = inv_matrix @ np.array([[loc.x, loc.y, loc.z, 1]]).T.flatten()
                    bb = np.array([[loc[0], loc[1], loc[2], 0.5, 0.5, 2, _obstacle.get_transform().rotation.yaw]])

                obst.append(bb)

        return obst

    def tick(self, input_data, timestamp):
        self.step += 1

        rgb_front = cv2.cvtColor(input_data['rgb_front'][1][:, :, :3], cv2.COLOR_BGR2RGB)
        radar_front = np.hstack((input_data['radar_front'][1], np.ones((input_data['radar_front'][1].shape[0], 1))))
        radar_rear = np.hstack((input_data['radar_rear'][1], np.zeros((input_data['radar_rear'][1].shape[0], 1))))
        radar_all = np.concatenate((radar_front, radar_rear), axis=0)
        gps = input_data['gps'][1][:2]
        speed = input_data['speed'][1]['speed']
        compass = input_data['imu'][1][-1]
        weather = self._weather_to_dict(self._world.get_weather())

        # print(input_data['lidar'])

        # assert False

        # smooth localization
        self.agent_vel = np.around(input_data['speed'][1]['speed'], 2)
        self.agent_loc = from_gps(input_data['gps'][1][0], input_data['gps'][1][1], input_data['gps'][1][2])

        self.imu_data.update(input_data['imu'], timestamp)
        self.yaw, self.agent_rot = from_imu(self, self.imu_data)
        self.agent_pose.update_pose(self.agent_loc, self.agent_vel, self.imu_data, self.yaw, self.agent_rot,
                                    USE_EKF=False)
        agent_tf = self.agent_pose._pose_msg.transform

        # 大致给了框 通过现有的定位来生成map地图
        birdview = self.birdview_producer.produce(agent_tf, np.array([2.51, 1.07]))
        rgb_birdview = cv2.cvtColor(BirdViewProducer.as_rgb(birdview), cv2.COLOR_BGR2RGB)

        bb3d = self._get_3d_bbs(50)
        result = {
            'rgb_front': rgb_front,
            'lidar': input_data['lidar'][1],
            'semantic_lidar_front': input_data['semantic_lidar_front'][1],
            'semantic_lidar_back': input_data['semantic_lidar_back'][1],
            'gps': gps,
            'speed': speed,
            'compass': compass,
            'weather': weather,
            'opendrive': rgb_birdview,
            'radar': radar_all,
            'bb3d': bb3d
        }

        if self.rough_map_have_load:
            pose2d = np.array([result['gps'][0], result['gps'][1], result['compass']]).astype(np.float)
            vectormap_lanes = self.rough_map.process(pose2d)
            if vectormap_lanes.shape[0] == 0:
                vectormap_lanes = np.zeros((1, 10, 5))
                print(f"====> {self.step // 10} warning, the vehicle is out of lane")
                result['vectormap_lanes'] = vectormap_lanes
            result['vectormap_lanes'] = vectormap_lanes
        return result

    def save(self, near_node, far_node, near_command, steer, throttle, brake, target_speed, tick_data, reverse):
        frame = self.step // 10

        # if not hasattr(self, "semantic_lidar"):
        #     sensor_location = carla.Location(x=0, y=0,
        #                                      z=1.6)
        #     sensor_rotation = carla.Rotation(pitch=0,
        #                                      roll=0,
        #                                      yaw=-90)
        #
        #     sensor_transform = carla.Transform(sensor_location, sensor_rotation)
        #     lidar_transform = carla.Transform(carla.Location(x=0, y=0, z=1.80),
        #                                       carla.Rotation(pitch=0, yaw=180, roll=0))
        #     self.semantic_lidar = gen.HDL64E(self._vehicle, self._world, [], self.save_path, sensor_transform)
        #     self.semantic_lidar.init()

        # self.semantic_lidar.save()


        pos = self._get_position(tick_data)
        theta = tick_data['compass']
        speed = tick_data['speed']
        weather = tick_data['weather']

        data = {
            'x': pos[0],
            'y': pos[1],
            'theta': theta,
            'speed': speed,
            'target_speed': target_speed,
            # 'x_command': far_node[0],
            # 'y_command': far_node[1],
            'x_command': -far_node[1],
            'y_command': far_node[0],
            'command': near_command.value,
            'steer': steer,
            'throttle': throttle,
            'brake': brake,
            'reverse': reverse,
            'weather_full': weather,
            'weather_id': self.weather_id,
            'transform': self._vehicle.get_transform().get_matrix()
        }

        measurements_file = self.save_path / 'measurements' / ('%04d.json' % frame)
        f = open(measurements_file, 'w')
        json.dump(data, f, indent=4)
        f.close()

        Image.fromarray(tick_data['rgb_front']).save(self.save_path / 'rgb_front' / ('%04d.png' % frame))
        Image.fromarray(tick_data['opendrive']).save(self.save_path / 'maps' / ('%04d.png' % frame))
        # lidar
        if self.prev_lidar is not None:
            lidar = np.append(tick_data['lidar'], self.prev_lidar, axis=0)
        else:
            lidar = tick_data['lidar']


        np.save(self.save_path / 'lidar' / ('%04d.npy' % frame), lidar, allow_pickle=True)

        semantic_lidar_front = tick_data['semantic_lidar_front']
        semantic_lidar_back = tick_data['semantic_lidar_back']
        semantic_lidar_back[:, (0, 1)] *= -1
        semantic_lidar = np.concatenate([semantic_lidar_front, semantic_lidar_back], axis=0)

        with open(self.save_path / 'semantic_lidar' / ('%04d.npy' % frame), 'wb') as file:
            np.save(file, semantic_lidar, allow_pickle=True)

        np.save(self.save_path / 'radar' / ('%04d.npy' % frame), tick_data['radar'], allow_pickle=True)

        np.save(self.save_path / 'bb3d' / ('%04d.npy' % frame), tick_data['bb3d'], allow_pickle=True)

        if self.rough_map_have_load:
            np.save(self.save_path / 'vectormap' / ('%04d.npy' % frame), tick_data['vectormap_lanes'],
                    allow_pickle=True)

        if self.config.weather_change:
            self.change_weather()

    def force_destory_actor(self, obs, light, walker):
        if obs:
            self._world.get_actor(obs.id).destroy()
            self.stop_counter = 0
            print(
                f"{self.step}, {bc.WARNING}ATTENTION:{bc.ENDC} force to detroy actor {obs.id} stopping for a long time")
        elif walker:
            self._world.get_actor(walker.id).destroy()
            self.stop_counter = 0
            print(
                f"{self.step}, {bc.WARNING}ATTENTION:{bc.ENDC} force to detroy actor {walker.id} stopping for a long time")
        elif light and self.stop_counter > self.config.counter_destory * 2:
            light.set_green_time(10.0)
            light.set_state(carla.TrafficLightState.Green)
            self.stop_counter = 0
            print(f"{self.step}, {bc.WARNING}ATTENTION:{bc.ENDC} force to setting green light {light.id}")
        else:
            print(f"{bc.WARNING}==========> warnning!!!! {bc.ENDC} None factor trigger the stop!!!")
            return

    def change_weather(self):
        index = random.choice(range(len(WEATHERS)))
        self.weather_id = WEATHERS_IDS[index]
        weather = WEATHERS[WEATHERS_IDS[index]]
        self._world.set_weather(weather)

    def _weather_to_dict(self, carla_weather):
        weather = {
            'cloudiness': carla_weather.cloudiness,
            'precipitation': carla_weather.precipitation,
            'precipitation_deposits': carla_weather.precipitation_deposits,
            'wind_intensity': carla_weather.wind_intensity,
            'sun_azimuth_angle': carla_weather.sun_azimuth_angle,
            'sun_altitude_angle': carla_weather.sun_altitude_angle,
            'fog_density': carla_weather.fog_density,
            'fog_distance': carla_weather.fog_distance,
            'wetness': carla_weather.wetness,
            'fog_falloff': carla_weather.fog_falloff,
        }

        return weather

    def _get_camera_to_car_calibration(self, sensor):
        """returns the calibration matrix for the given sensor

        Args:
            sensor ([type]): [description]

        Returns:
            [type]: [description]
        """
        calibration = np.identity(3)
        calibration[0, 2] = sensor["width"] / 2.0
        calibration[1, 2] = sensor["height"] / 2.0
        calibration[0, 0] = calibration[1, 1] = sensor["width"] / (2.0 * np.tan(sensor["fov"] * np.pi / 360.0))
        return calibration

    def _get_3d_bbs(self, max_distance=50):

        bounding_boxes = {
            "traffic_lights": [],
            "stop_signs": [],
            "vehicles": [],
            "pedestrians": [],
        }

        # bounding_boxes["traffic_lights"] = self._find_obstacle_3dbb(
        #     "*traffic_light*", max_distance
        # )
        # bounding_boxes["stop_signs"] = self._find_obstacle_3dbb("*stop*", max_distance)
        bounding_boxes["vehicles"] = self._find_obstacle_3dbb("*vehicle*", max_distance)
        # bounding_boxes["pedestrians"] = self._find_obstacle_3dbb(
        #     "*walker*", max_distance
        # )

        return bounding_boxes
