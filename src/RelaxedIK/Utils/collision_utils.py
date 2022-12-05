import fcl
import numpy as np
import yaml
from RelaxedIK.Utils.colors import bcolors as bc
from visualization_msgs.msg import Marker
import rospy
import RelaxedIK.Utils.transformations as T
import trimesh

# >>>>> DEBUG
CAPSULE_DATA_PANDA = [
    {
        "name": "panda_link0",
        "frame": 0,
        "rpy": [0, 1.5707963267948966, 0],
        "xyz": [-0.075, 0, 0.06],
        "length": 0.03,
        "radius": 0.09,
    },
    {
        "name": "panda_link1",
        "frame": 1,
        "rpy": [0, 0, 0],
        "xyz": [0, 0, -0.1915],
        "length": 0.283,
        "radius": 0.09,
    },
    {
        "name": "panda_link2",
        "frame": 2,
        "rpy": [0, 0, 0],
        "xyz": [0, 0, 0],
        "length": 0.12,
        "radius": 0.09,
    },
    {
        "name": "panda_link3",
        "frame": 3,
        "rpy": [0, 0, 0],
        "xyz": [0, 0, -0.145],
        "length": 0.15,
        "radius": 0.09,
    },
    {
        "name": "panda_link4",
        "frame": 4,
        "rpy": [0, 0, 0],
        "xyz": [0, 0, 0],
        "length": 0.12,
        "radius": 0.09,
    },
    {
        "name": "panda_link5",
        "frame": 5,
        "rpy": [0, 0, 0],
        "xyz": [0, 0, -0.26],
        "length": 0.1,
        "radius": 0.09,
    },
    {
        "name": "panda_link5_2",
        "frame": 5,
        "rpy": [0, 0, 0],
        "xyz": [0, 0.08, -0.13],
        "length": 0.14,
        "radius": 0.055,
    },
    {
        "name": "panda_link6",
        "frame": 6,
        "rpy": [0, 0, 0],
        "xyz": [0, 0, -0.03],
        "length": 0.08,
        "radius": 0.08,
    },
    {
        "name": "panda_link7",
        "frame": 7,
        "rpy": [0, 0, 0],
        "xyz": [0, 0, 0.01],
        "length": 0.14,
        "radius": 0.07,
    },
]
mesh_data = trimesh.exchange.stl.load_stl(
    open(
        "/root/MOTION_CONTROL_BENCHMARK/robots/meshes/robotiq_gripper_with_tweezer/collision/robotiq_85_base_link_simplified.stl",
        "rb",
    )
)
# mesh = trimesh.Trimesh(**mesh_data)
MESHES_DATA_PANDA = [
    {
        "name": "robotiq_85_base_link",
        "frame": 8,
        "rpy": [0, 0, 0],
        "xyz": [0, 0, 0],
        "file": "file:///root/MOTION_CONTROL_BENCHMARK/robots/meshes/robotiq_gripper_with_tweezer/collision/robotiq_85_base_link_simplified.stl",
        "verts": mesh_data["vertices"],  # duplicates are removed in `mesh`, so I have to use the original data`
        "tris": mesh_data["faces"],
    },
    # {
    #     "name": "finger_left",
    #     "frame": 8,
    #     "rpy": [0, 0, 0],
    #     "xyz": [1, 0, 0],
    #     "mesh_file": "/root/MOTION_CONTROL_BENCHMARK/robots/meshes/robotiq_gripper_with_tweezer/collision/robotiq_85_finger_tip_link_left.stl",
    # },
    # {
    #     "name": "finger_right",
    #     "frame": 8,
    #     "rpy": [0, 0, 0],
    #     "xyz": [-1, 0, 0],
    #     "mesh_file": "/root/MOTION_CONTROL_BENCHMARK/robots/meshes/robotiq_gripper_with_tweezer/collision/robotiq_85_finger_tip_link_right.stl",
    # },
]
# <<<<< DEBUG


class Collision_Object_Container:
    def __init__(self, yaml_path):
        self.collision_objects = []
        f = open(yaml_path)
        y = yaml.load(f, yaml.SafeLoader)

        keys = y.keys()
        for k in keys:
            if y[k] is not None:
                if k == "robot_link_radius" or k == "sample_states" or k == "training_states" or k == "problem_states":
                    continue
                for i in range(len(y[k])):
                    if k == "boxes":
                        self.collision_objects.append(Collision_Box(y[k][i]))
                    elif k == "spheres":
                        self.collision_objects.append(Collision_Sphere(y[k][i]))
                    elif k == "ellipsoids":
                        self.collision_objects.append(Collision_Ellipsoid(y[k][i]))
                    elif k == "capsules":
                        self.collision_objects.append(Collision_Capsule(y[k][i]))
                    elif k == "cones":
                        self.collision_objects.append(Collision_Cone(y[k][i]))
                    elif k == "cylinders":
                        self.collision_objects.append(Collision_Cylinder(y[k][i]))
                    elif k == "meshes":
                        self.collision_objects.append(Collision_Mesh(y[k][i]))

                    self.collision_objects[-1].type = "environment_object"

        self.robot_link_radius = 0.05
        if "robot_link_radius" in keys:
            self.robot_link_radius = float(y["robot_link_radius"])

        if "sample_states" in keys:
            self.sample_states = y["sample_states"]
        else:
            raise Exception("Must specify at least one sample state in collision yaml file!")

        self.set_rviz_ids()

    def set_rviz_ids(self):
        i = 0
        for c in self.collision_objects:
            if type(c.marker) is list:
                for cc in c.marker:
                    cc.id = i
                    i += 1
            else:
                c.marker.id = i
                i += 1

    def get_min_distance(self, ab):
        a, b = ab
        obja = self.collision_objects[a].obj
        objb = self.collision_objects[b].obj

        self.request = fcl.DistanceRequest()
        self.result = fcl.DistanceResult()

        ret = fcl.distance(obja, objb, self.request, self.result)
        return self.result.min_distance

    ######################################################  Alban 2022-11-29
    def add_collision_objects_from_robot(self, robot, exclusion=[]):
        numDOF = robot.numDOF

        frames_list = robot.getFrames(numDOF * [0])

        # >>>>> DEBUG
        # TODO: this is not useful in this function, but good as a note for now
        transforms_wrt_panda_link = [np.eye(4)]  # adding the transform for panda_link0
        frames = frames_list[0]  # get the first arm's frames
        for i in range(len(frames[0]) - 1):  # remove the last one that is not a true joint
            transform = np.vstack(
                (np.hstack((np.array(frames[1][i]), np.array(frames[0][i]).reshape(-1, 1))), np.array([0, 0, 0, 1]))
            )
            transforms_wrt_panda_link.append(transform)

        # create a capsule
        i = 0
        for c_data in CAPSULE_DATA_PANDA:
            capsule = Collision_Capsule.init_with_arguments(
                "robotLink_" + str(0) + "_" + str(i),
                c_data["frame"],
                c_data["rpy"],  # rot
                c_data["xyz"],  # tran
                [c_data["radius"], c_data["length"]],
            )
            capsule.type = "robot_link"
            self.collision_objects.append(capsule)
            i += 1
        for m_data in MESHES_DATA_PANDA:
            mesh = Collision_Mesh.init_with_arguments(
                "robotLink_" + str(0) + "_" + str(i) + "_MESH",
                m_data["frame"],
                m_data["rpy"],  # rot
                m_data["xyz"],  # tran
                {"file": m_data["file"], "verts": m_data["verts"], "tris": m_data["tris"]},
            )
            mesh.type = "robot_link"
            self.collision_objects.append(mesh)
            i += 1
        # <<<<< DEBUG

        # for arm_idx in range(len(frames_list)):
        #     frames = frames_list[arm_idx]
        #     jtPts = frames[0]
        #     numLinks = len(jtPts) - 1
        #     for link_i in range(numLinks):
        #         curr_idx = numLinks * arm_idx + link_i
        #         if curr_idx not in exclusion:
        #             ptA = jtPts[link_i]
        #             ptB = jtPts[link_i + 1]
        #             midPt = ptA + 0.5 * (ptB - ptA)
        #             dis = np.linalg.norm(ptA - ptB)
        #             if dis < 0.02:
        #                 continue

        #             cylinder = Collision_Cylinder.init_with_arguments(
        #                 "robotLink_" + str(arm_idx) + "_" + str(link_i),
        #                 curr_idx,
        #                 [0, 0, 0],
        #                 midPt,
        #                 [self.robot_link_radius, dis],
        #             )
        #             cylinder.type = "robot_link"
        #             self.collision_objects.append(cylinder)

        #             sphere1 = Collision_Sphere.init_with_arguments(
        #                 "robotLink_" + str(arm_idx) + "_" + str(link_i) + "_up",
        #                 curr_idx,
        #                 [0, 0, 0],
        #                 ptA,
        #                 self.robot_link_radius,
        #             )
        #             sphere1.type = "robot_link"
        #             self.collision_objects.append(sphere1)

        #             sphere2 = Collision_Sphere.init_with_arguments(
        #                 "robotLink_" + str(arm_idx) + "_" + str(link_i) + "_down",
        #                 curr_idx,
        #                 [0, 0, 0],
        #                 ptB,
        #                 self.robot_link_radius,
        #             )
        #             sphere2.type = "robot_link"
        #             self.collision_objects.append(sphere2)

        self.set_rviz_ids()

    ######################################################

    def update_all_transforms(self, all_frames):

        positions = []
        rotations = []
        for f in all_frames:
            for i, p in enumerate(f[0]):
                positions.append(f[0][i])
                rotations.append(f[1][i])

        # >>>>> DEBUG
        # the transform of the robot base (panda_link0) is Identify
        positions = [np.array([0, 0, 0])]
        rotations = [np.eye(3)]
        frames = all_frames[0]
        for i in range(len(frames[0]) - 1):  # remove the last one that is not a true joint
            positions.append(frames[0][i])
            rotations.append(frames[1][i])
        # <<<<< DEBUG

        for c in self.collision_objects:
            if c.type == "robot_link":
                # name = c.name
                # name_arr = name.split("_")
                # arm_id = int(name_arr[1])
                # link_id = int(name_arr[2])
                # ptA = all_frames[arm_id][0][link_id]
                # ptB = all_frames[arm_id][0][link_id + 1]
                # midPt = ptA + 0.5 * (ptB - ptA)
                # if len(name_arr) > 3:  # this is a sphere used to create a capsule
                #     if name_arr[3] == "up":
                #         final_pos = ptA
                #     elif name_arr[3] == "down":
                #         final_pos = ptB
                #     else:
                #         print("Error in marker name")
                # else:
                #     final_pos = midPt

                # rot_mat = np.zeros((3, 3))
                # z = ptB - ptA
                # norm = max(np.linalg.norm(z), 0.000001)
                # z = (1.0 / norm) * z
                # up = np.array([0, 0, 1])
                # if np.dot(z, up) == 1.0:
                #     up = np.array([1, 0, 0])
                # x = np.cross(up, z)
                # y = np.cross(z, x)
                # rot_mat[:, 0] = x
                # rot_mat[:, 1] = y
                # rot_mat[:, 2] = z

                # final_quat = T.quaternion_from_matrix(rot_mat)
                coordinate_frame = c.coordinate_frame
                # first, do local transforms
                if coordinate_frame >= len(rotations):
                    rot_mat = rotations[-1]
                    final_pos = positions[-1]
                else:
                    rot_mat = rotations[coordinate_frame]
                    final_pos = positions[coordinate_frame]

                final_quat = T.quaternion_from_matrix(rot_mat)

                local_translation = np.array(c.translation)
                rotated_local_translation = np.dot(rot_mat, local_translation)
                final_pos = final_pos + rotated_local_translation

                local_rotation = c.quaternion
                final_quat = T.quaternion_multiply(local_rotation, final_quat)
            else:
                coordinate_frame = c.coordinate_frame
                # first, do local transforms
                frame_len = len(positions)
                if coordinate_frame == 0:
                    rot_mat = rotations[0]
                    final_pos = positions[0]
                elif coordinate_frame >= frame_len:
                    rot_mat = rotations[frame_len - 1]
                    final_pos = positions[frame_len - 1]
                else:
                    rot_mat = rotations[coordinate_frame]
                    final_pos = positions[coordinate_frame - 1]

                final_quat = T.quaternion_from_matrix(rot_mat)

                local_translation = np.array(c.translation)
                rotated_local_translation = np.dot(rot_mat, local_translation)
                final_pos = final_pos + rotated_local_translation

                local_rotation = c.quaternion
                final_quat = T.quaternion_multiply(local_rotation, final_quat)

            c.update_transform(final_pos, final_quat)

    def draw_all(self):
        for c in self.collision_objects:
            if c.__class__ == Collision_Mesh:
                continue
            else:
                c.draw_rviz()

    def __str__(self):
        return str([str(c.name) for c in self.collision_objects])


class Collision_Object:
    def __init__(self, collision_dict):
        self.name = collision_dict["name"]
        self.coordinate_frame = collision_dict["coordinate_frame"]
        self.rotation = collision_dict["rotation"]
        rx, ry, rz = self.rotation[0], self.rotation[1], self.rotation[2]
        self.quaternion = T.quaternion_from_euler(rx, ry, rz)
        self.translation = collision_dict["translation"]
        self.params = collision_dict["parameters"]
        self.id = 0
        self.type = ""
        self.pub = rospy.Publisher("visualization_marker", Marker, queue_size=10)
        self.make_rviz_marker_super()

    @classmethod
    def init_with_arguments(self, name, coordinate_frame, rotation, translation, params):
        collision_dict = {
            "name": name,
            "coordinate_frame": coordinate_frame,
            "rotation": rotation,
            "translation": translation,
            "parameters": params,
        }
        return self(collision_dict)

    def update_transform(self, translation, rotation):
        if len(translation) == 1:
            translation = translation[0]
        self.t = fcl.Transform(rotation, translation)
        self.obj.setTransform(self.t)
        if type(self) == Collision_Capsule:
            # cylinder
            self.marker[0].pose.position.x = translation[0]
            self.marker[0].pose.position.y = translation[1]
            self.marker[0].pose.position.z = translation[2]
            self.marker[0].pose.orientation.w = rotation[0]
            self.marker[0].pose.orientation.x = rotation[1]
            self.marker[0].pose.orientation.y = rotation[2]
            self.marker[0].pose.orientation.z = rotation[3]
            # up sphere
            # TODO set proper position
            rotation_matrix = T.quaternion_matrix(rotation)[:3, :3]
            up_translaton = np.dot(rotation_matrix, np.array([0, 0, self.lz / 2]).reshape(-1, 1))
            self.marker[1].pose.position.x = translation[0] + up_translaton[0]
            self.marker[1].pose.position.y = translation[1] + up_translaton[1]
            self.marker[1].pose.position.z = translation[2] + up_translaton[2]
            self.marker[1].pose.orientation.w = rotation[0]
            self.marker[1].pose.orientation.x = rotation[1]
            self.marker[1].pose.orientation.y = rotation[2]
            self.marker[1].pose.orientation.z = rotation[3]
            # down sphere
            # TODO set proper position
            down_translaton = np.dot(rotation_matrix, np.array([0, 0, -self.lz / 2]).reshape(-1, 1))
            self.marker[2].pose.position.x = translation[0] + down_translaton[0]
            self.marker[2].pose.position.y = translation[1] + down_translaton[1]
            self.marker[2].pose.position.z = translation[2] + down_translaton[2]
            self.marker[2].pose.orientation.w = rotation[0]
            self.marker[2].pose.orientation.x = rotation[1]
            self.marker[2].pose.orientation.y = rotation[2]
            self.marker[2].pose.orientation.z = rotation[3]
        elif type(self) == Collision_Mesh:
            self.marker.pose.position.x = 0.0
            self.marker.pose.position.y = 0.0
            self.marker.pose.position.z = 0.0
            self.marker.pose.orientation.w = 1.0
            self.marker.pose.orientation.x = 0.0
            self.marker.pose.orientation.y = 0.0
            self.marker.pose.orientation.z = 0.0
        else:
            self.marker.pose.position.x = translation[0]
            self.marker.pose.position.y = translation[1]
            self.marker.pose.position.z = translation[2]
            self.marker.pose.orientation.w = rotation[0]
            self.marker.pose.orientation.x = rotation[1]
            self.marker.pose.orientation.y = rotation[2]
            self.marker.pose.orientation.z = rotation[3]

    def update_rviz_color(self, r, g, b, a):
        if type(self) == Collision_Capsule:
            for c in self.marker:
                c.color.r = r
                c.color.g = g
                c.color.b = b
                c.color.a = a
        else:
            self.marker.color.r = r
            self.marker.color.g = g
            self.marker.color.b = b
            self.marker.color.a = a

    def make_rviz_marker_super(self):
        if type(self) == Collision_Capsule:
            self.marker = []
            # cylinder
            marker1 = Marker()
            marker1.header.frame_id = "common_world"
            marker1.header.stamp = rospy.Time()
            marker1.id = self.id
            marker1.color.a = 0.4
            marker1.color.g = 1.0
            marker1.color.b = 0.7
            marker1.text = self.name
            self.marker.append(marker1)
            # up sphere
            marker2 = Marker()
            marker2.header.frame_id = "common_world"
            marker2.header.stamp = rospy.Time()
            marker2.id = self.id
            marker2.color.a = 0.4
            marker2.color.g = 1.0
            marker2.color.b = 0.7
            marker2.text = self.name
            self.marker.append(marker2)
            # down sphere
            marker3 = Marker()
            marker3.header.frame_id = "common_world"
            marker3.header.stamp = rospy.Time()
            marker3.id = self.id
            marker3.color.a = 0.4
            marker3.color.g = 1.0
            marker3.color.b = 0.7
            marker3.text = self.name
            self.marker.append(marker3)
        else:
            self.marker = Marker()
            self.marker.header.frame_id = "common_world"
            self.marker.header.stamp = rospy.Time()
            self.marker.id = self.id
            self.marker.color.a = 0.4
            self.marker.color.g = 1.0
            self.marker.color.b = 0.7
            self.marker.text = self.name

    def make_rviz_marker(self):
        pass

    def draw_rviz(self):
        if type(self) == Collision_Capsule:
            for c in self.marker:
                c.header.stamp.secs = rospy.get_rostime().secs
                c.header.stamp.nsecs = rospy.get_rostime().nsecs
                self.pub.publish(c)
        elif type(self) == Collision_Mesh:
            self.marker.header.frame_id = "common_world"
            self.marker.header.stamp = rospy.Time()
            self.marker.id = 27
            self.marker.color.a = 0.4
            self.marker.color.g = 1.0
            self.marker.color.b = 0.7
            self.marker.text = "test_shape"
            self.marker.type = self.marker.MESH_RESOURCE
            self.marker.mesh_resource = "file:///root/MOTION_CONTROL_BENCHMARK/robots/meshes/robotiq_gripper_with_tweezer/collision/robotiq_85_base_link_simplified.stl"
            self.marker.mesh_use_embedded_materials = False
            self.marker.scale.x = 3
            self.marker.scale.y = 3
            self.marker.scale.z = 3
            self.marker.pose.position.x = 0.0
            self.marker.pose.position.y = 0.0
            self.marker.pose.position.z = 0.0
            self.marker.pose.orientation.w = 1.0
            self.marker.pose.orientation.x = 0.0
            self.marker.pose.orientation.y = 0.0
            self.marker.pose.orientation.z = 0.0
            self.marker.header.stamp.secs = rospy.get_rostime().secs
            self.marker.header.stamp.nsecs = rospy.get_rostime().nsecs
            self.pub.publish(self.marker)
        else:
            self.marker.header.stamp.secs = rospy.get_rostime().secs
            self.marker.header.stamp.nsecs = rospy.get_rostime().nsecs
            self.pub.publish(self.marker)


class Collision_Box(Collision_Object):
    def __init__(self, collision_dict):
        Collision_Object.__init__(self, collision_dict)
        if not len(self.params) == 3:
            raise TypeError(bc.FAIL + "ERROR: parameters for collision box must be list of 3 floats." + bc.ENDC)

        self.x = self.params[0]
        self.y = self.params[1]
        self.z = self.params[2]
        self.g = fcl.Box(self.x, self.y, self.z)
        self.t = fcl.Transform()
        self.obj = fcl.CollisionObject(self.g, self.t)
        self.make_rviz_marker()

    def make_rviz_marker(self):
        self.marker.type = self.marker.CUBE
        self.marker.scale.x = self.x
        self.marker.scale.y = self.y
        self.marker.scale.z = self.z


class Collision_Sphere(Collision_Object):
    def __init__(self, collision_dict):
        Collision_Object.__init__(self, collision_dict)
        if not type(self.params) == float:
            raise TypeError(
                bc.FAIL + "ERROR: parameters for collision sphere must be a float value (for radius)." + bc.ENDC
            )

        self.r = self.params
        self.g = fcl.Sphere(self.r)
        self.t = fcl.Transform()
        self.obj = fcl.CollisionObject(self.g, self.t)
        self.make_rviz_marker()

    def make_rviz_marker(self):
        self.marker.type = self.marker.SPHERE
        self.marker.scale.x = self.r * 2
        self.marker.scale.y = self.r * 2
        self.marker.scale.z = self.r * 2


class Collision_Ellipsoid(Collision_Object):
    def __init__(self, collision_dict):
        Collision_Object.__init__(self, collision_dict)
        if not len(self.params) == 3:
            raise TypeError(bc.FAIL + "ERROR: parameters for collision ellipsoid must be list of 3 floats." + bc.ENDC)

        self.x, self.y, self.z = self.params[0], self.params[1], self.params[2]
        self.g = fcl.Ellipsoid(self.x, self.y, self.z)
        self.t = fcl.Transform()
        self.obj = fcl.CollisionObject(self.g, self.t)
        self.make_rviz_marker()

    def make_rviz_marker(self):
        self.marker.type = self.marker.SPHERE
        self.marker.scale.x = self.x * 2
        self.marker.scale.y = self.y * 2
        self.marker.scale.z = self.z * 2


class Collision_Capsule(Collision_Object):
    def __init__(self, collision_dict):
        Collision_Object.__init__(self, collision_dict)
        if not len(self.params) == 2:
            raise TypeError(bc.FAIL + "ERROR: parameters for collision capsule must be list of 2 floats." + bc.ENDC)

        self.r, self.lz = self.params[0], self.params[1]
        self.g = fcl.Capsule(self.r, self.lz)
        self.t = fcl.Transform()  # not the proper pose, but it will be updates later anyway
        self.obj = fcl.CollisionObject(self.g, self.t)
        self.make_rviz_marker()

    def make_rviz_marker(self):
        # cylinder
        self.marker[0].type = self.marker[0].CYLINDER
        self.marker[0].scale.x = self.r * 2
        self.marker[0].scale.y = self.r * 2
        self.marker[0].scale.z = self.lz
        # up sphere
        self.marker[1].type = self.marker[1].SPHERE
        self.marker[1].scale.x = self.r * 2
        self.marker[1].scale.y = self.r * 2
        self.marker[1].scale.z = self.r * 2
        # cylinder
        self.marker[2].type = self.marker[2].SPHERE
        self.marker[2].scale.x = self.r * 2
        self.marker[2].scale.y = self.r * 2
        self.marker[2].scale.z = self.r * 2


class Collision_Cone(Collision_Object):
    def __init__(self, collision_dict):
        Collision_Object.__init__(self, collision_dict)
        if not len(self.params) == 2:
            raise TypeError(bc.FAIL + "ERROR: parameters for collision cone must be list of 2 floats." + bc.ENDC)

        self.r, self.lz = self.params[0], self.params[1]
        self.g = fcl.Capsule(self.r, self.lz)
        self.t = fcl.Transform()
        self.obj = fcl.CollisionObject(self.g, self.t)
        self.make_rviz_marker()

    def make_rviz_marker(self):
        self.marker.type = self.marker.CYLINDER
        self.marker.scale.x = self.r * 2
        self.marker.scale.y = self.r * 2
        self.marker.scale.z = self.lz


class Collision_Cylinder(Collision_Object):
    def __init__(self, collision_dict):
        Collision_Object.__init__(self, collision_dict)
        if not len(self.params) == 2:
            raise TypeError(bc.FAIL + "ERROR: parameters for collision cylinder must be list of 2 floats." + bc.ENDC)

        self.r, self.lz = self.params[0], self.params[1]
        self.g = fcl.Cylinder(self.r, self.lz)
        self.t = fcl.Transform()
        self.obj = fcl.CollisionObject(self.g, self.t)
        self.make_rviz_marker()

    def make_rviz_marker(self):
        self.marker.type = self.marker.CYLINDER
        self.marker.scale.x = self.r * 2
        self.marker.scale.y = self.r * 2
        self.marker.scale.z = self.lz


class Collision_Mesh(Collision_Object):
    def __init__(self, collision_dict):
        Collision_Object.__init__(self, collision_dict)
        if not len(self.params) == 3:
            raise TypeError(
                bc.FAIL
                + "ERROR: parameters for collision mesh must be a dictionary consisting of a file_path, a list of verts and a list of tris."
                + bc.ENDC
            )

        # if not len(self.params["verts"]) == len(self.params["tris"]):
        #     raise TypeError(
        #         bc.FAIL + "ERROR: number of tris must equal the number of verts in collision mesh." + bc.ENDC
        #     )

        self.file = self.params["file"]
        self.verts = np.array(self.params["verts"])
        self.tris = np.array(self.params["tris"])

        self.g = fcl.BVHModel()
        self.g.beginModel(len(self.verts), len(self.tris))
        self.g.addSubModel(self.verts, self.tris)
        self.g.endModel()
        self.t = fcl.Transform()
        self.obj = fcl.CollisionObject(self.g, self.t)
        self.make_rviz_marker()

    def make_rviz_marker(self):
        # # print(bc.WARNING + "WARNING: Mesh collision object not supported in rviz visualization" + bc.ENDC)
        # self.marker.type = self.marker.MESH_RESOURCE
        # self.marker.mesh_resource = self.file
        # self.marker.mesh_use_embedded_materials = False
        # self.marker.scale.x = 1
        # self.marker.scale.y = 1
        # self.marker.scale.z = 1

        self.marker.header.frame_id = "common_world"
        self.marker.header.stamp = rospy.Time()
        self.marker.id = 27
        self.marker.color.a = 0.4
        self.marker.color.g = 1.0
        self.marker.color.b = 0.7
        self.marker.text = "test_shape"
        self.marker.type = self.marker.MESH_RESOURCE
        self.marker.mesh_resource = "file:///root/MOTION_CONTROL_BENCHMARK/robots/meshes/robotiq_gripper_with_tweezer/collision/robotiq_85_base_link_simplified.stl"
        self.marker.mesh_use_embedded_materials = False
        self.marker.scale.x = 3
        self.marker.scale.y = 3
        self.marker.scale.z = 3
        self.marker.pose.position.x = 0.0
        self.marker.pose.position.y = 0.0
        self.marker.pose.position.z = 0.0
        self.marker.pose.orientation.w = 1.0
        self.marker.pose.orientation.x = 0.0
        self.marker.pose.orientation.y = 0.0
        self.marker.pose.orientation.z = 0.0
