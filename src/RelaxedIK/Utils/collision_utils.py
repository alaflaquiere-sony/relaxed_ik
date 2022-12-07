import fcl
import numpy as np
import rospy
import trimesh
import yaml
from visualization_msgs.msg import Marker, MarkerArray

import RelaxedIK.Utils.transformations as T
from RelaxedIK.Utils.colors import bcolors as bc

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
    {
        "name": "temp_gripper",
        "frame": 10,
        "rpy": [0, 0, 0],
        "xyz": [0.1, 0, 0],
        "length": 0.1,
        "radius": 0.02,
    },
]
mesh_gripper_base = trimesh.exchange.stl.load_stl(
    open(
        "/root/MOTION_CONTROL_BENCHMARK/robots/meshes/robotiq_gripper_with_tweezer/collision/robotiq_85_base_link_simplified.stl",
        "rb",
    )
)
mesh_finger_left = trimesh.exchange.stl.load_stl(
    open(
        "/root/MOTION_CONTROL_BENCHMARK/robots/meshes/robotiq_gripper_with_tweezer/collision/robotiq_85_finger_tip_link_left.stl",
        "rb",
    )
)
mesh_finger_right = trimesh.exchange.stl.load_stl(
    open(
        "/root/MOTION_CONTROL_BENCHMARK/robots/meshes/robotiq_gripper_with_tweezer/collision/robotiq_85_finger_tip_link_left.stl",
        "rb",
    )
)
MESHES_DATA_PANDA = [
    {
        "name": "robotiq_85_base_link",
        "frame": 10,
        "rpy": [0, 0, 0],
        "xyz": [0, 0, 0],
        "file": "file:///root/MOTION_CONTROL_BENCHMARK/robots/meshes/robotiq_gripper_with_tweezer/collision/robotiq_85_base_link_simplified.stl",
        "verts": mesh_gripper_base["vertices"],
        "tris": mesh_gripper_base["faces"],
    },
    {
        "name": "finger_left",
        "frame": 10,
        "rpy": [3.14159265, 0.0, 3.14159265],
        "xyz": [1.04459598e-01, 5.02994082e-02, 7.77642643e-15],
        "file": "file:///root/MOTION_CONTROL_BENCHMARK/robots/meshes/robotiq_gripper_with_tweezer/collision/robotiq_85_finger_tip_link_left.stl",
        "verts": mesh_finger_left["vertices"],
        "tris": mesh_finger_left["faces"],
    },
    {
        "name": "finger_right",
        "frame": 10,
        "rpy": [0.0, 0.0, -3.14159265],
        "xyz": [0.1044596, -0.05029941, 0.0],
        "file": "file:///root/MOTION_CONTROL_BENCHMARK/robots/meshes/robotiq_gripper_with_tweezer/collision/robotiq_85_finger_tip_link_right.stl",
        "verts": mesh_finger_right["vertices"],
        "tris": mesh_finger_right["faces"],
    },
]


class Collision_Object_Container:
    def __init__(self, yaml_path):
        self.collision_objects = []
        f = open(yaml_path)
        y = yaml.load(f, yaml.SafeLoader)
        self.markers_collection = MarkerArray()
        self.pub = rospy.Publisher("visualization_marker_array", MarkerArray, queue_size=10)

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
        for i, m in enumerate(self.markers_collection.markers):
            m.id = i

    def get_min_distance(self, ab):
        a, b = ab
        obja = self.collision_objects[a].obj
        objb = self.collision_objects[b].obj

        self.request = fcl.DistanceRequest()
        self.result = fcl.DistanceResult()

        _ = fcl.distance(obja, objb, self.request, self.result)
        return self.result.min_distance

    def add_collision_objects_from_robot(self, robot, exclusion=[]):

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
            for marker in capsule.marker:
                self.markers_collection.markers.append(marker)
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
            self.markers_collection.markers.append(mesh.marker)
            i += 1

        self.set_rviz_ids()

    def update_all_transforms(self, all_frames):

        arm_idx = 0
        positions = all_frames[arm_idx][0]
        rotations = all_frames[arm_idx][1]

        # the transform of the robot base (panda_link0) is Identify
        positions.insert(0, np.array([0, 0, 0]))
        rotations.insert(0, np.eye(3))

        for c in self.collision_objects:
            coordinate_frame = c.coordinate_frame

            rot_mat = rotations[coordinate_frame]
            final_pos = positions[coordinate_frame]

            local_rotation = c.quaternion
            final_quat = T.quaternion_from_matrix(rot_mat)
            final_quat = T.quaternion_multiply(final_quat, local_rotation)

            local_translation = np.array(c.translation)
            rotated_local_translation = np.dot(rot_mat, local_translation)
            final_pos = final_pos + rotated_local_translation

            c.update_transform(final_pos, final_quat)

    def draw_all(self):
        for c in self.collision_objects:
            c.draw_rviz()
        self.pub.publish(self.markers_collection)

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
        # self.pub = rospy.Publisher("visualization_marker", Marker, queue_size=10)
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
            down_translaton = np.dot(rotation_matrix, np.array([0, 0, -self.lz / 2]).reshape(-1, 1))
            self.marker[2].pose.position.x = translation[0] + down_translaton[0]
            self.marker[2].pose.position.y = translation[1] + down_translaton[1]
            self.marker[2].pose.position.z = translation[2] + down_translaton[2]
            self.marker[2].pose.orientation.w = rotation[0]
            self.marker[2].pose.orientation.x = rotation[1]
            self.marker[2].pose.orientation.y = rotation[2]
            self.marker[2].pose.orientation.z = rotation[3]
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
        else:
            self.marker.header.stamp.secs = rospy.get_rostime().secs
            self.marker.header.stamp.nsecs = rospy.get_rostime().nsecs


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
        self.marker.type = self.marker.MESH_RESOURCE
        self.marker.mesh_resource = self.file
        self.marker.mesh_use_embedded_materials = False
        self.marker.scale.x = 1
        self.marker.scale.y = 1
        self.marker.scale.z = 1
