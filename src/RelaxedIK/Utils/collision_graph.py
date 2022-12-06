import itertools
import math

import numpy as np

from RelaxedIK.Utils.collision_utils import Collision_Object_Container

# #
# pass
# import numpy as np
# from scipy.spatial.transform import Rotation

# R_right_inner_knuckle = np.vstack(
#     (
#         np.hstack(
#             (
#                 Rotation.from_euler("xyz", [0, 0, 0]).as_matrix(),
#                 np.array([0.06142, -0.0127, 0]).reshape(-1, 1),
#             )
#         ),
#         np.array([0, 0, 0, 1]),
#     )
# )

# R_left_inner_knuckle = np.vstack(
#     (
#         np.hstack(
#             (
#                 Rotation.from_euler("xyz", [3.14159265359, 0.0, 0.0]).as_matrix(),
#                 np.array([0.06142, 0.0127, 0]).reshape(-1, 1),
#             )
#         ),
#         np.array([0, 0, 0, 1]),
#     )
# )

# R_right_finger_tip = np.vstack(
#     (
#         np.hstack(
#             (
#                 Rotation.from_euler("xyz", [0, 0, 3.14159265359]).as_matrix(),
#                 np.array([0.04303959807, -0.03759940821, 0]).reshape(-1, 1),
#             )
#         ),
#         np.array([0, 0, 0, 1]),
#     )
# )

# R_left_finger_tip = np.vstack(
#     (
#         np.hstack(
#             (
#                 Rotation.from_euler("xyz", [0, 0, 3.14159265359]).as_matrix(),
#                 np.array([0.04303959807, -0.03759940821, 0.0]).reshape(-1, 1),
#             )
#         ),
#         np.array([0, 0, 0, 1]),
#     )
# )

# bigR_right = R_right_inner_knuckle @ R_right_finger_tip
# Rotation.from_matrix(bigR_right[:3, :3]).as_euler("xyz")
# bigR_right[:3, -1]

# bigR_left = R_left_inner_knuckle @ R_left_finger_tip
# Rotation.from_matrix(bigR_left[:3, :3]).as_euler("xyz")
# bigR_left[:3, -1]
# #


class Collision_Graph:
    def __init__(self, yaml_path, robot, link_exclusion_list=[]):
        self.c = Collision_Object_Container(yaml_path)
        self.c.add_collision_objects_from_robot(robot, link_exclusion_list)
        self.robot = robot
        self.sample_states = self.c.sample_states
        self.num_objects = len(self.c.collision_objects)
        # self.b_value = 1.0/self.num_objects
        self.b_value = 5.0
        self.original_distances = 10000 * np.ones((self.num_objects, self.num_objects))
        self.combinations = list(itertools.combinations(range(self.num_objects), r=2))
        self.collision_color_array = self.num_objects * [0]
        self.danger_dis = 0.1

        self.initialize_table()
        self.c_values = self.get_c_values(self.original_distances)

    def get_collision_score_of_state(self, state):
        frames = self.robot.getFrames(state)
        return self.get_collision_score(frames)

    def get_collision_score(self, frames):
        sum = 0.0
        self.c.update_all_transforms(frames)
        c_values = self.c_values
        self.collision_color_array = self.num_objects * [0]
        for i, pair in enumerate(self.combinations):
            l1 = pair[0]
            l2 = pair[1]

            dis = self.c.get_min_distance(pair)
            if dis == -1:
                dis = 0.0

            c = c_values[l1, l2]
            if not c == 0.0:
                val = self.b_value * (math.e ** ((-((dis) ** 4.0)) / (2.0 * c**2)))
                if val > self.b_value / 2.0:
                    self.c.collision_objects[l1].update_rviz_color(1.0, 0, 0, 0.4)
                    self.c.collision_objects[l2].update_rviz_color(1.0, 0, 0, 0.4)
                    self.collision_color_array[l1] = 2
                    self.collision_color_array[l2] = 2
                elif val > self.b_value / 2.0 and val < self.b_value:
                    if not self.collision_color_array[l1] > 1:
                        self.c.collision_objects[l1].update_rviz_color(1.0, 1.0, 0.0, 0.4)
                        self.collision_color_array[l1] = 1
                    if not self.collision_color_array[l2] > 1:
                        self.c.collision_objects[l2].update_rviz_color(1.0, 1.0, 0.0, 0.4)
                        self.collision_color_array[l2] = 1
                else:
                    if not self.collision_color_array[l1] > 0:
                        self.c.collision_objects[l1].update_rviz_color(0, 1.0, 0.7, 0.4)
                    if not self.collision_color_array[l2] > 0:
                        self.c.collision_objects[l2].update_rviz_color(0, 1.0, 0.7, 0.4)

                sum += val

        return sum

    def initialize_table(self):
        for s in self.sample_states:
            frames = self.robot.getFrames(s)
            self.c.update_all_transforms(frames)
            for pair in self.combinations:
                a = pair[0]
                b = pair[1]
                dis = self.c.get_min_distance(pair)
                if dis == -1:
                    dis = 0.0
                if dis < self.original_distances[a, b]:
                    self.original_distances[a, b] = dis
                    self.original_distances[b, a] = dis

            for pair in self.combinations:
                a = pair[0]
                b = pair[1]
                if not self.original_distances[a, b] == 0.0:
                    self.original_distances[a, b] = min(self.danger_dis, self.original_distances[a, b])
                if not self.original_distances[b, a] == 0.0:
                    self.original_distances[b, a] = min(self.danger_dis, self.original_distances[b, a])

    def get_c_values(self, original_distances):
        shape = self.original_distances.shape
        c_values = np.zeros(shape)
        for i in range(shape[0]):
            for j in range(shape[1]):
                if original_distances[i, j] <= 0.001:
                    c_values[i, j] = 0.0
                else:
                    c_values[i, j] = self.get_c_value_from_dis(original_distances[i, j], self.b_value)
        return c_values

    def get_c_value_from_dis(self, dis, b, v=1.0e-15):
        return math.sqrt(-((dis) ** 4) / (2.0 * math.log(v / b)))
