from urdfpy import URDF

robot = URDF.load("/root/catkin_ws/src/relaxed_ik/TEST_Alban/panda_with_tweezer_absolute.urdf")

# list links
print("\n## LINKS")
for link in robot.links:
    print(link.name)

# list joints
print("\n## JOINTS")
for joint in robot.joints:
    print("{} connects {} <---> {}".format(joint.name, joint.parent, joint.child))

# list actuated joints
print("\n## ACTUATED JOINTS")
for joint in robot.actuated_joints:
    print(joint.name)

# robot base
print("\n## ROBOT BASE")
print(robot.base_link.name)

# forward kinematics of links
print("\n## FK LiNKS")
# cfg = {
#     "panda_joint1": -0.27002835999999997,
#     "panda_joint2": 0.90184848,
#     "panda_joint3": -1.39707806,
#     "panda_joint4": -2.6923472,
#     "panda_joint5": 2.14516092,
#     "panda_joint6": 0.560441,
#     "panda_joint7": 1.8154481800000002,
# }
# fk = robot.link_fk(cfg=cfg)
fk = robot.link_fk()
for link in robot.links:
    print(f"{link.name}:\n{fk[link]}")

# display
# robot.show(cfg=cfg, use_collision=True)  # requires trimesh==3.7.13
