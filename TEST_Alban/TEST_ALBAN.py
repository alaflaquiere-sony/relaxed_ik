from urdfpy import URDF

robot = URDF.load("/root/catkin_ws/src/relaxed_ik/TEST_Alban/panda_with_tweezer_absolute.urdf")

print(robot.links)
