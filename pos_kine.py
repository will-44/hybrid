# ############################################
# An example for doing position kinematics
#
# Author : Deepak Raina @ IIT Delhi
# Version : 0.1
# ############################################

import numpy as np
from pybullet_controller import RobotController

robot = RobotController(robot_type='ur5')
robot.createWorld(GUI=False)

# forward-inverse kinematics solver
# Input: numpy array of joint angles
joint_angles = np.array([0, -1.0, 1.0, -1.57, -1.57, -1.57])
robot.setJointPosition(joint_angles)
end_eff_pose = robot.solveForwardPositonKinematics(joint_angles)
joint_angles = robot.solveInversePositionKinematics(end_eff_pose)


