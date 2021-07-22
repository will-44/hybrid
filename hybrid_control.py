# ############################################
# An example for doing impedence control
#
# Author : Deepak Raina @ IIT Delhi
# Version : 0.1
# ############################################

import numpy as np
from pybullet_controller import RobotController

robot = RobotController(robot_type='ur5')
robot.createWorld(GUI=True)

# Impedence controller
# Input: numpy array of joint angles
thi = np.array([0, -1.0, 1.0, -1.57, -1.57, -1.57]) # initial joint angles
thf = np.array([-1.5, -1.0, 1.0, -1.57, -1.57, -1.57]) # final joint nagles
robot.setJointPosition(thi)
desired_pose = np.array([0.40, 0, 0.75, 0, 0, 0]) #ur5-link
desired_force = np.array([0, 0, 2, 0, 0, 0])
robot.hybridController(thi, desired_pose, desired_force, controller_gain=100, kpp = 1, kpi = 0.0, kpd = 0.0)






