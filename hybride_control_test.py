# ############################################
# An example for doing impedence control
#
# Author : Deepak Raina @ IIT Delhi
# Version : 0.1
# ############################################

import numpy as np
from pybullet_controller import RobotController
import pybullet as p
import time
from scipy.spatial.transform import Rotation as R

robot = RobotController(robot_type='ur5', end_eff_index=4)
robot.createWorld(GUI=True)

lecube = p.loadURDF("urdf/cube.urdf", useFixedBase=True)

p.setRealTimeSimulation(False)
# forward dynamics simulation loop
# for turning off link and joint damping
initPos = np.array([0, 0, -1, 1,0,0])
for link_idx in range(len(initPos)):
    p.resetJointState(robot.robot_id, link_idx, initPos[link_idx])

for link_idx in range(robot.num_joints+1):
    p.changeDynamics(robot.robot_id, link_idx, linearDamping=0.0, angularDamping=0.0, jointDamping=0.0)
    p.changeDynamics(robot.robot_id, link_idx, maxJointVelocity=200)

# # Enable torque control
p.setJointMotorControlArray(robot.robot_id, robot.controllable_joints,
                            p.POSITION_CONTROL,
                            forces=np.zeros(len(robot.controllable_joints)))

kd = 0.7 # from URDF file
Kp = 100
Kd = 2 * np.sqrt(Kp)
Md = 0.1*np.eye(6)

# Target position and velcoity
xd_list = [np.array([0.795, -0.198, 1.091, -1.5708, 0, 0]),
           np.array([0.795, 0.198, 1.091, -1.5708, 0, 0]),
           np.array([0.495, 0.198, 1.091, -1.5708, 0, 0]),
           np.array([0.495, -0.198, 1.091, -1.5708, 0, 0])]
index = 0
xd = xd_list[index]
dxd = np.zeros(6)
# define GUI sliders

ForceInitial = np.zeros(len(robot.controllable_joints))

last_ZForces = np.zeros(50)

C = np.asarray([(1, 0, 0, 0, 0, 0),
                (0, 1, 0, 0, 0, 0),
                (0, 0, 0, 0, 0, 0),
                (0, 0, 0, 0, 0, 0),
                (0, 0, 0, 0, 0, 0),
                (0, 0, 0, 0, 0, 0)])
sum_fe = 0.0

last_fe = np.zeros(50)

while True:

    #check for contact
    contact = p.getContactPoints(lecube)
    zForce = 0.0
    for items in contact:
        zForce += items[9]
    np.roll(last_ZForces, 1)
    last_ZForces[0] = zForce
    zForce = last_ZForces.mean()
    print("Force = " + str(zForce))

    # get current joint states
    q, dq, _ = robot.getJointStates()

    # Error in task space
    x = robot.solveForwardPositonKinematics(q)
    if np.linalg.norm(x[:2] - xd[:2]) < 0.1:
        index += 1
        if index > 3:
            index = 0
        xd = xd_list[index]

    x_e = xd - x
    x_e = np.dot(x_e, C)
    dx = robot.solveForwardVelocityKinematics(q, dq)
    dx_e = dxd - dx
    dx_e = np.dot(dx_e, C)

    # Task space dynamics
    # Jacobian
    J = robot.getJacobian(q)
    J_inv = np.linalg.pinv(J)
    # Inertia matrix in the joint space
    Mq, G, Cqq = robot.calculateDynamicMatrices()
    # Inertia matrix in the task space
    Mx = np.dot(np.dot(np.transpose(J_inv), Mq), J_inv)
    # Force in task space
    Fx = np.dot(np.dot(np.linalg.inv(Md), Mx), (np.dot(Kp, x_e) + np.dot(Kd, dx_e)))
    # External Force applied
    # F_w_ext = np.dot((np.dot(np.linalg.inv(Md), Mx) - np.eye(6)), np.array([0,0,zForce-10,0,0,0]))
    # Fx += F_w_ext
    # Force in joint space
    Fq = np.dot(np.transpose(J), Fx)

    fd = 2.0
    fe = fd-zForce
    np.roll(last_fe, 1)
    last_fe[0] = fe
    fkp = 10
    fki = 10
    sum_fe = np.mean(last_fe)
    fz = fkp*(fe) + fki*sum_fe

    quaternion = p.getLinkState(robot.robot_id, 4)[1]

    test = np.array(p.getMatrixFromQuaternion(quaternion))
    test = np.reshape(test, (3, 3))

    fd = np.dot(test, np.asarray((0, 0, -50)))

    fd = np.array([fd[0], fd[1], fd[2], 0, 0, 0])

    tes = np.dot(J_inv, fd)

    # Controlled Torque
    tau = G + Fq + Cqq + tes
    # tau += kd * np.asarray(dq) # if joint damping is turned off, this torque will not be required
    # print('tau:', tau)

    # Activate torque control
    p.setJointMotorControlArray(robot.robot_id, robot.controllable_joints,
                                controlMode=p.TORQUE_CONTROL,
                                forces=tau)

    p.stepSimulation()
    time.sleep(robot.time_step)
p.disconnect()