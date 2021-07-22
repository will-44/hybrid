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

robot = RobotController(robot_type='ur5')
robot.createWorld(GUI=True)

# Impedence controller
# Input: numpy array of joint angles
initPos = np.array([0, 0, -1, 1, -1, -1.5, 1])
for link_idx in range(len(initPos)):
    p.resetJointState(robot.robot_id, link_idx, initPos[link_idx])

thf = np.array([-1.5, -1.0, 1.0, -1.57, -1.57, -1.57]) # final joint nagles
desired_pose = np.array([0.595, -0.198, 1.091, -1.5708, 0, 0])
desired_force = np.array([0, 0, 0, 0, 0, 0])

p.setRealTimeSimulation(False)
# forward dynamics simulation loop
# for turning off link and joint damping
for link_idx in range(robot.num_joints + 1):
    p.changeDynamics(robot.robot_id, link_idx, linearDamping=0.0, angularDamping=0.0, jointDamping=0.0)
    p.changeDynamics(robot.robot_id, link_idx, maxJointVelocity=200)

# Enable torque control
p.setJointMotorControlArray(robot.robot_id, robot.controllable_joints,
                            p.VELOCITY_CONTROL,
                            forces=np.zeros(len(robot.controllable_joints)))

# definition des variables

qcx = 0  # position (x,y,z) apres l'operation avec C
C = np.asarray([(1, 0, 0, 0, 0, 0),
                (0, 1, 0, 0, 0, 0),
                (0, 0, 0, 0, 0, 0),
                (0, 0, 0, 1, 0, 0),
                (0, 0, 0, 0, 1, 0),
                (0, 0, 0, 0, 0, 1)])

q_prePI_somme = 0
qr_old = 0

kpp = 1
kpi = 0
kpd = 0

while True:
    # position (x,y,z) desiree: matrice(6*6)
    Xd = desired_pose + qcx
    print("Xd", Xd)
    # input()
    # position des joint desire suivant Xd:  matrice(6*6)
    qd = robot.solveInversePositionKinematics(Xd)
    qd = np.asarray(qd)
    print("qd", qd)
    # position  et vitesse des joint actuel:  matrice(6*6)
    qa, dqa, _ = robot.getJointStates()
    # Transfomation en matrice:
    qa = np.asarray(qa)
    dqa = np.asarray(dqa)
    # Calcule de l'erreur de position de joints:
    delta_q = qa - qd
    print("delta_q", delta_q)
    # calcule de la jacobienne dans la config actuel:
    J = robot.getJacobian(qa)
    J_inv = np.linalg.inv(J)
    # Supression de l'asservisement sur l'axe z et teta_Z
    qc = np.dot(J_inv, (np.dot(C, np.dot(J, delta_q))))
    # print ("qc", qc)
    qcx = np.dot(qc, J)
    print("qcX", qcx)
    qr = qc + qd
    q_prePI = qr - qa

    # Regulateur position 1

    q_prePI_somme += q_prePI
    CMD_pos1 = q_prePI * kpp + q_prePI_somme * kpi

    # regulateur position 2

    qr_diff = qr - qr_old
    q_preD = qr_diff - dqa

    CMD_pos2 = q_preD * kpd

    # Force

    CMD_force = np.dot(desired_force, J_inv)

    # Somme CMD

    CMD_final = CMD_pos1 + CMD_pos2 + CMD_force

    qr_old = qr
    print("cmd: ", CMD_final)
    # Activate torque control
    p.setJointMotorControlArray(robot.robot_id, robot.controllable_joints,
                                controlMode=p.TORQUE_CONTROL,
                                forces=CMD_final)

    p.stepSimulation()
    time.sleep(robot.time_step)
    # if (dq <= 0.001):
    #    return True





