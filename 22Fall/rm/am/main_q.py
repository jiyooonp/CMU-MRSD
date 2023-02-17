#!/usr/bin/env python3
"""
Air Mobility Project- 16665
Author: Shruti Gangopadhyay (sgangopa), Rathn Shah(rsshah)
"""

from distutils.log import error
from locale import currency
import numpy as np
import sys
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import math

state = 0


def lookup_waypoints(question):
    '''
    Input parameters:

    question: which question of the project we are on 
    (Possible arguments for question: 2, 3, 5, 6.2, 6.3, 6.5, 7, 9, 10)

    Output parameters:

    waypoints: of the form [x, y, z, yaw]
 
    waypoint_times: vector of times where n is the number of waypoints, 
    represents the seconds you should be at each respective waypoint
    '''

    # TO DO:

    # sample waypoints for hover trajectory 
    if question == 2:
        waypoints = np.array([[0, 0.1, 0.2, 0.3], [0, 0, 0, 0], [0.5, 0.5, 0.5, 0.5], [0, 0, 0, 0]])
        waypoint_times = np.array([0, 2, 4, 6])
    if question == 3:
        t1, t2, t3 = 5, 3, 5
        n = t1 + t2 + t3

        waypoints = np.array(
            [[0] * n, [0] * n, [0.2 * i for i in range(0, t1)] + [1] * t2 + [1 - 0.2 * i for i in range(0, t3)],
             [0] * n])

        a = 1
        # waypoint_times = np.array([0.2*i for i in range(n)])
        time1 = math.sqrt(2 / a) * waypoints[2, :][:t1] ** 2
        time2 = [time1[-1] + i for i in range(t2)]
        time3 = time2[-1] + math.sqrt(2 / a) * (1 - waypoints[2, :][t1 + t2:]) ** 2

        waypoint_times = np.append(time1, time2, axis=0)
        waypoint_times = np.append(waypoint_times, time3, axis=0)
        # waypoint_times = np.asarray(time1+np.array(time2)+time3)
        # print(waypoint_times)

    if question == 4:
        t1, t2, t3 = 5, 3, 5
        n = t1 + t2 + t3
        a = 1
        idle_n = 2
        waypoints = np.array(
            [[0] * n, [0] * n, [0.2 * i for i in range(0, t1)] + [1] * t2 + [1 - 0.2 * i for i in range(0, t3)],
             [0] * n])
        idle = np.array([[0] * idle_n, [0] * idle_n, [0] * idle_n, [0] * idle_n])
        idle_time = np.array([0, 2])

        # takeoff = np.array([0]*5,[0]*5, [0]*5,[0]*5)

        hover1 = np.array([[0, 0.1, 0.2, 0.3], [0, 0, 0, 0], [0.5, 0.5, 0.5, 0.5], [0, 0, 0, 0]])
        hover1_time = np.array(idle_time[-1] + [0, 2, 4, 6])

        tracking = np.array(
            [[0] * n, [0] * n, [0.5 + 0.1 * i for i in range(0, t1)] + [1] * t2 + [1 - 0.1 * i for i in range(0, t3)],
             [0] * n])

        time1 = math.sqrt(2 / a) * waypoints[2, :][:t1] ** 2
        time2 = [time1[-1] + i for i in range(t2)]
        time3 = time2[-1] + math.sqrt(2 / a) * (1 - waypoints[2, :][t1 + t2:]) ** 2

        waypoint_times = np.append(time1, time2, axis=0)
        waypoint_times = np.append(waypoint_times, time3, axis=0)
        tracking_time = hover1_time[-1] + waypoint_times

        hover2 = np.array([[0, 0.1, 0.2, 0.3], [0, 0, 0, 0], [0.5, 0.5, 0.5, 0.5], [0, 0, 0, 0]])
        hover2_time = tracking_time[-1] + np.array([0, 2, 4, 6])

        # land = np.array([[0.3, 0.4, 0.5, 0.6], [0, 0, 0, 0], [0.5, 0.3, 0.2, 0], [0, 0, 0, 0]])
        land = np.array([[0.3, 0.2, 0.1, 0], [0, 0, 0, 0], [0.5, 0.3, 0.2, 0], [0, 0, 0, 0]])
        land_time = hover2_time[-1] + np.array([0, 0.1, 0.2, 0.3])

        idle_n_2 = 2
        idle2 = np.array([[0] * idle_n_2, [0] * idle_n_2, [0] * idle_n_2, [0] * idle_n_2])
        idle2_time = land_time[-1] + np.array([0, 2])

        # waypoint_times = np.asarray(time1+np.array(time2)+time3)

        waypoints = np.hstack((idle, hover1, tracking, hover2, land, idle2))
        waypoint_times = np.hstack((idle_time, hover1_time, tracking_time, hover2_time, land_time, idle2_time))

        print("final time is ", waypoint_times)
        print("final x is ", waypoints[0])
        print("final y is ", waypoints[1])
        print("final z is ", waypoints[2])

    if question == 5:
        n1 = 5
        n2 = 20

        waypoints1 = np.array([[0] * n1, [0] * n1, [0.3] * n1, [0] * n1])
        waypoint_times1 = np.array([0.2 * i for i in range(n1)])
        print('time you transition', waypoint_times1[-1], 'len', waypoint_times1.shape)

        waypoints2 = np.array([[0] * n2, [0] * n2, [0.1] * n2, [0] * n2])
        waypoint_times2 = waypoint_times1[-1] + np.array([0.2 * i for i in range(n2)])

        waypoints = np.hstack((waypoints1, waypoints2))
        waypoint_times = np.hstack((waypoint_times1, waypoint_times2))
        # print(waypoints)
        # print('time',waypoint_times)

    return ([waypoints, waypoint_times])


# def state_machine(state, time, tracked):
#     global state
#     idle_time_thresh = 3
#     idle_time_thresh = 3
#     time_thresh = 30
#     states = ["idle", "takeoff", "hover", "tracking", "land"]
#
#     # have to give [x, y, z, ori]
#
#     if state == states[0]:
#         if time < idle_time_thresh:
#             x, y, z, ori = [0], [0], [0], [0]
#         elif time >=idle_time_thresh:
#             state = state[1]
#
#     elif state == states[1]:
#         if error<thresh:
#             state = states[2]


def trajectory_planner(question, waypoints, max_iteration, waypoint_times, time_step):
    '''
    Input parameters:

    question: Which question we are on in the assignment

    waypoints: Series of points in [x, y, z, yaw] format
 
    max_iter: Number of time steps
 
    waypoint_times: Time we should be at each waypoint
 
    time_step: Length of one time_step
 
    Output parameters:
 
    trajectory_sate: [15 x max_iter] output trajectory as a matrix of states:
    [x, y, z, xdot, ydot, zdot, phi, theta, psi, phidot, thetadot, psidot, xacc, yacc, zacc]   
    '''

    # TO DO:

    if question == 2:
        # sample code for hover trajectory
        trajectory_state = np.zeros((15, max_iteration))
        # height of 15 for: [x, y, z, xdot, ydot, zdot, phi, theta, psi, phidot, thetadot, psidot, xacc, yacc, zacc]

        current_waypoint_number = 0
        for i in range(0, max_iteration):
            if current_waypoint_number < len(waypoint_times) - 1:
                if (i * time_step) >= waypoint_times[current_waypoint_number + 1]:
                    current_waypoint_number += 1

            trajectory_state[0:3, i] = waypoints[0:3, current_waypoint_number]
            trajectory_state[8, i] = waypoints[3, current_waypoint_number]

    if question == 3:
        # sample code for hover trajectory
        trajectory_state = np.zeros((15, max_iteration))
        # height of 15 for: [x, y, z, xdot, ydot, zdot, phi, theta, psi, phidot, thetadot, psidot, xacc, yacc, zacc]
        a = 1
        current_waypoint_number = 0
        t = (math.sqrt(2 / a) * waypoints[2, :][:5] ** 2)[-1]
        for i in range(0, max_iteration):
            if current_waypoint_number < len(waypoint_times) - 1:
                if (i * time_step) >= waypoint_times[current_waypoint_number + 1]:
                    current_waypoint_number += 1
                    # print(waypoints[2, current_waypoint_number])
            if waypoints[2, current_waypoint_number] < 1 and current_waypoint_number < 10:
                trajectory_state[5, i] = i * a * time_step
                trajectory_state[14, i] = 1

            elif waypoints[2, current_waypoint_number] < 1 and current_waypoint_number > 10:
                trajectory_state[5, i] = a * (i * time_step - t)
                trajectory_state[14, i] = -1

            trajectory_state[0:3, i] = waypoints[0:3, current_waypoint_number]
            trajectory_state[8, i] = waypoints[3, current_waypoint_number]

    if question == 4:
        # sample code for hover trajectory
        trajectory_state = np.zeros((15, max_iteration))
        # height of 15 for: [x, y, z, xdot, ydot, zdot, phi, theta, psi, phidot, thetadot, psidot, xacc, yacc, zacc]
        a = 1
        current_waypoint_number = 0
        t = (math.sqrt(2 / a) * waypoints[2, :][:5] ** 2)[-1]
        for i in range(0, max_iteration):
            if current_waypoint_number < len(waypoint_times) - 1:
                if (i * time_step) >= waypoint_times[current_waypoint_number + 1]:
                    current_waypoint_number += 1
                    # print(waypoints[2, current_waypoint_number])
            # if waypoints[2, current_waypoint_number]<1 and current_waypoint_number<10:
            #     trajectory_state[5, i] = i*a*time_step
            #     trajectory_state[14, i] = 1
            #
            # elif waypoints[2, current_waypoint_number]<1 and current_waypoint_number>10:
            #     trajectory_state[5, i] = a * (i*time_step-t)
            #     trajectory_state[14, i] = -1

            trajectory_state[0:3, i] = waypoints[0:3, current_waypoint_number]
            trajectory_state[8, i] = waypoints[3, current_waypoint_number]
    if question == 5:
        # sample code for hover trajectory
        trajectory_state = np.zeros((15, max_iteration))
        # height of 15 for: [x, y, z, xdot, ydot, zdot, phi, theta, psi, phidot, thetadot, psidot, xacc, yacc, zacc]
        current_waypoint_number = 0
        for i in range(0, max_iteration):
            if current_waypoint_number < len(waypoint_times) - 1:
                if (i * time_step) >= waypoint_times[current_waypoint_number + 1]:
                    current_waypoint_number += 1
                    # print(waypoints[2, current_waypoint_number])

            trajectory_state[0:3, i] = waypoints[0:3, current_waypoint_number]
            trajectory_state[8, i] = waypoints[3, current_waypoint_number]

    return (trajectory_state)


def position_controller(current_state, desired_state, params, question):
    '''
    Input parameters:
  
    current_state: The current state of the robot with the following fields:
    current_state["pos"] = [x, y, z],
    current_state["vel"] = [x_dot, y_dot, z_dot]
    current_state["rot"] = [phi, theta, psi]
    current_state["omega"] = [phidot, thetadot, psidot]
    current_state["rpm"] = [w1, w2, w3, w4]
 
    desired_state: The desired states are:
    desired_state["pos"] = [x, y, z] 
    desired_state["vel"] = [x_dot, y_dot, z_dot]
    desired_state["rot"] = [phi, theta, psi]
    desired_state["omega"] = [phidot, thetadot, psidot]
    desired_state["acc"] = [xdotdot, ydotdot, zdotdot]
 
    params: Quadcopter parameters
 
    question: Question number
 
    Output parameters
 
    F: u1 or thrust
 
    acc: will be stored as desired_state["acc"] = [xdotdot, ydotdot, zdotdot]
    '''
    # Example PD gains
    Kp1 = 17
    Kd1 = 6.6  # 6.6

    Kp2 = 17
    Kd2 = 6.6  # 6.6

    Kp3 = 20
    Kd3 = 5

    # TO DO:

    [x, y, z] = current_state["pos"]
    [x_dot, y_dot, z_dot] = current_state["vel"]
    [phi, theta, psi] = current_state["rot"]
    [phidot, thetadot, psidot] = current_state["omega"]
    [w1, w2, w3, w4] = current_state["rpm"]

    [x_d, y_d, z_d] = desired_state["pos"]
    [x_dot_d, y_dot_d, z_dot_d] = desired_state["vel"]
    [phi_d, theta_d, psi_d] = desired_state["rot"]
    [phidot_d, thetadot_d, psidot_d] = desired_state["omega"]
    [xdotdot_d, ydotdot_d, zdotdot_d] = desired_state["acc"]

    m = params["mass"]
    g = params["gravity"]

    Kp = np.array([
        [Kp1, 0, 0],
        [0, Kp2, 0],
        [0, 0, Kp3]
    ])
    Kd = np.array([
        [Kd1, 0, 0],
        [0, Kd2, 0],
        [0, 0, Kd3]
    ])

    pos_e = current_state["pos"] - desired_state["pos"]
    vel_e = current_state["vel"] - desired_state["vel"]

    e_dot_dot = -Kp @ pos_e.T - Kd @ vel_e.T

    acc = e_dot_dot + desired_state["acc"]

    f = m * np.array([0, 0, 1]).T @ (g + acc)

    return [f, acc]


def attitude_by_flatness(desired_state, params):
    '''
    Input parameters:
    
    desired_state: The desired states are:
    desired_state["pos"] = [x, y, z]
    desired_state["vel"] = [x_dot, y_dot, z_dot]
    desired_state["rot"] = [phi, theta, psi]
    desired_state["omega"] = [phidot, thetadot, psidot]
    desired_state["acc"] = [xdotdot, ydotdot, zdotdot]

    params: Quadcopter parameters
    
    Output parameters:
    
    rot: will be stored as desired_state["rot"] = [phi, theta, psi]

    omega: will be stored as desired_state["omega"] = [phidot, thetadot, psidot]
    
    '''

    # TO DO:
    [phi_d, theta_d, psi_d] = desired_state["rot"]
    [phidot_d, thetadot_d, psidot_d] = desired_state["omega"]

    g = params["gravity"]

    A = np.array(
        [[math.sin(psi_d), -math.cos(psi_d)],
         [math.cos(psi_d), math.sin(psi_d)]]
    )
    rot = 1 / g * A @ desired_state["acc"][0:2]
    rot = np.append(rot, psi_d)

    B = np.array(
        [[math.cos(psi_d), math.sin(psi_d)],
         [-math.sin(psi_d), math.cos(psi_d)]]
    )
    C = np.array(
        [[desired_state["acc"][0] * psidot_d],
         [desired_state["acc"][1] * psidot_d]]
    )
    omega = 1 / g * B @ C
    omega = np.append(omega, psidot_d)

    return [rot, omega]


def attitude_controller(params, current_state, desired_state, question):
    '''
    Input parameters
 
    current_state: The current state of the robot with the following fields:
    current_state["pos"] = [x, y, z]
    current_state["vel"] = [x_dot, y_dot, z_dot]
    current_state["rot"] = [phi, theta, psi]
    current_state["omega"] = [phidot, thetadot, psidot]
    current_state["rpm"] = [w1, w2, w3, w4]

    desired_state: The desired states are:
    desired_state["pos"] = [x, y, z] 
    desired_state["vel"] = [x_dot, y_dot, z_dot]
    desired_state["rot"] = [phi, theta, psi]
    desired_state["omega"] = [phidot, thetadot, psidot]
    desired_state["acc"] = [xdotdot, ydotdot, zdotdot]

    params: Quadcopter parameters

    question: Question number

    Output parameters:

    M: u2 or moment [M1, M2, M3]
    '''
    # Example PD gains
    Kpphi = 190
    Kdphi = 30

    Kptheta = 198
    Kdtheta = 30

    Kppsi = 80
    Kdpsi = 17.88

    # TO DO:

    I3 = params["inertia"]

    Kp_ptp = np.diag((Kpphi, Kptheta, Kppsi))
    Kd_ptp = np.diag((Kdphi, Kdtheta, Kdpsi))

    vel_error = np.array(current_state["rot"]) - np.array(desired_state["rot"])
    acc_error = np.array(current_state["omega"]) - np.array(desired_state["omega"])

    M = I3 @ (-Kp_ptp @ vel_error - Kd_ptp @ acc_error)

    return M


def motor_model(F, M, current_state, params):
    '''
    Input parameters"

    F,M: required force and moment

    motor_rpm: current motor RPM

    params: Quadcopter parameters

    Output parameters:

    F_motor: Actual thrust generated by Quadcopter's Motors

    M_motor: Actual Moment generated by the Quadcopter's Motors

    rpm_dot: Derivative of the RPM
    '''

    # TO DO:
    Ct = params["thrust_coefficient"]
    Cq = params["moment_scale"]
    Km = params["motor_constant"]

    d = params["arm_length"]

    A = np.array([
        [Ct, Ct, Ct, Ct],
        [0, d * Ct, 0, -d * Ct],
        [-d * Ct, 0, d * Ct, 0],
        [-Cq, Cq, -Cq, Cq]
    ])

    B = np.append(F, M)

    wi_d_sq = np.linalg.inv(A) @ B
    wi_d = wi_d_sq ** 0.5
    # wi_d = math.sqrt(wi_d_sq)

    rpm_dot = Km * (wi_d - current_state["rpm"])

    T = A @ np.array(current_state["rpm"]) ** 2
    F_motor = T[0]
    M_motor = T[1:]

    return [F_motor, M_motor, rpm_dot]


def dynamics(t, state, params, F_actual, M_actual, rpm_motor_dot):
    '''
    Input parameters:
  
    state: current state, will be using RK45 to update
 
    F, M: actual force and moment from motor model
 
    rpm_motor_dot: actual change in RPM from motor model
  
    params: Quadcopter parameters
 
    question: Question number
 
    Output parameters:
 
    state_dot: change in state
    '''
    # TO DO:
    # print('state', state)
    pos = state[0:3]
    vel = state[3:6]
    rpy = state[6:9]
    ang_vel = state[9:12]
    acc = state[12:15]
    [phi, theta, psi] = rpy

    m = params['mass']
    I = params["inertia"]
    g = params['gravity']

    state_dot = np.zeros(16)
    state_dot[:3] = vel

    acc = 1 / m * np.vstack((
        F_actual * (math.cos(phi) * math.cos(psi) * math.sin(theta) + math.sin(phi) * math.sin(psi)),
        F_actual * (math.cos(phi) * math.sin(theta) * math.sin(psi) - math.cos(psi) * math.sin(phi)),
        F_actual * math.cos(theta) * math.cos(phi) - m * g
    ))

    R = np.vstack((
        [math.cos(theta), 0, -math.cos(phi) * math.sin(theta)],
        [0, 1, math.sin(phi)],
        [math.sin(phi), 0, math.cos(phi) * math.cos(theta)]
    ))
    inertia_dot = np.linalg.inv(R) @ ang_vel

    omega = np.linalg.inv(I) @ (M_actual.reshape(3, 1) - ang_vel @ I @ ang_vel)

    state_dot[3:6] = acc.reshape(3, )
    state_dot[6:9] = inertia_dot
    state_dot[9:12] = omega.reshape(3, )
    state_dot[12:] = rpm_motor_dot

    return state_dot


def plot_state_error(state, state_des, time_vector):
    # actual states
    pos = state[0:3, :]
    vel = state[3:6, :]
    rpy = state[6:9, :]
    ang_vel = state[9:12, :]
    acc = state[12:15, :]

    # desired states
    pos_des = state_des[0:3, :]
    vel_des = state_des[3:6, :]
    rpy_des = state_des[6:9, :]
    ang_vel_des = state_des[9:12, :]
    acc_des = state_des[12:15, :]

    # get error from des and act
    error_pos = pos - pos_des
    error_vel = vel - vel_des
    error_rpy = rpy - rpy_des
    error_ang_vel = ang_vel - ang_vel_des
    error_acc = acc - acc_des

    # plot erros
    fig = plt.figure(1)
    # plot position error
    axs = fig.subplots(5, 3)
    axs[0, 0].plot(time_vector, error_pos[0, :])
    axs[0, 0].set_title("Error in x")
    axs[0, 0].set(xlabel='time(s)', ylabel='x(m)')
    axs[0, 1].plot(time_vector, error_pos[1, :])
    axs[0, 1].set_title("Error in y")
    axs[0, 1].set(xlabel='time(s)', ylabel='y(m)')
    axs[0, 2].plot(time_vector, error_pos[2, :])
    axs[0, 2].set_title("Error in z")
    axs[0, 2].set(xlabel='time(s)', ylabel='z(m)')

    # plot orientation error
    axs[1, 0].plot(time_vector, error_rpy[0, :])
    axs[1, 0].set_title("Error in phi")
    axs[1, 0].set(xlabel='time(s)', ylabel='phi')
    axs[1, 1].plot(time_vector, error_rpy[1, :])
    axs[1, 1].set_title("Error in theta")
    axs[1, 1].set(xlabel='time(s)', ylabel='theta')
    axs[1, 2].plot(time_vector, error_rpy[2, :])
    axs[1, 2].set_title("Error in psi")
    axs[1, 2].set(xlabel='time(s)', ylabel='psi')

    # plot velocity error
    axs[2, 0].plot(time_vector, error_vel[0, :])
    axs[2, 0].set_title("Error in vx")
    axs[2, 0].set(xlabel='time(s)', ylabel='vx (m/s)')
    axs[2, 1].plot(time_vector, error_vel[1, :])
    axs[2, 1].set_title("Error in vy")
    axs[2, 1].set(xlabel='time(s)', ylabel='vy (m/s)')
    axs[2, 2].plot(time_vector, error_vel[2, :])
    axs[2, 2].set_title("Error in vz")
    axs[2, 2].set(xlabel='time(s)', ylabel='vz (m/s)')

    # plot angular velocity error
    axs[3, 0].plot(time_vector, error_ang_vel[0, :])
    axs[3, 0].set_title("Error in omega_x")
    axs[3, 0].set(xlabel='time(s)', ylabel='omega_x (rad/s)')
    axs[3, 1].plot(time_vector, error_ang_vel[1, :])
    axs[3, 1].set_title("Error in omega_y")
    axs[3, 1].set(xlabel='time(s)', ylabel='omega_y (rad/s)')
    axs[3, 2].plot(time_vector, error_ang_vel[2, :])
    axs[3, 2].set_title("Error in omega_z")
    axs[3, 2].set(xlabel='time(s)', ylabel='omega_z (rad/s)')

    # plot acceleration error
    axs[4, 0].plot(time_vector, error_acc[0, :])
    axs[4, 0].set_title("Error in acc_x")
    axs[4, 0].set(xlabel='time(s)', ylabel='acc_x (m/s2)')
    axs[4, 1].plot(time_vector, error_acc[1, :])
    axs[4, 1].set_title("Error in acc_y")
    axs[4, 1].set(xlabel='time(s)', ylabel='acc_y (m/s2)')
    axs[4, 2].plot(time_vector, error_acc[2, :])
    axs[4, 2].set_title("Error in acc_z")
    axs[4, 2].set(xlabel='time(s)', ylabel='acc_z (m/s2)')

    fig.tight_layout(pad=0.005)

    # plot values
    fig1 = plt.figure(2)
    # plot position
    axs1 = fig1.subplots(5, 3)
    axs1[0, 0].plot(time_vector, pos[0, :])
    axs1[0, 0].set_title("x")
    axs1[0, 0].set(xlabel='time(s)', ylabel='x(m)')
    axs1[0, 1].plot(time_vector, pos[1, :])
    axs1[0, 1].set_title("y")
    axs1[0, 1].set(xlabel='time(s)', ylabel='y(m)')
    axs1[0, 2].plot(time_vector, pos[2, :])
    axs1[0, 2].set_title("z")
    axs1[0, 2].set(xlabel='time(s)', ylabel='z(m)')

    # plot orientation 
    axs1[1, 0].plot(time_vector, rpy[0, :])
    axs1[1, 0].set_title("phi")
    axs1[1, 0].set(xlabel='time(s)', ylabel='phi')
    axs1[1, 1].plot(time_vector, rpy[1, :])
    axs1[1, 1].set_title("theta")
    axs1[1, 1].set(xlabel='time(s)', ylabel='theta')
    axs1[1, 2].plot(time_vector, rpy[2, :])
    axs1[1, 2].set_title("psi")
    axs1[1, 2].set(xlabel='time(s)', ylabel='psi')

    # plot velocity 
    axs1[2, 0].plot(time_vector, vel[0, :])
    axs1[2, 0].set_title("vx")
    axs1[2, 0].set(xlabel='time(s)', ylabel='vx (m/s)')
    axs1[2, 1].plot(time_vector, vel[1, :])
    axs1[2, 1].set_title("vy")
    axs1[2, 1].set(xlabel='time(s)', ylabel='vy (m/s)')
    axs1[2, 2].plot(time_vector, vel[2, :])
    axs1[2, 2].set_title("vz")
    axs1[2, 2].set(xlabel='time(s)', ylabel='vz (m/s)')

    # plot angular velocity
    axs1[3, 0].plot(time_vector, ang_vel[0, :])
    axs1[3, 0].set_title("omega_x")
    axs1[3, 0].set(xlabel='time(s)', ylabel='omega_x (rad/s)')
    axs1[3, 1].plot(time_vector, ang_vel[1, :])
    axs1[3, 1].set_title("omega_y")
    axs1[3, 1].set(xlabel='time(s)', ylabel='omega_y (rad/s)')
    axs1[3, 2].plot(time_vector, ang_vel[2, :])
    axs1[3, 2].set_title("omega_z")
    axs1[3, 2].set(xlabel='time(s)', ylabel='omega_z (rad/s)')

    # plot acceleration 
    axs1[4, 0].plot(time_vector, acc[0, :])
    axs1[4, 0].set_title("acc_x")
    axs1[4, 0].set(xlabel='time(s)', ylabel='acc_x (m/s2)')
    axs1[4, 1].plot(time_vector, acc[1, :])
    axs1[4, 1].set_title("acc_y")
    axs1[4, 1].set(xlabel='time(s)', ylabel='acc_y (m/s2)')
    axs1[4, 2].plot(time_vector, acc[2, :])
    axs1[4, 2].set_title("acc_z")
    axs1[4, 2].set(xlabel='time(s)', ylabel='acc_z (m/s2)')

    fig1.tight_layout(pad=0.05)
    plt.show()


def plot_state_error_five(state, state_des, time_vector):
    # actual states
    pos = state[0:3, :]
    vel = state[3:6, :]
    rpy = state[6:9, :]
    ang_vel = state[9:12, :]
    acc = state[12:15, :]

    # desired states
    pos_des = state_des[0:3, :]
    vel_des = state_des[3:6, :]
    rpy_des = state_des[6:9, :]
    ang_vel_des = state_des[9:12, :]
    acc_des = state_des[12:15, :]

    # get error from des and act
    error_pos = pos - pos_des
    error_vel = vel - vel_des
    error_rpy = rpy - rpy_des
    error_ang_vel = ang_vel - ang_vel_des
    error_acc = acc - acc_des
    # plot errors
    fig = plt.figure(1)
    # plot position error
    axs = fig.subplots(5, 1)
    axs[0].plot(time_vector, error_pos[2, :])
    axs[0].set_title("Error in z")
    axs[0].set(xlabel='time(s)', ylabel='z(m)')

    # plot orientation error
    axs[1].plot(time_vector, error_rpy[2, :])
    axs[1].set_title("Error in psi")
    axs[1].set(xlabel='time(s)', ylabel='psi')

    # plot velocity error
    axs[2].plot(time_vector, error_vel[2, :])
    axs[2].set_title("Error in vz")
    axs[2].set(xlabel='time(s)', ylabel='vz (m/s)')

    # plot angular velocity error
    axs[3].plot(time_vector, error_ang_vel[2, :])
    axs[3].set_title("Error in omega_z")
    axs[3].set(xlabel='time(s)', ylabel='omega_z (rad/s)')

    # plot acceleration error
    axs[4].plot(time_vector, error_acc[2, :])
    axs[4].set_title("Error in acc_z")
    axs[4].set(xlabel='time(s)', ylabel='acc_z (m/s2)')

    fig.tight_layout(pad=0.005)

    # plot values
    fig1 = plt.figure(2)
    # plot position
    axs1 = fig1.subplots(5, 1)
    axs1[0].plot(time_vector, pos[2, :])
    axs1[0].set_title("z")
    axs1[0].set(xlabel='time(s)', ylabel='z(m)')

    # plot orientation
    axs1[1].plot(time_vector, rpy[2, :])
    axs1[1].set_title("psi")
    axs1[1].set(xlabel='time(s)', ylabel='psi')

    # plot velocity
    axs1[2].plot(time_vector, vel[2, :])
    axs1[2].set_title("vz")
    axs1[2].set(xlabel='time(s)', ylabel='vz (m/s)')

    # plot angular velocity
    axs1[3].plot(time_vector, ang_vel[2, :])
    axs1[3].set_title("omega_z")
    axs1[3].set(xlabel='time(s)', ylabel='omega_z (rad/s)')

    # plot acceleration
    axs1[4].plot(time_vector, acc[2, :])
    axs1[4].set_title("acc_z")
    axs1[4].set(xlabel='time(s)', ylabel='acc_z (m/s2)')

    fig1.tight_layout(pad=0.05)
    plt.show()


# Helper to visualize positions for the flight of the quadrotor
def plot_position_3d(state, state_des):
    pos = state[0:3, :]
    pos_des = state_des[0:3, :]
    fig = plt.figure(3)
    ax = fig.add_subplot(111, projection='3d')

    ax.plot(pos[0, :], pos[1, :], pos[2, :], color='blue',
            label='Actual position')
    ax.plot(pos_des[0, :], pos_des[1, :], pos_des[2, :], color='red',
            label='Desired position')

    ax.set(xlabel='x (m)')
    ax.set(ylabel='y (m)')
    ax.set(zlabel='z (m)')
    ax.set_title('Position')
    ax.legend()

    # ax.axes.set_xlim3d(left=-.5, right=.5)
    # ax.axes.set_ylim3d(bottom=-.5, top=.5)
    # ax.axes.set_zlim3d(bottom=0, top=.5)

    ax.axes.set_xlim3d(left=-.5, right=.5)
    ax.axes.set_ylim3d(bottom=-.5, top=.5)
    ax.axes.set_zlim3d(bottom=0, top=2)

    plt.show()


def info(state, time_vector):
    # actual states

    pos = state[0:3, :]
    vel = state[3:6, :]

    print("==========position=========")
    z = pos[2, :]
    yp = np.min(z)
    ys = z[-1]
    start = z[4]

    nine = (start - (start - ys) / 10)

    rise_i = np.where((nine - 0.001 <= z) & (z <= nine + 0.001))[-1][-1]
    rise = z[rise_i]
    rise_time = time_vector[rise_i]

    one = (ys + (start - ys) / 10)
    settle_i = np.where((one - 0.001 <= z) & (z <= one + 0.001))[-1][-1]
    settle = z[settle_i]
    settle_time = time_vector[settle_i]
    Mp = 100 * abs(yp - ys) / abs(ys)
    n = 10
    npercent = n * abs(ys) / 100
    # print(f'rise: {rise}, settle: {settle}')
    # print(f'rise time: {rise_time}, settle time: {settle_time}')
    print(f'settle time - rise time: {settle_time - rise_time}')
    print(f'steady state value: {ys}')
    print(f'Mp: {Mp}')

    print("==========linear velocity=========")

    vel = vel[2, :]
    vstart = vel[4]

    # plt.show()
    vyp = np.max(vel)
    vys = vel[-1]

    vnine = (vys - (vys - vstart) / 10)
    # print('vnine', vnine)
    # print('vel', vel)

    vrise_i = np.where((vnine - 0.002 <= vel) & (vel <= vnine + 0.002))[-1][-1]
    vrise = vel[vrise_i]
    vrise_time = time_vector[vrise_i]

    vone = (vstart + (vys - vstart) / 10)
    # print('vone', vone)
    vsettle_i = np.where((vone - 0.002 <= vel) & (vel <= vone + 0.002))[-1][-1]
    # print('vsettle_i', vsettle_i)
    vsettle = vel[vsettle_i]
    vsettle_time = time_vector[vsettle_i]
    vys = vel[-1]
    vMp = 100 * abs(vyp - vys) / abs(vys)
    vn = 10
    npercent = n * abs(vys) / 100
    # print(f'rise: {vrise}, settle: {vsettle}')
    # print(f'rise time: {vrise_time}, settle time: {vsettle_time}')
    print(f'rise time - settle time : {vrise_time - vsettle_time}')
    print(f'steady state value: {vys}')
    print(f"{vyp}, {vys}")
    print(f'Mp: {vMp}')


def main(question):
    # Set up quadrotor physical parameters
    params = {"mass": 0.770, "gravity": 9.80665, "arm_length": 0.1103, "motor_spread_angle": 0.925, \
              "thrust_coefficient": 8.07e-9, "moment_scale": 1.3719e-10, "motor_constant": 36.5, "rpm_min": 3000, \
              "rpm_max": 20000, "inertia": np.diag([0.0033, 0.0033, 0.005]), "COM_vertical_offset": 0.05}

    # Get the waypoints for this specific question
    [waypoints, waypoint_times] = lookup_waypoints(question)
    # waypoints are of the form [x, y, z, yaw]
    # waypoint_times are the seconds you should be at each respective waypoint
    # make sure the simulation parameters below allow you to get to all points

    # Set the simualation parameters
    time_initial = 0
    time_final = (waypoint_times[-1] + 1) // 1
    time_step = 0.005  # in secs
    # 0.005 sec is a reasonable time step for this system

    time_vec = np.arange(time_initial, time_final, time_step).tolist()
    max_iteration = len(time_vec)
    # print(max_iteration)

    # Create the state vector
    state = np.zeros((16, 1))
    # state: [x,y,z,xdot,ydot,zdot,phi,theta,psi,phidot,thetadot,psidot,rpm]

    # Populate the state vector with the first waypoint 
    # (assumes robot is at first waypoint at the initial time)
    state[0] = waypoints[0, 0]
    state[1] = waypoints[1, 0]
    state[2] = waypoints[2, 0]
    state[8] = waypoints[3, 0]

    # Create a trajectory consisting of desired state at each time step
    # Some aspects of this state we can plan in advance, some will be filled during the loop
    trajectory_matrix = trajectory_planner(question, waypoints, max_iteration, waypoint_times, time_step)
    # [x, y, z, xdot, ydot, zdot, phi, theta, psi, phidot, thetadot, psidot, xacc, yacc, zacc]

    # Create a matrix to hold the actual state at each time step
    actual_state_matrix = np.zeros((15, max_iteration))
    # [x, y, z, xdot, ydot, zdot, phi, theta, psi, phidot, thetadot, psidot, xacc, yacc, zacc]
    actual_state_matrix[:, 0] = np.vstack((state[0:12], np.array([[0], [0], [0]])))[:, 0]

    # Create a matrix to hold the actual desired state at each time step

    # Need to store the actual desired state for acc, omega dot, omega as it will be updated by the controller
    actual_desired_state_matrix = np.zeros((15, max_iteration))

    # state list created for the RK45 solver
    state_list = state.T.tolist()
    state_list = state_list[0]

    # Loop through the timesteps and update quadrotor
    for i in range(max_iteration - 1):
        # print(i)

        # convert current state to stuct for control functions
        current_state = {"pos": state_list[0:3], "vel": state_list[3:6], "rot": state_list[6:9], \
                         "omega": state_list[9:12], "rpm": state_list[12:16]}

        # Get desired state from matrix, put into struct for control functions
        desired_state = {"pos": trajectory_matrix[0:3, i], "vel": trajectory_matrix[3:6, i], \
                         "rot": trajectory_matrix[6:9, i], "omega": trajectory_matrix[9:12, i],
                         "acc": trajectory_matrix[12:15, i]}

        # Get desired acceleration from position controller
        [F, desired_state["acc"]] = position_controller(current_state, desired_state, params, question)
        # print(desired_state["acc"])

        # Computes desired pitch and roll angles
        [desired_state["rot"], desired_state["omega"]] = attitude_by_flatness(desired_state, params)

        # Get torques from attitude controller
        M = attitude_controller(params, current_state, desired_state, question)

        # Motor model
        [F_actual, M_actual, rpm_motor_dot] = motor_model(F, M, current_state, params)

        # Get the change in state from the quadrotor dynamics
        time_int = tuple((time_vec[i], time_vec[i + 1]))

        sol = solve_ivp(dynamics, time_int, state_list, args=(params, F_actual, M_actual, rpm_motor_dot),
                        t_eval=np.linspace(time_vec[i], time_vec[i + 1], (int(time_step / 0.00005))))

        state_list = sol.y[:, -1]
        acc = (sol.y[3:6, -1] - sol.y[3:6, -2]) / (sol.t[-1] - sol.t[-2])

        # Update desired state matrix
        actual_desired_state_matrix[0:3, i + 1] = desired_state["pos"]
        actual_desired_state_matrix[3:6, i + 1] = desired_state["vel"]
        actual_desired_state_matrix[6:9, i + 1] = desired_state["rot"]
        actual_desired_state_matrix[9:12, i + 1] = desired_state["omega"]
        actual_desired_state_matrix[12:15, i + 1] = desired_state["acc"]

        # Update actual state matrix
        actual_state_matrix[0:12, i + 1] = state_list[0:12]
        actual_state_matrix[12:15, i + 1] = acc

    # plot for values and errors
    if question == 5:
        plot_state_error_five(actual_state_matrix, actual_desired_state_matrix, time_vec)
        info(actual_state_matrix, time_vec)
    else:
        plot_state_error(actual_state_matrix, actual_desired_state_matrix, time_vec)

    # plot for 3d visualization
    plot_position_3d(actual_state_matrix, actual_desired_state_matrix)


if __name__ == '__main__':
    '''
    Usage: main takes in a question number and executes all necessary code to
    construct a trajectory, plan a path, simulate a quadrotor, and control
    the model. Possible arguments: 2, 3, 5, 6.2, 6.3, 6.5, 7, 9. THE
    TAS WILL BE RUNNING YOUR CODE SO PLEASE KEEP THIS MAIN FUNCTION CALL 
    STRUCTURE AT A MINIMUM.
    '''
    # run the file with command "python3 main.py question_number" in the terminal
    # question = sys.argv[1]
    # main(question)
    main(5)
