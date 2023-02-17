import mujoco as mj
from mujoco import viewer
import numpy as np
import math
import quaternion
from scipy.spatial.transform import Rotation as R

right_now = 0
# Set the XML filepath
xml_filepath = "../franka_emika_panda/panda_nohand_torque_fixed_board.xml"
# xml_filepath = "../franka_emika_panda/panda_nohand_torque.xml"

################################# Control Callback Definitions #############################

# Control callback for gravity compensation
def gravity_comp(model, data):

    # data.ctrl exposes the member that sets the actuator control inputs that participate in the
    # physics, data.qfrc_bias exposes the gravity forces expressed in generalized coordinates, i.e.
    # as torques about the joints

    data.ctrl[:7] = data.qfrc_bias[:7]

# Force control callback
def force_control(model, data): #TODO:

    # Implement a force control callback here that generates a force of 15 N along the global x-axis,
    # i.e. the x-axis of the robot arm base. You can use the comments as prompts or use your own flow
    # of code. The comments are simply meant to be a reference.

    try:
        global right_now

        # Instantite a handle to the desired body on the robot
        body = data.body("hand")

        # Get the Jacobian for the desired location on the robot (The end-effector)
        jacp = np.zeros((3, model.nv))
        jacr = np.zeros((3, model.nv))
        # print(body.id, model.nv)

        # This function works by taking in return parameters!!! Make sure you supply it with placeholder
        # variables
        mj.mj_jacBody(model, data, jacp, jacr, body.id)

        J = np.vstack((jacp, jacr))

        # Specify the desired force in global coordinates
        F_des = np.array([15, 0, 0, 0, 0 ,0])

        # Compute the required control input using desied force values
        # print(J.shape, F_des.shape)

        torque = np.transpose(J)@F_des
        # print("torque", torque)

        # Set the control inputs
        data.ctrl[:7] = torque[:7] + data.qfrc_bias[:7]

        
        # DO NOT CHANGE ANY THING BELOW THIS IN THIS FUNCTION
        if right_now <=7000:
            # Force readings updated here
            force[:] = np.roll(force, -1)[:]
            force[-1] = data.sensordata[2]


        current_orientation = body.xquat                                                                    # Cartesian 
        r = R.from_quat(current_orientation)
        current_orientation_rot = r.as_rotvec()


        print("pose: ", body.xpos)
        print("orientation: ", current_orientation_rot)

    except Exception as e: print(e)

# Control callback for an impedance controller
def impedance_control(model, data): #TODO:
    try:
        global right_now
        # Implement an impedance control callback here that generates a force of 15 N along the global x-axis,
        # i.e. the x-axis of the robot arm base. You can use the comments as prompts or use your own flow
        # of code. The comments are simply meant to be a reference.

        # Instantite a handle to the desired body on the robot
        body = data.body("hand")

        # Set the desired position
        offset = 0.5
        desired_positions = np.array([0.59526383 + offset, 0.00142364, 0.59517832, -1.21350223, 1.2065749, -1.21243029]).reshape((6, 1))
        # desired_joint_positions = np.array([3.30755052e-03,1.13437630e-01,8.52247579e-04,-2.00667732e+00,7.33036917e-03 ,3.69150669e+00,-7.86032053e-01])
        
        # Set the desired velocities
        desired_velocities = np.zeros((6,1))

        # Set the desired orientation (Use numpy quaternion manipulation functions)
        desired_orientation = np.zeros((3,1))

        # Get the current orientation
        current_orientation = body.xquat                                                                    # Cartesian 
        r = R.from_quat(current_orientation)
        current_orientation_rot = r.as_rotvec()

        # Get orientation error
        # orientation_err = desired_orientation - current_orientation_rot

        # Get the position error
        current_position = np.hstack((body.xpos.T, current_orientation_rot)).reshape((6,1))#np.zeros((3,))) # Global Frame

        err_pose = desired_positions - current_position
        err_pose[3:] = np.zeros((3,1))
        # err_pose = desired_joint_positions - data.qpos

        current_velocity = np.zeros((6, 1)) # data.qvel
        mj.mj_objectVelocity(model, data, mj.mjtObj.mjOBJ_BODY, body.id, current_velocity, True) 
        err_velocity = desired_velocities - current_velocity

        # Get the Jacobian at the desired location on the robot
        jacp = np.zeros((3, model.nv))
        jacr = np.zeros((3, model.nv))
        
        # This function works by taking in return parameters!!! Make sure you supply it with placeholder
        # variables
        mj.mj_jacBody(model, data, jacp, jacr, body.id)
        J = np.vstack((jacp, jacr))

        # Compute the impedance control input torques
        Kp = 30
        Kd = 10
        F_give = Kd*err_velocity + Kp*err_pose
        T_give  = J.T @ F_give
        
        # Set the control inputs
        data.ctrl[:7] = T_give[:7].T + data.qfrc_bias[:7]

        # DO NOT CHANGE ANY THING BELOW THIS IN THIS FUNCTION
        if right_now <=7000:
            # Update force sensor readings
            force[:] = np.roll(force, -1)[:]
            force[-1] = data.sensordata[2]

    except Exception as e: print(e)

def position_control(model, data):

    # Instantite a handle to the desired body on the robot
    body = data.body("hand")

    # Set the desired joint angle positions
    # desired_joint_positions = np.array([0,0,0,-1.57079,0,1.57079,-0.7853])

    # desired_joint_positions = np.array([0.5767914507000499, 0.3122892679211738, -0.6830103902049547, -2.078731205593762, -0.11554827174364649, 3.914213627217315, -2.8628557307097973])
    desired_joint_positions = np.array([0.577,0.312, -0.683, -2.079, -0.116,  3.914 ,-2.863])
    # # Set the desired joint velocities
    desired_joint_velocities = np.array([0,0,0,0,0,0,0])

    # Desired gain on position error (K_p)
    Kp = 1000

    # Desired gain on velocity error (K_d)
    Kd = 1000

    # Set the actuator control torques
    data.ctrl[:7] = data.qfrc_bias[:7] + Kp*(desired_joint_positions-data.qpos[:7]) + Kd*(np.array([0,0,0,0,0,0,0])-data.qvel[:7])


####################################### MAIN #####################################

if __name__ == "__main__":
    
    # Load the xml file here
    model = mj.MjModel.from_xml_path(xml_filepath)
    data = mj.MjData(model)

    # Set the simulation scene to the home configuration
    mj.mj_resetDataKeyframe(model, data, 0)

    ################################# Swap Callback Below This Line #################################
    # This is where you can set the control callback. Take a look at the Mujoco documentation for more
    # details. Very briefly, at every timestep, a user-defined callback function can be provided to
    # mujoco that sets the control inputs to the actuator elements in the model. The gravity
    # compensation callback has been implemented for you. Run the file and play with the model as
    # explained in the PDF

    # mj.set_mjcb_control(gravity_comp) #TODO:
    # mj.set_mjcb_control(force_control) #TODO:
    mj.set_mjcb_control(impedance_control)

    ################################# Swap Callback Above This Line #################################

    # Initialize variables to store force and time data points
    force_sensor_max_time = 10
    force = np.zeros(int(force_sensor_max_time/model.opt.timestep))
    time = np.linspace(0, force_sensor_max_time, int(force_sensor_max_time/model.opt.timestep))

    # Launch the simulate viewer
    viewer.launch(model, data)   

    # Save recorded force and time points as a csv file
    force = np.reshape(force, (5000, 1))
    time = np.reshape(time, (5000, 1))
    plot = np.concatenate((time, force), axis=1)
    np.savetxt('force_vs_time_2.csv', plot, delimiter=',')
