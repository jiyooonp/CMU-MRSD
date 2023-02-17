import Franka
import numpy as np

#Initialize robot object
mybot=Franka.FrankArm()

# Compute forward kinematics
deg_to_rad = np.pi/180.

joint_targets = [[0., 0., 0., 0., 0., 0., 0.], \
                 [0, 0, -45.*deg_to_rad,-15.*deg_to_rad,20.*deg_to_rad,15.*deg_to_rad,-75.*deg_to_rad], \
                 [0, 0, 30.*deg_to_rad, -60.*deg_to_rad,-65.*deg_to_rad,45.*deg_to_rad,0.*deg_to_rad],\
		            ]

for joint_target in joint_targets:
  print ('\nJoints:')
  print (joint_target)
  Hcurr,J =mybot.ForwardKin(joint_target)
  ee_pose = Hcurr[-1]
  rot_ee = ee_pose[:3,:3]
  pos_ee = ee_pose[:3,3]
  print('computed FK ee position')
  print(np.round(pos_ee, 3))
  print('computed FK ee rotation')
  print(np.round(rot_ee, 3))


# Compute inverse kinematics
qInit = [0,0,0,-2.11,0,3.65,-0.785]
HGoal= np.array([[0.,0.,1.,0.6], # target EE pose
		 [0.,1.,0.,0.0],
		 [-1.,0,0.,0.5],
		 [0.,0.,0.,1]])

q,Err=mybot.IterInvKin(qInit, HGoal)
print('Error', np.linalg.norm(Err[0:3]),np.linalg.norm(Err[3:6]))
print('Computed IK angles', np.round(q, 3))


'''
Joints:
[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
computed FK ee position
[ 8.80000000e-02 -8.93992163e-18  9.26000000e-01]
computed FK ee rotation
[[ 1.0000000e+00  0.0000000e+00  0.0000000e+00]
 [ 0.0000000e+00 -1.0000000e+00 -1.2246468e-16]
 [ 0.0000000e+00  1.2246468e-16 -1.0000000e+00]]

Joints:
[0, 0, -0.7853981633974483, -0.2617993877991494, 0.3490658503988659, 0.2617993877991494, -1.3089969389957472]
computed FK ee position
[ 0.15710277 -0.10259332  0.93602711]
computed FK ee rotation
[[ 0.64935398  0.75871099  0.05193309]
 [ 0.7552124  -0.65137389  0.07325497]
 [ 0.08940721 -0.00834789 -0.99596017]]

Joints:
[0, 0, 0.5235987755982988, -1.0471975511965976, -1.1344640137963142, 0.7853981633974483, 0.0]
computed FK ee position
[0.40136375 0.08742801 0.85526363]
computed FK ee rotation
[[ 0.98015816 -0.18113365 -0.08050201]
 [-0.17410263 -0.5925751  -0.78647507]
 [ 0.09475362  0.78488557 -0.61235316]]
Error 5.386357844421866e-06 0.0002414221596293924
Computed IK angles [0.5767914507000499, 0.3122892679211738, -0.6830103902049547, -2.078731205593762, -0.11554827174364649, 3.914213627217315, -2.8628557307097973]
'''


'''
Joints:
[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
computed FK ee position
[ 0.088 -0.     0.926]
computed FK ee rotation
[[ 1.  0.  0.]
 [ 0. -1. -0.]
 [ 0.  0. -1.]]

Joints:
[0, 0, -0.7853981633974483, -0.2617993877991494, 0.3490658503988659, 0.2617993877991494, -1.3089969389957472]
computed FK ee position
[ 0.157 -0.103  0.936]
computed FK ee rotation
[[ 0.649  0.759  0.052]
 [ 0.755 -0.651  0.073]
 [ 0.089 -0.008 -0.996]]

Joints:
[0, 0, 0.5235987755982988, -1.0471975511965976, -1.1344640137963142, 0.7853981633974483, 0.0]
computed FK ee position
[0.401 0.087 0.855]
computed FK ee rotation
[[ 0.98  -0.181 -0.081]
 [-0.174 -0.593 -0.786]
 [ 0.095  0.785 -0.612]]
Error 5.386357844421866e-06 0.0002414221596293924
Computed IK angles [ 0.577  0.312 -0.683 -2.079 -0.116  3.914 -2.863]
'''