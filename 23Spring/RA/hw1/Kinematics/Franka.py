import numpy as np
import RobotUtil as rt
import math
# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D


class FrankArm:

	def __init__(self):
		# Robot descriptor taken from URDF file (rpy xyz for each rigid link transform) - NOTE: don't change
		self.Rdesc=[
			[0, 0, 0, 0., 0, 0.333], # From robot base to joint1
			[-np.pi/2, 0, 0, 0, 0, 0],
			[np.pi/2, 0, 0, 0, -0.316, 0],
			[np.pi/2, 0, 0, 0.0825, 0, 0],
			[-np.pi/2, 0, 0, -0.0825, 0.384, 0],
			[np.pi/2, 0, 0, 0, 0, 0],
			[np.pi/2, 0, 0, 0.088, 0, 0],
			[0, 0, 0, 0, 0, 0.107] # From joint5 to end-effector center
			]
		
		#Define the axis of rotation for each joint 
		self.axis=[
				[0, 0, 1],
				[0, 0, 1],
				[0, 0, 1],
				[0, 0, 1],
				[0, 0, 1],
				[0, 0, 1],
				[0, 0, 1],
				[0, 0, 1]
				]

		#Set base coordinate frame as identity - NOTE: don't change
		self.Tbase= [[1,0,0,0],
			[0,1,0,0],
			[0,0,1,0],
			[0,0,0,1]]
		
		#Initialize matrices - NOTE: don't change this part
		self.Tlink=[] #Transforms for each link (const)
		self.Tjoint=[] #Transforms for each joint (init eye)
		self.Tcurr=[] #Coordinate frame of current (init eye)
		for i in range(len(self.Rdesc)):
			self.Tlink.append(rt.rpyxyz2H(self.Rdesc[i][0:3],self.Rdesc[i][3:6]))
			self.Tcurr.append([[1,0,0,0],[0,1,0,0],[0,0,1,0.],[0,0,0,1]])
			self.Tjoint.append([[1,0,0,0],[0,1,0,0],[0,0,1,0.],[0,0,0,1]])

		self.Tlinkzero=rt.rpyxyz2H(self.Rdesc[0][0:3],self.Rdesc[0][3:6])  

		self.Tlink[0]=np.matmul(self.Tbase,self.Tlink[0])					

		# initialize Jacobian matrix
		self.J=np.zeros((6,7))
		
		self.q=[0.,0.,0.,0.,0.,0.,0.]
		self.ForwardKin([0.,0.,0.,0.,0.,0.,0.])
		
	def ForwardKin(self,ang):
		'''
		inputs: joint angles
		outputs: joint transforms for each joint, Jacobian matrix
		'''
		
		self.q[0:-1]=ang
		
		#Compute current joint and end effector coordinate frames (self.Tjoint). Remember that not all joints rotate about the z axis!
		
		for i in range(len(self.q)):
			self.Tjoint[i] = [[math.cos(self.q[i]), -math.sin(self.q[i]), 0, 0],
								[math.sin(self.q[i]), math.cos(self.q[i]), 0, 0], 
								[0, 0, 1, 0],
								[0, 0, 0, 1]]
			if i == 0:
				self.Tcurr[i] = self.Tlink[i]@self.Tjoint[i]
			else:
				self.Tcurr[i] = self.Tcurr[i-1]@self.Tlink[i]@self.Tjoint[i]
		
		# Compute Jacobian
		for i in range(len(self.Tcurr)-1):
			p = self.Tcurr[-1][0:3, 3] - self.Tcurr[i][0:3, 3]
			a = self.Tcurr[i][0:3, 2]
			self.J[0:3, i] = np.cross(a, p)
			self.J[3:7, i] = a

		return self.Tcurr, self.J

		

	def IterInvKin(self,ang,TGoal,x_eps=1e-3, r_eps=1e-3):
		'''
		inputs: starting joint angles (ang), target end effector pose (TGoal)

		outputs: computed joint angles to achieve desired end effector pose, 
		Error in your IK solution compared to the desired target
		'''	

		W = np.eye(7)
		C = np.eye(6)

		W = np.array(
			[[1, 0, 0, 0, 0, 0, 0],
			[0, 1, 0, 0, 0, 0, 0],
			[0, 0, 100, 0, 0, 0, 0],
			[0, 0, 0, 100, 0, 0, 0],
			[0, 0, 0, 0, 1, 0, 0],
			[0, 0, 0, 0, 0, 1, 0],
			[0, 0, 0, 0, 0, 0, 100]]
		)
		C = np.array(
			[[100000, 0, 0, 0, 0, 0],
			[0, 100000, 0, 0, 0, 0],
			[0, 0, 100000, 0, 0, 0],
			[0, 0, 0, 1000, 0, 0],
			[0, 0, 0, 0, 1000, 0],
			[0, 0, 0, 0, 0, 1000]]
		)

		Err = np.zeros((6,))

		self.q[0:7] = ang
		self.ForwardKin(self.q[0:7])

		# Compute rotational error
		rErrR = TGoal[0:3, 0:3] @ np.transpose(self.Tcurr[-1][0:3, 0:3])
		rErrAxis, rErrAng = rt.R2axisang(rErrR)
		rErr = [rErrAxis[0]*rErrAng, rErrAxis[1]*rErrAng, rErrAxis[2]*rErrAng]

		# Compute Error Position
		xErr = TGoal[0:3, 3] - self.Tcurr[-1][0:3, 3]
		xErr = np.linalg.norm(xErr)
		# print(f"1. {rErrR}, \n2. {rErrAxis}, \n 3.{x_eps}, \n 4. {rErrAng}, \n 5. {r_eps}")
		Err[0:3] = xErr
		Err[3:6] = rErr
		
		while np.linalg.norm(Err[0:3]) >x_eps or np.linalg.norm(Err[3:6]) > r_eps:
			# print(f"==={np.linalg.norm(xErr)}, {np.linalg.norm(rErr)}")
			# print(f"+++{np.linalg.norm(Err[0:3]) }, {np.linalg.norm(Err[3:]) }")
			# Compute rotational error
			rErrR = TGoal[0:3, 0:3] @ np.transpose(self.Tcurr[-1][0:3, 0:3])
			rErrAxis, rErrAng = rt.R2axisang(rErrR)
			
			if rErrAng > 0.1:
				rErrAng = 0.1
			if rErrAng < -0.1:
				rErrAng = -0.1
			rErr = [rErrAxis[0]*rErrAng, rErrAxis[1]*rErrAng, rErrAxis[2]*rErrAng]

			# Compute Error Position
			xErr = TGoal[0:3, 3] - self.Tcurr[-1][0:3, 3]
			
			if np.linalg.norm(xErr)>0.01:
				xErr = xErr*0.01/np.linalg.norm(xErr)

			# Update angles with 
			Err[0:3] = xErr
			Err[3:6] = rErr

			# psuedo inverse
			# self.q[0:7] = self.q[0:7] + self.J.T @ np.linalg.pinv(self.J @ self.J.T) @ Err
			# print(self.q)

			dq = np.linalg.inv(W) @ self.J.T @ np.linalg.inv(self.J@ np.linalg.inv(W)@self.J.T+np.linalg.inv(C))@ Err
			self.q[0:7] += dq
			
			# Recompute forward kinematics for new angles
			self.ForwardKin(self.q[0:7])

		return self.q[0:-1], Err



'''
# damped least squares
W = np.array(
	[[1, 0, 0, 0, 0, 0, 0],
	[0, 1, 0, 0, 0, 0, 0],
	[0, 0, 100, 0, 0, 0, 0],
	[0, 0, 0, 100, 0, 0, 0],
	[0, 0, 0, 0, 1, 0, 0],
	[0, 0, 0, 0, 0, 1, 0],
	[0, 0, 0, 0, 0, 0, 100]]
)
C = np.array(
	[[100000, 0, 0, 0, 0, 0],
	[0, 100000, 0, 0, 0, 0],
	[0, 0, 100000, 0, 0, 0],
	[0, 0, 0, 1000, 0, 0],
	[0, 0, 0, 0, 1000, 0],
	[0, 0, 0, 0, 0, 1000]]
)
'''