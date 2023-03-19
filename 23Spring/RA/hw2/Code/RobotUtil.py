import numpy as np 
import math

def rpyxyz2H(rpy,xyz):
	Ht=[[1,0,0,xyz[0]],
	    [0,1,0,xyz[1]],
            [0,0,1,xyz[2]],
            [0,0,0,1]]

	Hx=[[1,0,0,0],
	    [0,math.cos(rpy[0]),-math.sin(rpy[0]),0],
            [0,math.sin(rpy[0]),math.cos(rpy[0]),0],
            [0,0,0,1]]

	Hy=[[math.cos(rpy[1]),0,math.sin(rpy[1]),0],
            [0,1,0,0],
            [-math.sin(rpy[1]),0,math.cos(rpy[1]),0],
            [0,0,0,1]]

	Hz=[[math.cos(rpy[2]),-math.sin(rpy[2]),0,0],
            [math.sin(rpy[2]),math.cos(rpy[2]),0,0],
            [0,0,1,0],
            [0,0,0,1]]

	H=np.matmul(np.matmul(np.matmul(Ht,Hz),Hy),Hx)

	return H

def R2axisang(R):
	ang = math.acos(( R[0,0] + R[1,1] + R[2,2] - 1)/2)
	Z = np.linalg.norm([R[2,1] - R[1,2], R[0,2] - R[2,0], R[1,0] - R[0,1]])
	if Z==0:
		return[1,0,0], 0.
	x = (R[2,1] - R[1,2])/Z
	y = (R[0,2] - R[2,0])/Z
	z = (R[1,0] - R[0,1])/Z 	

	return[x, y, z], ang


def BlockDesc2Points(H, Dim):
	center = H[0:3,3]
	axes=[ H[0:3,0],H[0:3,1],H[0:3,2]]	

	corners=[
		center,
		center+(axes[0]*Dim[0]/2.)+(axes[1]*Dim[1]/2.)+(axes[2]*Dim[2]/2.),
		center+(axes[0]*Dim[0]/2.)+(axes[1]*Dim[1]/2.)-(axes[2]*Dim[2]/2.),
		center+(axes[0]*Dim[0]/2.)-(axes[1]*Dim[1]/2.)+(axes[2]*Dim[2]/2.),
		center+(axes[0]*Dim[0]/2.)-(axes[1]*Dim[1]/2.)-(axes[2]*Dim[2]/2.),
		center-(axes[0]*Dim[0]/2.)+(axes[1]*Dim[1]/2.)+(axes[2]*Dim[2]/2.),
		center-(axes[0]*Dim[0]/2.)+(axes[1]*Dim[1]/2.)-(axes[2]*Dim[2]/2.),
		center-(axes[0]*Dim[0]/2.)-(axes[1]*Dim[1]/2.)+(axes[2]*Dim[2]/2.),
		center-(axes[0]*Dim[0]/2.)-(axes[1]*Dim[1]/2.)-(axes[2]*Dim[2]/2.)
		]	
	# returns corners of BB and axes
	return corners, axes



def CheckPointOverlap(pointsA,pointsB,axis):	
	# TODO: check if sets of points projected on axis are overlapping

	# print("Shape of A:", pointsA.shape)
	# print("Shape of B:", pointsB.shape)
	# print("Shape of axis:", axis.shape)

	# print()

	status = False

	pointsA = np.asarray(pointsA)
	pointsB = np.asarray(pointsB)

	# print("Shape of A:", pointsA.shape)
	# print("Shape of B:", pointsB.shape)

	# print("Points A:", pointsA)
	# print("Points B:", pointsB)
	# print("Axis:", axis)

	# print(np.linalg.norm(axis))

	norm_axis = axis/np.linalg.norm(axis)
	projectionA_scals = np.dot(axis, pointsA.T)
	project_axis = np.column_stack((norm_axis,norm_axis,norm_axis,norm_axis,norm_axis,norm_axis,norm_axis,norm_axis,norm_axis))
	
	projectionA = project_axis*projectionA_scals

	projectionB_scals = np.dot(axis, pointsB.T)

	projectionB = project_axis*projectionB_scals

	# check overlap
	minA = np.min(projectionA[0,:])
	maxA = np.max(projectionA[0,:])
	minB = np.min(projectionB[0,:])
	maxB = np.max(projectionB[0,:])


	# print("Projected Shape of A:", projectionA.shape)

	if (np.min(projectionA[0,:])>np.max(projectionB[0,:]) or np.max(projectionA[0,:])<np.min(projectionB[0,:])) or (np.min(projectionA[1,:])>np.max(projectionB[1,:]) or np.max(projectionA[1,:])<np.min(projectionB[1,:])) or (np.min(projectionA[2,:])>np.max(projectionB[2,:]) or np.max(projectionA[2,:])<np.min(projectionB[2,:])):
		# print("There is a separation")
		# print("Points A:", pointsA)
		# print("Points B:", pointsB)

		# print("Points projected on axis", axis)
		# print("Projected Points A:", projectionA)
		# print("Projected Points B:", projectionB)

		return False


	return True

def CheckPointOverlap(pointsA,pointsB,axis):
	# TODO: check if sets of points projected on axis are overlapping
# convert the points to np arrays
	pointsA = np.asarray(pointsA)
	pointsB = np.asarray(pointsB)

	# proj of u onto v = dot(u, v) * (v / mag(v))
	# the second term int the unit vector
	unit_vec_axis = axis/np.linalg.norm(axis)

	# find the proj of A/B onto the axis
	proj_A2axis = np.multiply(np.dot(pointsA, axis), np.expand_dims(unit_vec_axis, axis=1))
	proj_B2axis = np.multiply(np.dot(pointsB, axis), np.expand_dims(unit_vec_axis, axis=1))

	# Conditions for failing
	# 1. smallest A is greater than greated B (x_axis)
	smallA_greatestB_x = np.min(proj_A2axis[0, :]) > np.max(proj_B2axis[0, :])
	# 2. largest A is smaller than smallest B (x_axis)
	largestA_smallestB_x = np.max(proj_A2axis[0, :]) < np.min(proj_B2axis[0, :])
	# 3. smallest A is greater than greated B (y_axis)
	smallA_greatestB_y = np.min(proj_A2axis[1, :]) > np.max(proj_B2axis[1, :])
	# 4. largest A is smaller than smallest B (y_axis)
	largestA_smallestB_y = np.max(proj_A2axis[1, :]) < np.min(proj_B2axis[1, :])
	# 5. smallest A is greater than greated B (z_axis)
	smallA_greatestB_z = np.min(proj_A2axis[2, :]) > np.max(proj_B2axis[2, :])
	# 6. largest A is smaller than smallest B (z_axis)
	largestA_smallestB_z = np.max(proj_A2axis[2, :]) < np.min(proj_B2axis[2, :])

	# x-axis overlap
	if smallA_greatestB_x or largestA_smallestB_x:
		return False
	# y-axis overlap
	elif smallA_greatestB_y or largestA_smallestB_y:
		return False
	# z-axis overlap
	elif smallA_greatestB_z or largestA_smallestB_z:
		return False
	else:
		return True
def CheckPointOverlap1(pointsA, pointsB, axis):
	# TODO: check if sets of points projected on axis are overlapping
	pointsA = np.asarray(pointsA)
	pointsB = np.asarray(pointsB)
	norm_axis = axis / np.linalg.norm(axis)
	projectionA_scals = np.dot(axis, pointsA.T)
	project_axis = np.column_stack(
		(norm_axis, norm_axis, norm_axis, norm_axis, norm_axis, norm_axis, norm_axis, norm_axis, norm_axis))
	projectionA = project_axis * projectionA_scals
	projectionB_scals = np.dot(axis, pointsB.T)
	projectionB = project_axis * projectionB_scals

	# check overlap
	# minA_0, minA_1, minA_2 = np.min(projectionA[0, :]), np.min(projectionA[1, :]), np.min(projectionA[2, :])
	# maxA_0, maxA_1, maxA_2 = np.max(projectionA[0, :]), np.max(projectionA[1, :]), np.max(projectionA[2, :])
	# maxB_0, maxB_1, maxB_2 = np.max(projectionB[0, :]), np.max(projectionB[1, :]), np.max(projectionB[2, :])
	# minB_0, minB_1, minB_2 = np.min(projectionB[0, :]), np.min(projectionB[1, :]), np.min(projectionB[2, :])
	#
	#
	if (np.min(projectionA[0, :]) > np.max(projectionB[0, :]) or np.max(projectionA[0, :]) < np.min(
			projectionB[0, :])) or (
			np.min(projectionA[1, :]) > np.max(projectionB[1, :]) or np.max(projectionA[1, :]) < np.min(
			projectionB[1, :])) or (
			np.min(projectionA[2, :]) > np.max(projectionB[2, :]) or np.max(projectionA[2, :]) < np.min(
			projectionB[2, :])):

		return False
	return True
	# if minB_0 <= maxA_0 <= maxB_0:
	# 	return True
	# if minB_0<=minA_0<=maxB_0:
	# 	return True
	# if minA_0<=maxB_0<=maxA_0:
	# 	return True
	# if minA_0<=minB_0<=maxA_0:
	# 	return True
	# for i in range(3):
	# 	minA = np.min(projectionA[i, :])
	# 	maxA = np.max(projectionA[i, :])
	# 	minB = np.min(projectionB[i, :])
	# 	maxB = np.max(projectionB[i, :])
	# 	if minB <= maxA <= maxB:
	# 		return True
	# 	if minB <= minA <= maxB:
	# 		return True
	# 	if minA <= maxB<= maxA:
	# 		return True
	# 	if minA <= minB<= maxA:
	# 		return True
	# return False


def CheckBoxBoxCollision(pointsA,axesA,pointsB,axesB):	
	#Sphere check
	
	if np.linalg.norm(pointsA[0]-pointsB[0])> (np.linalg.norm(pointsA[0]-pointsA[1])+np.linalg.norm(pointsB[0]-pointsB[1])):
		return False

	#TODO - SAT cuboid-cuboid collision check

	#Checking for collision along surface normals

	for i in range(3):
		if not CheckPointOverlap(pointsA,pointsB,axesA[i]):
			return False

	for j in range(3):
		if not CheckPointOverlap(pointsA,pointsB,axesB[j]):
			return False


	#Checking for edge-edge collisions
	for i in range(3):
		for j in range(3):
			if not CheckPointOverlap(pointsA,pointsB, np.cross(axesA[i],axesB[j])):
				return False


	return True
	


