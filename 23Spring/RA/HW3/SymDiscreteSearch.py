import numpy as np

def CheckCondition(state,condition):
	if (np.sum(np.multiply(state, condition))-np.sum(np.multiply(condition, condition)))==0:
		return True
	else:
		return False


def CheckVisited(state,vertices):
	for i in range(len(vertices)):
		if np.linalg.norm(np.subtract(state,vertices[i]))==0:
			return True
	return False


def ComputeNextState(state, effect):
	newstate=np.add(state, effect)
	return newstate


def Heuristic(state, GoalIndicesOneStep, GoalIndicesTwoStep): 
	score=0

	for idx in GoalIndicesOneStep:
		if state[idx[0]][idx[1]]==-1:
			score+=1

	for idx in GoalIndicesTwoStep:
		if state[idx[0]][idx[1]]==-1 and state[idx[0]][-1]==-1:
			score+=2
		elif state[idx[0]][idx[1]]==-1 and state[idx[0]][-1]==1:
			score+=1	
	return score


Predicates=['InHallway', 'InKitchen', 'InOffice', 'InLivingRoom', 'InGarden','InPantry','Chopped','OnRobot']

Objects=['Robot','Strawberry','Lemon', 'Paper', 'Knife'] 

nrPredicates=len(Predicates)
nrObjects=len(Objects)

def print_all():
	print("====================================")
	print("ActionPre: ", ActionPre)
	print("ActionEff: ", ActionEff)
	print("ActionDesc: ", ActionDesc)

ActionPre=[]
ActionEff=[]
ActionDesc=[]

###Move to hallway
for i in range(1,5,1):
	Precond=np.zeros([nrObjects, nrPredicates])
	Precond[0][0]=-1 #Robot not in hallway
	Precond[0][i]=1  #Robot in i-th room

	Effect=np.zeros([nrObjects, nrPredicates])
	Effect[0][0]=2.  #Robot in the hallway
	Effect[0][i]=-2. #Robot not in the i-th room

	ActionPre.append(Precond)
	ActionEff.append(Effect)
	ActionDesc.append("Move to InHallway from "+Predicates[i])
# print_all()
###Move to room
for i in range(1,5,1):
	Precond=np.zeros([nrObjects, nrPredicates])
	Precond[0][0]=1  #Robot in the hallway
	Precond[0][i]=-1 #Robot not in the ith room

	Effect=np.zeros([nrObjects, nrPredicates])
	Effect[0][0]=-2. #Robot not in the hallway
	Effect[0][i]=2.  #Robot in the ith room

	ActionPre.append(Precond)
	ActionEff.append(Effect)
	ActionDesc.append("Move to "+Predicates[i]+" from InHallway")


###Move to Pantry TODO
Precond=np.zeros([nrObjects, nrPredicates])
Precond[0][1]=1  #Robot in the kitchen
Precond[0][5]=-1 #Robot not in the pantry

Effect=np.zeros([nrObjects, nrPredicates])
Effect[0][1]=-2. #Robot not in the kitchen
Effect[0][5]= 2#Robot in the pantry

ActionPre.append(Precond)
ActionEff.append(Effect)
ActionDesc.append("Move to Pantry from Kitchen")


###Move from Pantry TODO
Precond=np.zeros([nrObjects, nrPredicates])
Precond[0][1]= -1#Robot not in the kitchen
Precond[0][5]= 1#Robot in the pantry

Effect=np.zeros([nrObjects, nrPredicates])
Effect[0][1]=2. #Robot in the kitchen
Effect[0][5]=-2. #Robot not in the pantry

ActionPre.append(Precond)
ActionEff.append(Effect)
ActionDesc.append("Move to Kitchen from Pantry")

###Cut fruit in kitchen TODO
for j in [1,2]:
	Precond=np.zeros([nrObjects, nrPredicates])
	Precond[0][1]=1 #Robot in the kitchen
	Precond[j][1]=1 #Fruit in the kitchen
	Precond[4][1]=1 #Knife in the kitchen
	Precond[j][6]=-1 #Fruit not chopped


	Effect=np.zeros([nrObjects, nrPredicates])
	Effect[j][6]=2. #Fruit chopped

	ActionPre.append(Precond)
	ActionEff.append(Effect)
	ActionDesc.append("Cut "+Objects[j]+" in the kitchen")


###Pickup object
for i in range(1,6,1):
	for j in range(1,5,1):
		Precond=np.zeros([nrObjects, nrPredicates])
		Precond[0][i]=1 #Robot in ith room
		Precond[j][i]=1 #Object j in ith room
		Precond[j][-1]=-1 #Object j not on robot

		Effect=np.zeros([nrObjects, nrPredicates])
		Effect[j][i]=-2 #Object j not in ith room
		Effect[j][-1]=2 # Object j on robot

		ActionPre.append(Precond)
		ActionEff.append(Effect)
		ActionDesc.append("Pick up "+Objects[j]+" from "+Predicates[i])


###Place object
for i in range(1,6,1):
	for j in range(1,5,1):
		Precond=np.zeros([nrObjects, nrPredicates])
		Precond[0][i]=1 #Robot in ith room
		Precond[j][i]=-1 #Object j not in ith room
		Precond[j][-1]=1 #Object j on robot

		Effect=np.zeros([nrObjects, nrPredicates])
		Effect[j][i]=2.  #Object j in ith room
		Effect[j][-1]=-2 #Object j not on robot

		ActionPre.append(Precond)
		ActionEff.append(Effect)
		ActionDesc.append("Place "+Objects[j]+" at "+Predicates[i])



InitialState=-1*np.ones([nrObjects, nrPredicates])
InitialState[0][0]=1 # Robot is in the hallway
InitialState[1][4]=1 # Strawberry is in the garden
InitialState[2][5]=1 # Lemon is in the pantry
InitialState[3][2]=1 # Paper is in the office
InitialState[4][2]=1 # Knife is in the office

GoalState=np.zeros([nrObjects, nrPredicates])
GoalState[0][1]=1 # Robot is in the kitchen
GoalState[1][1]=1 # Strawberry is in the kitchen
GoalState[2][4]=1 # Lemon is in the Garden
GoalState[1][6]=1 # Strawberry is chopped

GoalIndicesOneStep=[[0,1],[1,6]]
GoalIndicesTwoStep=[[1,1],[2,4]]

np.random.seed(13)


# Search for Solution
vertices=[]
parent=[]
action=[]

cost2come=[]

Queue=[]
Queue.append(0)
vertices.append(InitialState)
parent.append(0)
action.append(-1)
cost2come.append(0)

FoundPath=False

# Dijkstra
# while len(Queue)>0:
# 	#TODO perform search
# 	P = np.array([cost2come[q] for q in Queue])
# 	id = P.argmin()
#
# 	x = Queue[id]
# 	del Queue[id]
#
# 	# Check if Goal
# 	if CheckCondition(vertices[x],GoalState):
# 		FoundPath=True
# 		break
#
# 	for i in range(len(ActionPre)):
# 		if CheckCondition(vertices[x],ActionPre[i]):
# 			if not CheckVisited(ComputeNextState(vertices[x],ActionEff[i]),vertices):
# 				vertices.append(ComputeNextState(vertices[x],ActionEff[i]))
# 				parent.append(x)
# 				action.append(i)
# 				cost2come.append(cost2come[x]+1)
# 				Queue.append(len(vertices)-1)

# A*
while len(Queue)>0:
	#TODO perform search
	P = np.array([cost2come[q] +Heuristic(vertices[q], GoalIndicesOneStep, GoalIndicesTwoStep) for q in Queue])
	id = P.argmin()

	x = Queue[id]
	del Queue[id]

	# Check if Goal
	if CheckCondition(vertices[x],GoalState):
		FoundPath=True
		break

	for i in range(len(ActionPre)):
		if CheckCondition(vertices[x],ActionPre[i]):
			if not CheckVisited(ComputeNextState(vertices[x],ActionEff[i]),vertices):
				vertices.append(ComputeNextState(vertices[x],ActionEff[i]))
				parent.append(x)
				action.append(i)
				cost2come.append(cost2come[x]+1)
				# cost2come.append(cost2come[x]+1+Heuristic(vertices[x], GoalIndicesOneStep, GoalIndicesTwoStep) )
				Queue.append(len(vertices)-1)





# Print Plan
print("\n FoundPath: ", FoundPath," ", len(vertices))

Plan=[]
if FoundPath:
	while not x==0:
		Plan.insert(0,action[x])
		x=parent[x]

for i in range(len(Plan)):
	print(ActionDesc[Plan[i]])


'''

 FoundPath:  True   5894
Move to InKitchen from InHallway
Move to Pantry from Kitchen
Pick up Lemon from InPantry
Move to Kitchen from Pantry
Move to InHallway from InKitchen
Move to InOffice from InHallway
Pick up Knife from InOffice
Move to InHallway from InOffice
Move to InGarden from InHallway
Pick up Strawberry from InGarden
Place Lemon at InGarden
Move to InHallway from InGarden
Move to InKitchen from InHallway
Place Strawberry at InKitchen
Place Knife at InKitchen
Cut Strawberry in the kitchen


'''

'''
 FoundPath:  True   2415
Move to InGarden from InHallway
Pick up Strawberry from InGarden
Move to InHallway from InGarden
Move to InKitchen from InHallway
Place Strawberry at InKitchen
Move to Pantry from Kitchen
Pick up Lemon from InPantry
Move to Kitchen from Pantry
Move to InHallway from InKitchen
Move to InGarden from InHallway
Place Lemon at InGarden
Move to InHallway from InGarden
Move to InOffice from InHallway
Pick up Knife from InOffice
Move to InHallway from InOffice
Move to InKitchen from InHallway
Place Knife at InKitchen
Cut Strawberry in the kitchen

Process finished with exit code 0

'''

'''
FoundPath:  True   2005
Move to InKitchen from InHallway
Move to Pantry from Kitchen
Pick up Lemon from InPantry
Move to Kitchen from Pantry
Move to InHallway from InKitchen
Move to InGarden from InHallway
Pick up Strawberry from InGarden
Place Lemon at InGarden
Move to InHallway from InGarden
Move to InOffice from InHallway
Pick up Knife from InOffice
Move to InHallway from InOffice
Move to InKitchen from InHallway
Place Strawberry at InKitchen
Place Knife at InKitchen
Cut Strawberry in the kitchen
'''


