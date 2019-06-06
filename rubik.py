from datetime import datetime
from os import path 
import abc
import multiprocessing
import random

from anytree import AnyNode
from numba import jit

import numpy as np


class Style:
	BASIC = 0


class Face:
	F = 0
	U = 1
	L = 2
	R = 3
	D = 4
	B = 5


class Action:
	ROT90 = 0
	ROT270 = 1
	SPIN90 = 2
	SPIN270 = 3	


class Edge(abc.ABC):
	
	@property 
	def index(self):
		return self._index

	@index.setter
	def index(self, value):
		self._index = value	

	@abc.abstractmethod
	def rot90(self, state):
		...
	
	@abc.abstractmethod
	def rot270(self, state):
		...

	def spin90(self, state):
		return np.copy(state)

	def spin270(self, state):
		return np.copy(state)


class CenterLR(Edge):
	
	def __init__(self, index=1):
		self._index = index
		
	def rot90(self, state):
		nextState = np.copy(state)
		nextState[Face.F, self.index, :] = state[Face.U, self.index, :]
		nextState[Face.D, self.index, :] = state[Face.F, self.index, :]
		nextState[Face.B, self.index, :] = state[Face.D, self.index, :]
		nextState[Face.U, self.index, :] = state[Face.B, self.index, :]			
		return nextState

	def rot270(self, state):
		nextState = np.copy(state)
		nextState[Face.F, self.index, :] = state[Face.D, self.index, :]
		nextState[Face.D, self.index, :] = state[Face.B, self.index, :]
		nextState[Face.B, self.index, :] = state[Face.U, self.index, :]
		nextState[Face.U, self.index, :] = state[Face.F, self.index, :]				
		return nextState


class CenterFB(Edge):

	def __init__(self, index=1):
		self._index = index
		
	def rot90(self, state):
		nextState = np.copy(state)
		nextState[Face.L, self.index, :] = state[Face.D, self.index, :]
		nextState[Face.D, self.index, :] = state[Face.R, self.index, :]
		nextState[Face.R, self.index, :] = state[Face.U, self.index, :]
		nextState[Face.U, self.index, :] = state[Face.L, self.index, :]			
		return nextState

	def rot270(self, state):
		nextState = np.copy(state)
		nextState[Face.L, self.index, :] = state[Face.U, self.index, :]
		nextState[Face.D, self.index, :] = state[Face.L, self.index, :]
		nextState[Face.R, self.index, :] = state[Face.D, self.index, :]
		nextState[Face.U, self.index, :] = state[Face.R, self.index, :]				
		return nextState


class CenterUD(Edge):

	def __init__(self, index=1):
		self._index = index

	def rot90(self, state):
		nextState = np.copy(state)
		nextState[Face.L, self.index, :] = state[Face.B, self.index, :]
		nextState[Face.F, self.index, :] = state[Face.L, self.index, :]
		nextState[Face.R, self.index, :] = state[Face.F, self.index, :]
		nextState[Face.B, self.index, :] = state[Face.R, self.index, :]			
		return nextState

	def rot270(self, state):
		nextState = np.copy(state)
		nextState[Face.L, self.index, :] = state[Face.F, self.index, :]
		nextState[Face.F, self.index, :] = state[Face.R, self.index, :]
		nextState[Face.R, self.index, :] = state[Face.B, self.index, :]
		nextState[Face.B, self.index, :] = state[Face.L, self.index, :]				
		return nextState


class FrontFace(Edge):

	def rot90(self, state):
		m = np.size(state, axis=1)
		nextState = np.copy(state)
		nextState[Face.F, :, :] = np.rot90(nextState[Face.F, :, :], 1)
		nextState[Face.L, m - 1, :] = state[Face.D, m - 1, :]
		nextState[Face.D, m - 1, :] = state[Face.R, m - 1, :]
		nextState[Face.R, m - 1, :] = state[Face.U, m - 1, :]
		nextState[Face.U, m - 1, :] = state[Face.L, m - 1, :]
		return nextState

	def rot270(self, state):
		m = np.size(state, axis=1)
		nextState = np.copy(state)
		nextState[Face.F, :, :] = np.rot90(nextState[Face.F, :, :], -1)
		nextState[Face.L, m - 1, :] = state[Face.U, m - 1, :]
		nextState[Face.D, m - 1, :] = state[Face.L, m - 1, :]
		nextState[Face.R, m - 1, :] = state[Face.D, m - 1, :]
		nextState[Face.U, m - 1, :] = state[Face.R, m - 1, :]

		return nextState
	
	def spin90(self, state):
		nextState = np.copy(state)
		nextState[Face.L, :, :] = state[Face.D, :, :]
		nextState[Face.D, :, :] = state[Face.R, :, :]
		nextState[Face.R, :, :] = state[Face.U, :, :]
		nextState[Face.U, :, :] = state[Face.L, :, :]
		return nextState

	def spin270(self, state):
		nextState = np.copy(state)
		nextState[Face.L, :, :] = state[Face.U, :, :]
		nextState[Face.D, :, :] = state[Face.L, :, :]
		nextState[Face.R, :, :] = state[Face.D, :, :]
		nextState[Face.U, :, :] = state[Face.R, :, :]
		return nextState
	

class BackFace(Edge):

	def rot90(self, state):
		nextState = np.copy(state)
		nextState[Face.B, :, :] = np.rot90(nextState[Face.B, :, :], 1)
		nextState[Face.L, 0, :] = state[Face.U, 0, :]
		nextState[Face.D, 0, :] = state[Face.L, 0, :]
		nextState[Face.R, 0, :] = state[Face.D, 0, :]
		nextState[Face.U, 0, :] = state[Face.R, 0, :]
		return nextState

	def rot270(self, state):
		nextState = np.copy(state)
		nextState[Face.B, :, :] = np.rot90(nextState[Face.B, :, :], -1)
		nextState[Face.L, 0, :] = state[Face.D, 0, :]
		nextState[Face.D, 0, :] = state[Face.R, 0, :]
		nextState[Face.R, 0, :] = state[Face.U, 0, :]
		nextState[Face.U, 0, :] = state[Face.L, 0, :]
		return nextState

	def spin90(self, state):
		nextState = np.copy(state)
		nextState[Face.L, :, :] = state[Face.U, :, :]
		nextState[Face.D, :, :] = state[Face.L, :, :]
		nextState[Face.R, :, :] = state[Face.D, :, :]
		nextState[Face.U, :, :] = state[Face.R, :, :]
		return nextState
		
	def spin270(self, state):
		nextState = np.copy(state)
		nextState[Face.L, :, :] = state[Face.D, :, :]
		nextState[Face.D, :, :] = state[Face.R, :, :]
		nextState[Face.R, :, :] = state[Face.U, :, :]
		nextState[Face.U, :, :] = state[Face.L, :, :]
		return nextState


class RightFace(Edge):

	def rot90(self, state):
		m = np.size(state, axis=1)
		nextState = np.copy(state)
		nextState[Face.R, :, :] = np.rot90(nextState[Face.R, :, :], 1)
		nextState[Face.F, m - 1, :] = state[Face.D, m - 1, :]
		nextState[Face.D, m - 1, :] = state[Face.B, m - 1, :]
		nextState[Face.B, m - 1, :] = state[Face.U, m - 1, :]
		nextState[Face.U, m - 1, :] = state[Face.F, m - 1, :]
		return nextState

	def rot270(self, state):
		m = np.size(state, axis=1)
		nextState = np.copy(state)
		nextState[Face.R, :, :] = np.rot90(nextState[Face.R, :, :], -1)
		nextState[Face.F, m - 1, :] = state[Face.U, m - 1, :]
		nextState[Face.D, m - 1, :] = state[Face.F, m - 1, :]
		nextState[Face.B, m - 1, :] = state[Face.D, m - 1, :]
		nextState[Face.U, m - 1, :] = state[Face.B, m - 1, :]
		return nextState

	def spin90(self, state):
		nextState = np.copy(state)
		nextState[Face.F, :, :] = state[Face.D, :, :]
		nextState[Face.D, :, :] = state[Face.B, :, :]
		nextState[Face.B, :, :] = state[Face.U, :, :]
		nextState[Face.U, :, :] = state[Face.F, :, :]
		return nextState
	
	def spin270(self, state):
		nextState = np.copy(state)
		nextState[Face.F, :, :] = state[Face.U, :, :]
		nextState[Face.D, :, :] = state[Face.F, :, :]
		nextState[Face.B, :, :] = state[Face.D, :, :]
		nextState[Face.U, :, :] = state[Face.B, :, :]
		return nextState


class LeftFace(Edge):

	def rot90(self, state):
		nextState = np.copy(state)
		nextState[Face.L, :, :] = np.rot90(nextState[Face.L, :, :], 1)
		nextState[Face.F, 0, :] = state[Face.U, 0, :]
		nextState[Face.D, 0, :] = state[Face.F, 0, :]
		nextState[Face.B, 0, :] = state[Face.D, 0, :]
		nextState[Face.U, 0, :] = state[Face.B, 0, :]
		return nextState

	def rot270(self, state):
		nextState = np.copy(state)
		nextState[Face.L, :, :] = np.rot90(nextState[Face.L, :, :], -1)
		nextState[Face.F, 0, :] = state[Face.D, 0, :]
		nextState[Face.D, 0, :] = state[Face.B, 0, :]
		nextState[Face.B, 0, :] = state[Face.U, 0, :]
		nextState[Face.U, 0, :] = state[Face.F, 0, :]
		return nextState

	def spin90(self, state):
		nextState = np.copy(state)
		nextState[Face.F, :, :] = state[Face.U, :, :]
		nextState[Face.D, :, :] = state[Face.F, :, :]
		nextState[Face.B, :, :] = state[Face.D, :, :]
		nextState[Face.U, :, :] = state[Face.B, :, :]
		return nextState

	def spin270(self, state):
		nextState = np.copy(state)
		nextState[Face.F, :, :] = state[Face.D, :, :]
		nextState[Face.D, :, :] = state[Face.B, :, :]
		nextState[Face.B, :, :] = state[Face.U, :, :]
		nextState[Face.U, :, :] = state[Face.F, :, :]
		return nextState
			

class DownFace(Edge):

	def rot90(self, state):
		n = np.size(state, axis=2)
		nextState = np.copy(state)
		nextState[Face.D, :, :] = np.rot90(nextState[Face.D, :, :], 1)
		nextState[Face.L, :, n - 1] = state[Face.B, :, n - 1]
		nextState[Face.F, :, n - 1] = state[Face.L, :, n - 1]
		nextState[Face.R, :, n - 1] = state[Face.F, :, n - 1]
		nextState[Face.B, :, n - 1] = state[Face.R, :, n - 1]

		return nextState

	def rot270(self, state):
		n = np.size(state, axis=2)
		nextState = np.copy(state)
		nextState[Face.D, :, :] = np.rot90(nextState[Face.D, :, :], -1)
		nextState[Face.L, :, n - 1] = state[Face.F, :, n - 1]
		nextState[Face.F, :, n - 1] = state[Face.R, :, n - 1]
		nextState[Face.R, :, n - 1] = state[Face.B, :, n - 1]
		nextState[Face.B, :, n - 1] = state[Face.L, :, n - 1]
		return nextState

	def spin90(self, state):
		nextState = np.copy(state)
		nextState[Face.L, :, :] = state[Face.B, :, :]
		nextState[Face.F, :, :] = state[Face.L, :, :]
		nextState[Face.R, :, :] = state[Face.F, :, :]
		nextState[Face.B, :, :] = state[Face.R, :, :]
		return nextState

	def spin270(self, state):
		nextState = np.copy(state)
		nextState[Face.L, :, :] = state[Face.F, :, :]
		nextState[Face.F, :, :] = state[Face.R, :, :]
		nextState[Face.R, :, :] = state[Face.B, :, :]
		nextState[Face.B, :, :] = state[Face.L, :, :]
		return nextState
		

class UpFace(Edge):

	def rot90(self, state):
		nextState = np.copy(state)
		nextState[Face.U, :, :] = np.rot90(nextState[Face.U, :, :], 1)
		nextState[Face.L, :, 0] = state[Face.F, :, 0]
		nextState[Face.F, :, 0] = state[Face.R, :, 0]
		nextState[Face.R, :, 0] = state[Face.B, :, 0]
		nextState[Face.B, :, 0] = state[Face.L, :, 0]
		return nextState

	def rot270(self, state):
		nextState = np.copy(state)
		nextState[Face.U, :, :] = np.rot90(nextState[Face.U, :, :], -1)
		nextState[Face.L, :, 0] = state[Face.B, :, 0]
		nextState[Face.F, :, 0] = state[Face.L, :, 0]
		nextState[Face.R, :, 0] = state[Face.F, :, 0]
		nextState[Face.B, :, 0] = state[Face.R, :, 0]
		return nextState

	def spin90(self, state):
		nextState = np.copy(state)
		nextState[Face.L, :, :] = state[Face.F, :, :]
		nextState[Face.F, :, :] = state[Face.R, :, :]
		nextState[Face.R, :, :] = state[Face.B, :, :]
		nextState[Face.B, :, :] = state[Face.L, :, :]
		return nextState

	def spin270(self, state):
		nextState = np.copy(state)
		nextState[Face.L, :, :] = state[Face.B, :, :]
		nextState[Face.F, :, :] = state[Face.L, :, :]
		nextState[Face.R, :, :] = state[Face.F, :, :]
		nextState[Face.B, :, :] = state[Face.R, :, :]
		return nextState

		
class Rubik(object):

	def __init__(self, style=Style.BASIC, size=[2, 2]):
		self._style = style
		self._size = size
		self.initState()

	@property 
	def style(self):
		return self._style

	@style.setter
	def style(self, value):
		self._style = value

	@property 
	def size(self):
		return self._size

	@size.setter
	def size(self, value):
		self._size = value

	@property 
	def edges(self):
		return self._edges

	@edges.setter
	def edges(self, value):
		self._edges = value

	@property 
	def state(self):
		return self._state

	@state.setter
	def state(self, value):
		self._state = value

	def resetState(self):
		self.state = np.copy(self.baseState)
		
	def initState(self):
		m = self.size[0]
		n = self.size[1]
		e = 6 + (m - 2) * 2 + (n - 2)

		if self.style == Style.BASIC:
			self.state = np.zeros(shape=(6, m, n), dtype=int)
			front = FrontFace()
			back = BackFace()
			left = LeftFace()
			right = RightFace()
			up = UpFace()
			down = DownFace()
			
			self.edges = {
				Face.F: front,
				Face.B: back,
				Face.L: left,
				Face.R: right,
				Face.U: up,
				Face.D: down
			}
			
			j = len(list(self.edges.keys()))
			for i in range(1, m - 1):
				self.edges.update({j:CenterLR(i)})
				j = j + 1
				self.edges.update({j:CenterFB(i)})
				j = j + 1
				
			for i in range(1, n - 1):
				self.edges.update({j:CenterUD(i)})
				j = j + 1
			
			for i in range(6):
				self.state[i, :, :] = i
			
			# print('e = %d self.edges = %d' % (e, len(self.edges)))
			
			self.baseState = np.copy(self.state)
	
	def isSolved(self):
		m = self.size[0]
		n = self.size[1]

		if self.style == Style.BASIC:
			for i in range(6):
				iFace = self.state[i, :, :]
				if np.sum(iFace) != (self.state[i, 0, 0] * m * n):
					return False
			return True
		else:
			return False

	def isNotSolved(self):
		result = not self.isSolved()
		return result
	
	def isIndependentFeature(self, i, j, k):
		m = self.size[0]
		n = self.size[1]
		if (self.style == Style.BASIC):
			if m % 2 == 1 and n % 2 == 1 and self.isCenterSquare(j, k):
				return False
			elif (i == Face.L or i == Face.R) and self.isCornerSquare(j, k):
				return False
			else: 
				return True
		else:
			return True
		
	def isCenterSquare(self, j, k):
		m = self.size[0]
		n = self.size[1]
		if (self.style == Style.BASIC and j == int(m / 2) and k == int(n / 2)):
			return True
		else:
			return False
		
	def isCornerSquare(self, j, k):
		m = self.size[0]
		n = self.size[1]
		if self.style == Style.BASIC and (j == 0 or j == m - 1) and (k == 0 or k == n - 1):
			return True
		else:
			return False		
	
	def printState(self):
		if self.style == Style.BASIC:
			print("Front Face: ")
			print(self.state[Face.F, :, :])
			print("Back Face: ")
			print(self.state[Face.B, :, :])
			print("Left Face: ")
			print(self.state[Face.L, :, :])
			print("Right Face: ")
			print(self.state[Face.R, :, :])
			print("Up Face: ")
			print(self.state[Face.U, :, :])
			print("Down Face: ")
			print(self.state[Face.D, :, :])

	def doAction(self, edge, action):
		edgeObject = self.edges.get(edge)
			
		if action == Action.ROT90:
			self.state = edgeObject.rot90(self.state)
		elif action == Action.ROT270:
			self.state = edgeObject.rot270(self.state)
		elif action == Action.SPIN90:
			self.state = edgeObject.spin90(self.state)
		elif action == Action.SPIN270:
			self.state = edgeObject.spin270(self.state)
	
	def doRandomActions(self, count=1, low=0, high=11):
		m = self.size[0]
		n = self.size[1]		
		e = 6 + (m - 2) * 2 + (n - 2)		
		for i in range(count):
			r = random.randint(low, high)
			edge = int(r / ((high + 1) / e))
			action = int(r / e)
			self.doAction(edge, action)
		self.latestAction = r
		return self

	def spinRubik(self, count):
		if self.style == Style.BASIC:
			m = self.size[0]
			n = self.size[1]		
			e = 6 + (m - 2) * 2 + (n - 2)			
			self.doRandomActions(count, low=2 * e, high=4 * e - 1)
	
	def getFrontFace(self):
		m = self.size[0]
		n = self.size[1]	
		f = Face.F
		if (self.style == Style.BASIC):
			fSum = 5 * m * n
			for i in range(6):
				iSum = np.sum(self.state[i, :, :])
				iMid = self.state[i, m % 2 + 1, n % 2 + 1]
				if m % 2 == 0 and n % 2 == 0:
					if iSum < fSum:
						fSum = iSum
						f = i
				elif m % 2 == 1 and n % 2 == 1:
					if iMid == Face.F:
						f = i
		return f

	def getFrontFaceAsString(self):
		f = self.getFrontFace()
		if(f == Face.F):
			return 'F'
		if(f == Face.B):
			return 'B'
		if(f == Face.L):
			return 'L'
		if(f == Face.R):
			return 'R'
		if(f == Face.U):
			return 'U'
		if(f == Face.D):
			return 'D'		
	
	def reorient(self, withFrontFace=None):
		if withFrontFace != None:
			f = withFrontFace
		else:
			f = self.getFrontFace()
		if(f == Face.F):
			pass
		elif(f == Face.B):
			self.doAction(Face.L, Action.SPIN90)
			self.doAction(Face.L, Action.SPIN90)
		elif(f == Face.U):
			self.doAction(Face.L, Action.SPIN90)
		elif(f == Face.D):
			self.doAction(Face.L, Action.SPIN270)
		elif(f == Face.L):
			self.doAction(Face.U, Action.SPIN270)
		elif(f == Face.R):
			self.doAction(Face.U, Action.SPIN90)

	def stateInCsv(self):
		row_csv = ""
		m = self.size[0]
		n = self.size[1]
		
		if self.style == Style.BASIC:
			for i in [Face.F, Face.B, Face.L, Face.R, Face.U, Face.D]:
				for j in range(m):
					for k in range(n):
						row_csv = row_csv + str(self.state[i, j, k]) + ","
			return row_csv[:-1]

	def setState(self, inputVector):
		m = self.size[0]
		n = self.size[1]
		
		newState = np.zeros(shape=(6, m, n), dtype=int)
		index = 0
		if self.style == Style.BASIC:
			for i in [Face.F, Face.B, Face.L, Face.R, Face.U, Face.D]:
				for j in range(m):
					for k in range(n):
						newState[i, j, k] = inputVector[index]
						index = index + 1
										
			self.state = np.copy(newState)


def consolidateTreeForNode(jNode, treeData, allowedActionMax, m, n, e, nodeDepth, exploreAllActions):
	oldState = jNode.state
	if exploreAllActions:
		jCube = Rubik(size=[m, n])
		exploredActions = treeData[oldState]['exploredActions']
		unexploredActions = [action for action in range(allowedActionMax + 1) if action not in exploredActions]
		for a in unexploredActions:
			jCube.setState(oldState.split(','))
			edge = int(a / ((allowedActionMax + 1) / e))
			action = int(a / e)
			jCube.doAction(edge, action)
			newState = jCube.stateInCsv()
			if newState in list(treeData.keys()):
				jNode_a = treeData[newState]['node']
				oldParent = treeData[newState]['parent']
				newParent = jNode
				if oldParent != jNode and oldParent.depth > nodeDepth:
					jNode_a.id = 'Action:' + str(a)
					# #detach iNode_a from children of oldParent
					oldParent.children = [c for c in oldParent.children if c.state != jNode_a.state]
					# #attache of iNode_a to children of newParent 
					jNode_a.parent = None
					jNode_a.parent = newParent
					treeData[newState]['parent'] = newParent
					treeData[oldState]['exploredActions'].append(a)
		
	return [treeData, oldState]


def consolidateTree(root, treeData, allowedActionMax, m, n, exploreAllActions=False):
		i = 1
		iParentNodes = [root] 
		# print('\tStarting: Tree Height = %d' % root.height)
		allNodes = 0
		fullNodes = 0
		partialNodes = 0
		exploredNodeActions = 0
		
		fullNodesInFirstPass = 0
		partialNodesInFirstPass = 0
		exploredNodeActionsInFirstPass = 0
		
		e = 6 + (m - 2) * 2 + (n - 2)		

		while i <= root.height:
			iNodes = []
			for pNode in iParentNodes:
				iNodes = iNodes + [c for c in pNode.children]
			# iNodes = findall_by_attr(root, value=i, name='depth')	
			for jNode in iNodes:	
				[treeData, oldState] = consolidateTreeForNode(jNode, treeData, allowedActionMax, m, n, e, i, exploreAllActions)
				exploredNodeActions = exploredNodeActions + len(treeData[oldState]['exploredActions'])
				if len(treeData[oldState]['exploredActions']) == allowedActionMax + 1:
					fullNodes = fullNodes + 1
				else:
					partialNodes = partialNodes + 1
				
				exploredNodeActionsInFirstPass = exploredNodeActionsInFirstPass + len(treeData[oldState]['exploredActionsInFirstPass'])				
				if len(treeData[oldState]['exploredActionsInFirstPass']) == allowedActionMax + 1:
					fullNodesInFirstPass = fullNodesInFirstPass + 1
				else:
					partialNodesInFirstPass = partialNodesInFirstPass + 1
			
			allNodes = allNodes + len(iNodes)
		
			iParentNodes = iNodes
			i = i + 1
		
		return [treeData, allNodes, fullNodes, partialNodes, fullNodesInFirstPass, partialNodesInFirstPass, exploredNodeActions, exploredNodeActionsInFirstPass]


def generateData(m, maxKValues, minActions, maxActions, filedir, returnDict):	
		result = dict()
		n = m
		e = 6 + (m - 2) * 2 + (n - 2)		
		allowedActionMin = 0
		allowedActionMax = 2 * e - 1
				
		maxK = maxKValues.get(m)
		data = dict() 
		treeData = dict()
	
		outpath = filedir + '/basic' + str(m) + 'x' + str(n) + 'RubikCube.csv'
	
		t1 = datetime.now()
		# print('\n=================================================================================================')
		print("\nGenerating data for Basic %dx%d Rubik's Cube" % (m, n))

		f = open(outpath, 'w')
		randomCube = Rubik(size=[m, n])
		root = AnyNode(id='root', state=None)		
		minSteps = np.Infinity
		maxSteps = 0
		solvedStateId = 0
		for k in range(maxK):
			shuffle = True
			randomCube.resetState()
			randomCube.spinRubik(e)
			kSolveState = randomCube.stateInCsv()
			try:
				kRoot = treeData[kSolveState]['node']
			except:
				kRootId = 'SolvedState:' + str(solvedStateId)
				solvedStateId = solvedStateId + 1
				kRoot = AnyNode(id=kRootId, parent=root, state=kSolveState)
				treeData.update({kSolveState:{'node':kRoot, 'parent':root, 'exploredActions':list(), 'exploredActionsInFirstPass':list() }})
				data.update({kRootId:kSolveState})
				f.write('0,' + kSolveState + '\n')
				
			while(shuffle):
				i = random.randint(minActions[m], maxActions[m])
				for j in range(i):
					oldState = randomCube.stateInCsv()
					newState = randomCube.doRandomActions(low=allowedActionMin, high=allowedActionMax).stateInCsv()
					try:
						newNode = treeData[newState]['node']
					except:
						oldNode = treeData[oldState]['node']
						treeData[oldState]['exploredActions'].append(randomCube.latestAction)
						treeData[oldState]['exploredActionsInFirstPass'].append(randomCube.latestAction)
						newNodeId = 'Action:' + str(randomCube.latestAction)
						newNode = AnyNode(id=newNodeId, parent=oldNode, state=newState)
						treeData.update({newState:{'node':newNode, 'parent':oldNode, 'exploredActions':list(), 'exploredActionsInFirstPass':list()}})

				try:
					key = list(data.values()).index(newState)
				except:
					if randomCube.isNotSolved():
						steps = newNode.depth - kRoot.depth
						f.write(str(steps) + ',' + newState + '\n')
						data.update({k:newState})
						if steps < minSteps:
							minSteps = steps
						if steps > maxSteps:
							maxSteps = steps						
						# print('\tBasic%dx%dRubikCube: k = %6d i = %4d moves = %4d minSteps = %4d maxSteps = %4d' % (m, n, k, i, steps, minSteps, maxSteps))
						shuffle = False
					else:
						minSteps = 0

		f.close()
	
		t2 = datetime.now()
		delta = t2 - t1
		result.update({'First Pass:Machine-Time(sec)                ':delta.total_seconds()})
		
		print("\nFirst Pass:Finished generating data for Basic %dx%d Rubik's Cube:Run Time = % d sec" % (m, n, delta.total_seconds()))
		# print(RenderTree(root))
		print("\nConsolidating generated Tree Data for Basic %dx%d Rubik's Cube" % (m, n))
		result.update({'\n\tPre Consolidation:Tree Height               ':root.height})
		
		consolidatedData = consolidateTree(root, treeData, allowedActionMax, m, n)
		
		[treeData, allNodes, fullNodes, partialNodes, fullNodesInFirstPass, partialNodesInFirstPass, exploredNodeActions, exploredNodeActionsInFirstPass] = consolidatedData
		t1 = t2

		result.update({'Post Consolidation:Tree Height              ':root.height})
	
		# print('\tEnding: Tree Height = %d' % root.height)

		t2 = datetime.now()
		delta = t2 - t1
		result.update({'Tree Consolidation:Machine-Time(sec)        ':delta.total_seconds()})

		print("\nFinished Consolidating generated Tree Data for Basic %dx%d Rubik's Cube:Run Time = %d sec" % (m, n, delta.total_seconds()))		
		t1 = t2
		outpath = filedir + '/basic' + str(m) + 'x' + str(n) + 'RubikCubeOptimized.csv'
		f = open(outpath, 'w')
		minSteps = np.Infinity
		maxSteps = 0
		i = 1
		for k in data.keys():
			kState = data[k]
			kNode = treeData[kState]['node']
			steps = kNode.depth - 1
			f.write(str(steps) + ',' + kState + '\n')
			if steps < minSteps:
				minSteps = steps
			if steps > maxSteps:
				maxSteps = steps			
			# print('\tBasic%dx%dRubikCubeOptimized: index = %6d moves = %4d minSteps = %4d maxSteps = %4d' % (m, n, i, steps, minSteps, maxSteps))
			i = i + 1
			
		t2 = datetime.now()
		delta = t2 - t1
		result.update({'\n\tSecond Pass:Machine-Time(sec)               ':delta.total_seconds()})		

		result.update({'\n\tTotal Nodes in the Tree                     ':allNodes})

		result.update({'Pre Consolidation:Fully Explored Nodes      ':fullNodesInFirstPass})
		result.update({'Pre Consolidation:Partially Explored Nodes  ':partialNodesInFirstPass})
		result.update({'Pre Consolidation:% of Fully Explored Nodes ':int(100 * fullNodesInFirstPass / allNodes)})
		result.update({'Pre Consolidation:% of Explored Actions     ':int(100 * exploredNodeActionsInFirstPass / (allNodes * (allowedActionMax + 1)))})
			
		result.update({'\n\tPost Consolidation:Fully Explored Nodes     ':fullNodes})
		result.update({'Post Consolidation:Partially Explored Nodes ':partialNodes})
		result.update({'Post Consolidation:% of Fully Explored Nodes':int(100 * fullNodes / allNodes)})
		result.update({'Post Consolidation:% of Explored Actions    ':int(100 * exploredNodeActions / (allNodes * (allowedActionMax + 1)))})
		
		print("\nSecond Pass:Finished generating data for Basic %dx%d Rubik's Cube:Run Time = %d sec" % (m, n, delta.total_seconds()))

		returnDict.update({m:result}) 


if __name__ == '__main__':
	
	filedir = path.dirname(path.abspath(__file__)).replace('\\', '/').replace('C:', '')
	
	maxKValues = {2:99988, 3:99988, 4:99988, 5:99988}
	# maxKValues = {2:1000, 3:1000, 4:100, 5:1000}
	minActions = {2:1, 3:1, 4:1, 5:1}
	maxActions = {2:100, 3:100, 4:100, 5:100}
	
	manager = multiprocessing.Manager()
	returnDict = manager.dict()
	
	jobs = dict()
	t1 = datetime.now()
	print('=================================================================================================')
	
	for m in range(2, 6):
		jobs.update({m:multiprocessing.Process(target=generateData , args=(m, maxKValues, minActions, maxActions, filedir, returnDict))})
		jobs[m].start()
	
	for m in range(2, 6):
		jobs[m].join()
	for m in range(2, 6):
		print('=================================================================================================')		
		print("Run-Time statistics for Basic %dx%d Rubik's Cube:" % (m, m))
		for key in returnDict.get(m).keys():
			print('\t%s = %10d' % (key, returnDict.get(m)[key]))
	t2 = datetime.now()
	delta = t2 - t1
	print('=================================================================================================')
	print("Finished generating data for all the Rubik's Cubes:Run Time(Clock) = % d sec" % delta.total_seconds())
