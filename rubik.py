from datetime import datetime
from os import path 
import abc
import random
import sys

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

	
class InnerLayer:
		O = 0
		P = 1


class Action:
	ROT90 = 0
	ROT270 = 1
	SPIN90 = 2
	SPIN270 = 3	


class Edge(abc.ABC):

	@abc.abstractmethod
	def rot90(self):
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
		
	@property 
	def index(self):
		return self._index

	@index.setter
	def index(self, value):
		self._index = value

	def rot90(self, state):
		m = np.size(state, axis=1)
		n = np.size(state, axis=2)
		nextState = np.copy(state)
		nextState[Face.F, self.index, :] = state[Face.U, self.index, :]
		nextState[Face.D, self.index, :] = state[Face.F, self.index, :]
		nextState[Face.B, self.index, :] = state[Face.D, self.index, :]
		nextState[Face.U, self.index, :] = state[Face.B, self.index, :]			
		return nextState

	def rot270(self, state):
		m = np.size(state, axis=1)
		n = np.size(state, axis=2)
		nextState = np.copy(state)
		nextState[Face.F, self.index, :] = state[Face.D, self.index, :]
		nextState[Face.D, self.index, :] = state[Face.B, self.index, :]
		nextState[Face.B, self.index, :] = state[Face.U, self.index, :]
		nextState[Face.U, self.index, :] = state[Face.F, self.index, :]				
		return nextState


class CenterFB(Edge):

	def __init__(self, index=1):
		self._index = index
		
	@property 
	def index(self):
		return self._index

	@index.setter
	def index(self, value):
		self._index = value

	def rot90(self, state):
		m = np.size(state, axis=1)
		n = np.size(state, axis=2)
		nextState = np.copy(state)
		nextState[Face.L, self.index, :] = state[Face.D, self.index, :]
		nextState[Face.D, self.index, :] = state[Face.R, self.index, :]
		nextState[Face.R, self.index, :] = state[Face.U, self.index, :]
		nextState[Face.U, self.index, :] = state[Face.L, self.index, :]			
		return nextState

	def rot270(self, state):
		m = np.size(state, axis=1)
		n = np.size(state, axis=2)
		nextState = np.copy(state)
		nextState[Face.L, self.index, :] = state[Face.U, self.index, :]
		nextState[Face.D, self.index, :] = state[Face.L, self.index, :]
		nextState[Face.R, self.index, :] = state[Face.D, self.index, :]
		nextState[Face.U, self.index, :] = state[Face.R, self.index, :]				
		return nextState


class CenterUD(Edge):

	def __init__(self, index=1):
		self._index = index
		
	@property 
	def index(self):
		return self._index

	@index.setter
	def index(self, value):
		self._index = value

	def rot90(self, state):
		m = np.size(state, axis=1)
		n = np.size(state, axis=2)
		nextState = np.copy(state)
		nextState[Face.L, self.index, :] = state[Face.B, self.index, :]
		nextState[Face.F, self.index, :] = state[Face.L, self.index, :]
		nextState[Face.R, self.index, :] = state[Face.F, self.index, :]
		nextState[Face.B, self.index, :] = state[Face.R, self.index, :]			
		return nextState

	def rot270(self, state):
		m = np.size(state, axis=1)
		n = np.size(state, axis=2)
		nextState = np.copy(state)
		nextState[Face.L, self.index, :] = state[Face.F, self.index, :]
		nextState[Face.F, self.index, :] = state[Face.R, self.index, :]
		nextState[Face.R, self.index, :] = state[Face.B, self.index, :]
		nextState[Face.B, self.index, :] = state[Face.L, self.index, :]				
		return nextState


class FrontFace(Edge):

	def rot90(self, state):
		m = np.size(state, axis=1)
		n = np.size(state, axis=2)
		nextState = np.copy(state)
		nextState[Face.F, :, :] = np.rot90(nextState[Face.F, :, :], 1)
		nextState[Face.L, m - 1, :] = state[Face.D, m - 1, :]
		nextState[Face.D, m - 1, :] = state[Face.R, m - 1, :]
		nextState[Face.R, m - 1, :] = state[Face.U, m - 1, :]
		nextState[Face.U, m - 1, :] = state[Face.L, m - 1, :]
		return nextState

	def rot270(self, state):
		m = np.size(state, axis=1)
		n = np.size(state, axis=2)
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
		m = np.size(state, axis=1)
		n = np.size(state, axis=2)
		nextState = np.copy(state)
		nextState[Face.B, :, :] = np.rot90(nextState[Face.B, :, :], 1)
		nextState[Face.L, 0, :] = state[Face.U, 0, :]
		nextState[Face.D, 0, :] = state[Face.L, 0, :]
		nextState[Face.R, 0, :] = state[Face.D, 0, :]
		nextState[Face.U, 0, :] = state[Face.R, 0, :]
		return nextState

	def rot270(self, state):
		m = np.size(state, axis=1)
		n = np.size(state, axis=2)
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
		n = np.size(state, axis=2)
		nextState = np.copy(state)
		nextState[Face.R, :, :] = np.rot90(nextState[Face.R, :, :], 1)
		nextState[Face.F, m - 1, :] = state[Face.D, m - 1, :]
		nextState[Face.D, m - 1, :] = state[Face.B, m - 1, :]
		nextState[Face.B, m - 1, :] = state[Face.U, m - 1, :]
		nextState[Face.U, m - 1, :] = state[Face.F, m - 1, :]
		return nextState

	def rot270(self, state):
		m = np.size(state, axis=1)
		n = np.size(state, axis=2)
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
		m = np.size(state, axis=1)
		n = np.size(state, axis=2)
		nextState = np.copy(state)
		nextState[Face.L, :, :] = np.rot90(nextState[Face.L, :, :], 1)
		nextState[Face.F, 0, :] = state[Face.U, 0, :]
		nextState[Face.D, 0, :] = state[Face.F, 0, :]
		nextState[Face.B, 0, :] = state[Face.D, 0, :]
		nextState[Face.U, 0, :] = state[Face.B, 0, :]
		return nextState

	def rot270(self, state):
		m = np.size(state, axis=1)
		n = np.size(state, axis=2)
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
		m = np.size(state, axis=1)
		n = np.size(state, axis=2)
		nextState = np.copy(state)
		nextState[Face.D, :, :] = np.rot90(nextState[Face.D, :, :], 1)
		nextState[Face.L, :, n - 1] = state[Face.B, :, n - 1]
		nextState[Face.F, :, n - 1] = state[Face.L, :, n - 1]
		nextState[Face.R, :, n - 1] = state[Face.F, :, n - 1]
		nextState[Face.B, :, n - 1] = state[Face.R, :, n - 1]

		return nextState

	def rot270(self, state):
		m = np.size(state, axis=1)
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
		m = np.size(state, axis=1)
		n = np.size(state, axis=2)
		nextState = np.copy(state)
		nextState[Face.U, :, :] = np.rot90(nextState[Face.U, :, :], 1)
		nextState[Face.L, :, 0] = state[Face.F, :, 0]
		nextState[Face.F, :, 0] = state[Face.R, :, 0]
		nextState[Face.R, :, 0] = state[Face.B, :, 0]
		nextState[Face.B, :, 0] = state[Face.L, :, 0]
		return nextState

	def rot270(self, state):
		m = np.size(state, axis=1)
		n = np.size(state, axis=2)
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

	def initState(self):
		m = self.size[0]
		n = self.size[1]

		if self.style == Style.BASIC:
			self.state = np.zeros(shape=(6, m, n), dtype=int)
			front = FrontFace()
			back = BackFace()
			left = LeftFace()
			right = RightFace()
			up = UpFace()
			down = DownFace()
			
			centerLR = CenterLR(m % 2)
			centerFB = CenterFB(m % 2)
			centerUD = CenterUD(n % 2)

			self.edges = {
				Face.F: front,
				Face.B: back,
				Face.L: left,
				Face.R: right,
				Face.U: up,
				Face.D: down
			}
			
			for i in range(6):
				self.state[i, :, :] = i
			self.spinRubik(np.size(self.state))
	
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
	
	def doRandomActions(self, count, low=0, high=11):
		m = self.size[0]
		n = self.size[1]		
		e = 6 + (m - 2) * 2 + (n - 2)		
		for i in range(count):
			r = random.randint(low, high)
			edge = r % ((high + 1) / e)
			action = int(r / e)
			self.doAction(edge, action)
		self.reorient()
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
				if iSum < fSum and m % 2 == 0 and n % 2 :
					fSum = iSum
					f = i
				if iMid == Face.F and m % 2 == 1 and n % 2 == 1:
					f = i
		return f
	
	def reorient(self):
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
			for i in range(6):
				for j in range(m):
					for k in range(n):
						row_csv = row_csv + str(self.state[i, j, k]) + ","
			
			return row_csv[:-1]
			

def randNormInt(low, high, count):

	mid = (low + high) / 2
	randomNums = np.random.normal(loc=mid, scale=(high - mid) / 4 , size=count)
	return np.round(randomNums)


if __name__ == '__main__':
	
	m = 2
	n = 2
	e = 6 + (m - 2) * 2 + (n - 2)
	maxK = 100000
	actionsCount = randNormInt(1, 100, maxK)
	data = dict() 
	
	filedir = path.dirname(path.abspath(__file__)).replace('\\', '/').replace('C:', '')
	outpath = filedir + '/basic' + str(m) + 'x' + str(n) + 'RubikCube.csv'
	
	t1 = datetime.now()
	
	print("Generating data for Basic %dx%d Rubik's Cube" % (m, n))

	f = open(outpath, 'w')

	for k in range(maxK):
		shuffle = True
		randomCube = Rubik(size=[m, n])
		
		while(shuffle):
			i = random.randint(0, maxK - 1)
			newState = randomCube.doRandomActions(int(actionsCount[i]), low=0, high=2 * e - 1).stateInCsv()
			
			try:
				key = data.values().index(newState)
			except:
				if randomCube.isNotSolved():
					f.write(newState + '\n')
					data.update({k:newState})
					shuffle = False
	
	t2 = datetime.now()
	delta = t2 - t1
	print("Finished generating data for Basic %dx%d Rubik's Cube:\n\t runtime = % d sec" % (m, n, delta.total_seconds()))

	f.close()
					
