from hex_engine import hexPosition
import time
import random
from math import sqrt, log
from copy import copy, deepcopy
from sys import stderr
from queue import Queue
import numpy as np
inf = float('inf')

class node:
	def __init__(self, move = None, parent = None):
		self.move = move
		self.parent = parent
		self.N = 0 
		self.Q = 0 
		self.children = []
		self.outcome = 0

	def add_children(self, children):
		self.children += children

	def set_outcome(self, outcome):
		self.outcome = outcome

	def value(self):
		if(self.N == 0):
			return inf
		else:
			return self.Q/self.N + sqrt(2*log(self.parent.N)/self.N)


class mctsagent:

	def __init__(self, state):
		self.rootstate = deepcopy(state)
		self.root = node()

	def best_move(self):
		if(self.rootstate.calc_winner() != 0):
			return -1

		max_value = max(self.root.children, key = lambda n: n.N).N
		max_nodes = [n for n in self.root.children if n.N == max_value]
		bestchild = random.choice(max_nodes)
		return bestchild.move

	def get_float_matrix(self):
		
		size = self.rootstate.size
		Y = np.zeros(shape=(size,size),dtype=np.single)
		for c in self.root.children:
			m = c.move
			Y[m[0]][m[1]] = np.single(c.N) 

		return Y

	def move(self, move):
		for child in self.root.children:
			if move == child.move:
				child.parent = None
				self.root = child
				self.rootstate.play(child.move)
				return

		self.rootstate.play(move)
		self.root = node()

	def search(self, roll_outs):
		
		num_rollouts = 0

		while(num_rollouts < roll_outs):
			node, state = self.select_node()
			turn = state.player
			outcome = self.roll_out(state)
			self.backup(node, turn, outcome)
			num_rollouts += 1


	def select_node(self):
		node = self.root
		state = deepcopy(self.rootstate)

		while(len(node.children)!=0):
			max_value = max(node.children, key = lambda n: n.value()).value()
			max_nodes = [n for n in node.children if n.value() == max_value]
			node = random.choice(max_nodes)
			state.play(node.move)
			
			if node.N == 0:
				return (node, state)

		if(self.expand(node, state)):
			node = random.choice(node.children)
			state.play(node.move)
		return (node, state)

	def roll_out(self, state):
		moves = state.getActionSpace()

		while(state.calc_winner() == 0):
			move = random.choice(moves)
			state.play(move)
			moves.remove(move)

		return state.calc_winner()

	def backup(self, node, turn, outcome):
		reward = -1 if outcome == turn else 1

		while node!=None:
			node.N += 1
			node.Q +=reward
			reward = -reward
			node = node.parent

	def expand(self, parent, state):
		children = []
		if(state.calc_winner() != 0):
			return False


		for move in state.getActionSpace():
			children.append(node(move, parent))

		parent.add_children(children)
		return True

	def set_gamestate(self, state):
		self.rootstate = deepcopy(state)
		self.root = node()

	def tree_size(self):
		Q = Queue()
		count = 0
		Q.put(self.root)
		while not Q.empty():
			node = Q.get()
			count +=1
			for child in node.children:
				Q.put(child)
		return count
			
