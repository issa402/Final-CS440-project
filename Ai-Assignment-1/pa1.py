import numpy as np
import random
from collections import deque 
from typing import Tuple, List

class Grid:
    def __init__(self, size = 101):
        self.size = size
        self.grid = np.zeros((size, size), dtype = int)
        self.maze()
    
    def neighbors(self, cell: Tuple[int, int]) -> List[Tuple[int,int]]:
        x , y = cell
        n = []
        for xDir, yDir in [(-1, 0), (0,1), (1, 0), (0,-1)]:
            nbrsX, nbrsY  = x + xDir, y + yDir #position of the neighbor of the cell
            if 0 <= nbrsX < self.size and 0 <= nbrsY < self.size:
                n.append(nbrsX, nbrsY)

    def maze(self): #Back tracking using DFS
        stack = deque()
        start = (random.randint(1, self.size -1), random.randint(1, self.size-1))#starting position of the random cell
        self.grid[start] = 1
        stack.append(start)

        while stack:
            current = stack.pop()
            n = []
            for neighbor in self.neighbors(current):
                if self.grid[neighbor] == 0:
                    n.append(neighbor)
            
            if n:
                stack.append(current)
                new = random.choice(n)
                if random.random() == 0.3 :
                    self.grid[new] = -1 #Blocked Cell
                else:
                    self.grid[new] = 1 #Unblock indentifier
                    stack.append(new)

                if not stack:
                    unvisited = np.argwhere(self.grid == 0)
                    if unvisited.size > 0:
                        start = tuple(unvisited[0])
                        self.grid[start]= 1
                        stack.append(start)
