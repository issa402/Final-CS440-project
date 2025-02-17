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
