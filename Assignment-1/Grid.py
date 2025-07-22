import numpy as np
import random
from collections import deque 
from typing import Tuple, List
import matplotlib.pyplot as plt
import os

class Grid:
    def __init__(self, size = 101):
        """Sets gridworld with size of 101 x 101"""
        self.size = size
        self.grid = np.zeros((size, size), dtype = int) # 0 is unvisited , 1 is open and -1 is blocked
        self.maze()
    
    def neighbors(self, cell: tuple) -> list:
        x = cell[0] #Coordinates from cell tuple
        y = cell[1]
        n = [] #list for neighboring cell
        directions  = [(-1, 0), (1,0), (0, -1), (0, 1)] 
        for direction in directions:
            nbrsX = x + direction[0] #grabs the coordinate of the neighbor
            nbrsY = y + direction[1]
            #cheks if coordinate are in bound
            if 0 <=  nbrsX < self.size and 0 <= nbrsY < self.size: 
                    n.append(( nbrsX, nbrsY))
        return n


    def maze(self): #Back tracking using DFS
        stack = deque() # stack for the DFS algo
        start = (random.randint(0, self.size -1), random.randint(0, self.size-1))#starting position of the random cell
        self.grid[start] = 1 #open cell
        stack.append(start)

        while True:
            while stack:
                current = stack[-1] 
                n = [] #list for unvisited neighbor
                for neighbor in self.neighbors(current): #for loop to check neighbors of the current
                    if self.grid[neighbor] == 0: #looks for unvisited
                        n.append(neighbor)
                #if the unvisited neighbor exists it gives its property whether its a blocked or unblocked
                if n:
                    random.shuffle(n)
                    new =  n[0]
                    if random.random() < 0.3 :
                        self.grid[new] = -1 #Blocked Cell
                    else:
                        self.grid[new] = 1 #Unblock indentifier
                        stack.append(new)
                else:
                    stack.pop() #back track

                
            unvisited = np.argwhere(self.grid == 0)#looks for remaining unvisited cells after DFS
            if unvisited.size == 0: #If all cells are visited
                    break
            start = tuple(unvisited[0]) #starts the DFS from first unvisisted cell
            self.grid[start]= 1
            stack.append(start)

    def view(self) -> None:
        cmap = plt.cm.colors.ListedColormap(['black', 'gray', 'white'])
        bounds = [ -1.5, -0.5, 0.5, 1.5] 
        norm  = plt.cm.colors.BoundaryNorm(bounds, cmap.N)

        
        plt.figure(figsize=(10,10))  
        plt.imshow(self.grid, cmap=cmap, norm=norm, interpolation='none')
        plt.title(f"Maze {self.size}x{self.size}")  
        plt.show()  

    def save(self, filename: str) -> None:
        np.save(filename, self.grid)  

def generate(num_grids: int = 50, size: int = 101, save_dir: str = 'gridworlds') -> None:
   
    os.makedirs(save_dir, exist_ok=True)  
    
    for i in range(num_grids):
        grid = Grid(size)  
        
        grid.save(f"{save_dir}/gridworld_{i:02d}.npy")
        
        print(f"Grid {i+1}/{num_grids} generated")

def loadGrid(filename: str) -> np.ndarray:
    
    return np.load(filename)  

def visualize(grid: np.ndarray) -> None:

    
    cmap = plt.cm.colors.ListedColormap(['black', 'gray', 'white'])
    bounds = [-1.5, -0.5, 0.5, 1.5]
    norm = plt.cm.colors.BoundaryNorm(bounds, cmap.N)

    
    plt.figure(figsize=(10,10))
    plt.imshow( grid, cmap=cmap,  norm=norm, interpolation='none') 
    plt.title("Loaded gridworld ")
    plt.show()


if __name__ == "__main__":
    
    generate()
    
    
    load = loadGrid("gridworlds/gridworld_00.npy")  
    visualize(load)  


         