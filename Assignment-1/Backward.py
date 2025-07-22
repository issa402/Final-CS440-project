import heapq
from Grid import Grid
import numpy as np

class Repeated:
    def __init__(self, maze, mode = "high g-value"):
        self.maze = maze.grid
        self.seen = set() #track block cells
        self.mode = mode #use for tie breaking strategies
        self.number = 10**6 #multplier so we dont encounter a negative f value

    def  find(self, startPos , targetPos, backwards = False):
        
        if backwards:
            startPos, targetPos = targetPos, startPos
        
        current = startPos
        max_moves =  0 
        cells_passed = 0
        full_path = [startPos]

        #loop for reaching the target
        while current != targetPos:
            heap  = [] #priority queue
            path_history  = {} #Parent pointers 
            g_costs = {current : 0}
            counter = 0 #cells processsed

            f_start =  self.estimate_distance(current, targetPos) #use start position to begin search
            heapq.heappush(heap, self.priority(f_start, 0, current))
            found_target =  False
            while heap :
                _, node = heapq.heappop(heap)
                counter += 1

                if node  == targetPos:
                    found_target  = True
                    break

                for adj in self.find_adj(node): #go through neighbors
                    if adj in self.seen:#Skip known blocked cells
                        continue
                    new_g = g_costs[node] + 1 #cost incrmented to one to next cell
                    if adj not in g_costs or new_g < g_costs[adj]: 
                        g_costs[adj] = new_g
                        f = new_g + self.estimate_distance(adj, targetPos)
                        heapq.heappush(heap, self.priority(f, new_g, adj))
                        path_history[adj] = node
            cells_passed +=  counter
            max_moves += 1

            if not found_target:
                return None, max_moves, cells_passed
            
            whole_path = self.follow_path(path_history, current, targetPos)#only tracked recent path since path_history didnt account for old steps
            go = True

            for next_step in whole_path[1:]: #Skips the current position
                if self.maze[next_step[0], next_step[1]] ==-1:
                    self.seen.add(next_step)
                    go = False
                    break
                full_path.append(next_step)
                current  = next_step
            if go:
                return full_path, max_moves, cells_passed
            
        return full_path, max_moves, cells_passed






        

    def estimate_distance(self,  pointA ,  pointB): # Manhattan distance 
        xDiff = abs(pointA[0]- pointB[0])
        yDiff = abs(pointA[1] - pointB[1])
        return xDiff + yDiff

    def priority(self, f_val, g_val, location): #tie breaker for the priority queue
        if self.mode == "high g-value" :
            return(self.number  * f_val - g_val , location)
        else:
            return(self.number * f_val + g_val, location) 
    
    def find_adj(self, position):#check neighboring cell validation
        adjacent = []
        row = position[0]
        col = position[1]

        directions = [
            (-1, 0),
            (1,0),
            (0 ,-1),
            (0,1)
        ]
        for dr, dc in directions:
            new_row = row + dr
            new_col = col + dc

            valid_row = 0 <= new_row < self.maze.shape[0]
            valid_col = 0 <= new_col< self.maze.shape[1]

            if valid_row and valid_col:
                adjacent.append((new_row, new_col))
        return adjacent
    
    def  follow_path(self, history, beginning, destination):
        reversed_path = [destination]
        while reversed_path[-1] != beginning: #works its way backwards from target to start
            reversed_path.append(history[reversed_path[-1]])
        return reversed_path[::-1]
    
def compare(worlds):
    strategy = {
        'forward' : {'counts': [], 'wins': 0},
        'backwards' : {'counts': [], 'wins' : 0}
    }
    for world in worlds:
        forward_agent = Repeated(world, 'high g-value') 
        path, _, cellnum = forward_agent.find((0,0), (100,100))
        if path is not None:
            strategy['forward']['wins'] +=1
        strategy['forward']['counts'].append(cellnum)

        backwards_agent = Repeated(world, 'high g-value')
        path_backwards, _, cellmax = backwards_agent.find((0,0), (100, 100), backwards = True)
        if path_backwards is not None:
            strategy['backwards']['wins'] +=1
        strategy['backwards']['counts'].append(cellmax)


    average_forward = sum(strategy['forward']['counts'])/50
    average_backwards = sum(strategy['backwards']['counts'])/50

    print("Forward A*:")
    print(f"Succesful paths: {strategy['forward']['wins']}")
    print(f"Average cells : {average_forward:0f}")
    print("\nBackwards A* Results:")
    print(f"Succesful paths:{strategy['backwards']['wins']}")
    print(f"Average cells :{average_backwards:0f}")
    print(f"\nBackwards A*saves {average_backwards - average_forward:.0f} checks({(average_backwards/average_forward -1)*100:.1f}% better)")


if __name__ == "__main__":
    loaded = [Grid(101) for _ in range(50)]
    for i, maze in enumerate(loaded):
        maze.grid = np.load(f"gridworlds/gridworld_{i:02d}.npy")

    compare(loaded)
