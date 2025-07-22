import heapq
from Grid import Grid


class Repeated:
    def __init__(self, maze, mode = "high g-value"):
        self.maze = maze.grid
        self.seen = set()
        self.mode = mode
        self.number = 10**6

    def find(self, startPos , targetPos):
        current = startPos
        max_moves = 0
        cells_passed = 0

        while current != targetPos:
            heap  = []
            path_history  = {}
            g_costs = {current : 0}
            counter = 0

            f_start = self.estimate_distance(current, targetPos)
            heapq.heappush(heap, self.priority(f_start, 0, current))
            found_target = False
            while heap:
                priority, node = heapq.heappop(heap)
                counter += 1

                if node == targetPos:
                    found_target = True
                    break

                for adj in self.find_adj(node):
                    if adj in self.seen:
                        continue
                    new_g = g_costs[node] + 1
                    if adj not in g_costs or new_g < g_costs[adj]:
                        g_costs[adj] = new_g
                        f = new_g + self.estimate_distance(adj, targetPos)
                        heapq.heappush(heap, self.priority(f, new_g, adj))
                        path_history[adj] = node
            cells_passed += counter
            max_moves += 1

            if not found_target:
                return None, max_moves, cells_passed
            
            whole_path = self.follow_path(path_history, current, targetPos)
            go = True

            for next_step in whole_path[1:]:
                if self.maze[next_step[0], next_step[1]] == -1:
                    self.seen.add(next_step)
                    go = False
                    break
                current = next_step
            if go:
                return whole_path, max_moves, cells_passed
            
        return whole_path, max_moves, cells_passed









    def estimate_distance(self, pointA, pointB):
        xDiff = abs(pointA[0]- pointB[0])
        yDiff = abs(pointA[1] - pointB[1])
        return xDiff + yDiff

    def priority(self, f_val, g_val, location):
        if self.mode == "high g-value" :
            return(self.number * f_val - g_val , location)
        else:
            return(self.number * f_val + g_val, location) 
    
    def find_adj(self, position):
        adjacent = []
        row = position[0]
        col = position[1]

        directions = [
            (-1,0),
            (1,0),
            (0,-1),
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
    
    def follow_path(self, history, beginning, destination):
        reversed_path = [destination]
        while reversed_path[-1] != beginning:
            reversed_path.append(history[reversed_path[-1]])
        return reversed_path[::-1]
    
def compare_tiebreaking(worlds):
    strategy = {
        'high_priority' : {'counts': [], 'wins': 0},
        'low_priority' : {'counts': [], 'wins' : 0}
    }

    for world in worlds:
        high_agent = Repeated(world, 'high g-value')
        path, _, cellnum = high_agent.find((0,0), (100,100))
        if path is not None:
            strategy['high_priority']['wins'] +=1
        strategy['high_priority']['counts'].append(cellnum)

        low_agent = Repeated(world, 'low g-value')
        pathlow, _, cellmax = low_agent.find((0,0), (100, 100))
        if pathlow is not None:
            strategy['low_priority']['wins'] +=1
        strategy['low_priority']['counts'].append(cellmax)


    average_high = sum(strategy['high_priority']['counts'])/50
    average_low = sum(strategy['low_priority']['counts'])/50

    print("High Priority Results:")
    print(f"Succesful paths: {strategy['high_priority']['wins']}")
    print(f"Average cells processed: {average_high:0f}")
    print("\nLow Priority Results:")
    print(f"Succesful paths:{strategy['low_priority']['wins']}")
    print(f"Average cells processed:{average_low:0f}")
    print(f"\nHigh strategy saves {average_low - average_high:.0f} checks({(average_low/average_high -1)*100:.1f}% better)")

if __name__ == "__main__":
    test = [Grid(101) for _ in range(50)]
    compare_tiebreaking(test)