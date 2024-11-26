import numpy as np
import itertools
import heapq

class Node:
    def __init__(self, x, y, theta, vl, vr, g, h, parent=None):
        self.x = x
        self.y = y
        self.theta = theta # Heading Angle
        self.vl = vl       # Left Side Velocity
        self.vr = vr       # Right Side Velocity
        self.g = g  # Cost from start to this node
        self.h = h  # Heuristic (estimated cost to goal)
        self.f = g + h  # Total cost (f = g + h)
        self.parent = parent  # To reconstruct the path
    
    # Priority queue comparison based on f value
    def __lt__(self, other):
        return self.f < other.f

class A_Star_Planner:
    def __init__(self, start_x, start_y, goal_x, goal_y, other_args):
        self.start_x = start_x
        self.start_y = start_y
        self.goal_x = goal_x
        self.goal_y = goal_y

        self.dt = other_args[0]
        self.rad = other_args[1]
        self.wheelbase = other_args[2]
        self.wgt_heur = other_args[3]
        self.velo_discretize = other_args[4]
        self.goal_radius = other_args[5]

    def heuristic(self, x, y):
        return pow(x-self.goal_x, 2)+pow(y-self.goal_y, 2)

    def a_star(self):
        open_list = []
        closed_list = set()
        open_set = {}

        possible_actions = list(range(-self.velo_discretize//2,self.velo_discretize//2+1))
        possible_actions = np.array(list(itertools.product(possible_actions, repeat=2)))

        start_node = Node(self.start_x, self.start_y, 0, 0, 0, 0, self.heuristic(self.start_x, self.start_y), parent=None)
        open_set[(self.start_x, self.start_y, 0, 0, 0)] = start_node  # Track in open set
        heapq.heappush(open_list, start_node)
        while open_list:
            curr = heapq.heappop(open_list)
            
            # print(curr.f, curr.h) # Logger of f and h
            # Return the path by back-tracking
            if curr.h <= self.goal_radius:
                path = []
                while curr:
                    path.append((curr.x, curr.y, curr.theta, curr.vl, curr.vr))
                    curr = curr.parent
                return path[::-1] 

            closed_list.add((curr.x, curr.y, curr.theta))

            for vl, vr in possible_actions:
                # Four Wheel Differential Steering Calcs
                velo = (self.rad/2)*(vl+vr)
                ang_velo = (self.rad/self.wheelbase)*(vr-vl)
                next_th = curr.theta+ang_velo*self.dt

                next_x = curr.x+(velo*-np.sin(curr.theta))*(ang_velo*self.dt)
                next_y = curr.y+(velo*np.cos(curr.theta))*(ang_velo*self.dt)

                # Skip if the neighbor is outside the grid or blocked
                if not (0 <= next_x < 5 and 0 <= next_y < 5):
                    continue
                # Skip if theta is not within rational bounds
                if next_th >= 2*np.pi or next_th <= 0:
                    continue
                # Skip if the neighbor is already in the closed list
                if (next_x, next_y, next_th) in closed_list:
                    continue

                # Calculate g, h, and f
                g_cost = curr.g + (np.abs(curr.vl-vl)+np.abs(curr.vr-vr))**2  # Decided to make cost function exponential higher for changing wheel velos
                h_cost = wgt_heur*self.heuristic(next_x, next_y)
                neighbor_node = Node(next_x, next_y, next_th, vl, vr, g_cost, h_cost, curr)

                # If the neighbor is not in the open list, add
                if (next_x, next_y, next_th, vl, vr) not in open_set:
                    heapq.heappush(open_list, neighbor_node)
                    open_set[(next_x, next_y, next_th, vl, vr)] = neighbor_node

        return None

if __name__ == '__main__':
    dt = 2                     # Time step
    rad = 0.35                 # Wheel Radius
    wheelbase = .6             # Wheelbase of ZOE2
    wgt_heur = 1.5             # A star weighting hyperparameter
    velo_discretize = 5        # How many wheel velos to consider (7 -> (-3,-2,-1,0,1,2,3))
    goal_radius = np.sqrt(.05) # Goal Radius for Termination

    start_x, start_y = 0, 0    # Start position
    goal_x, goal_y = 4, 4      # Goal position

    new_planner = A_Star_Planner(start_x, start_y, goal_x, goal_y, [dt, rad, wheelbase, wgt_heur, velo_discretize, goal_radius]) 
    plan = new_planner.a_star()  # Plan A-star
    print(plan)

    




        