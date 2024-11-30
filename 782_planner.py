import numpy as np
import itertools
import heapq

class Node:
    def __init__(self, x, y, theta, vl, vr, g, h, v_comm, arc_rad, parent=None):
        self.x = x
        self.y = y
        self.theta = theta         # Heading Angle
        self.vl = vl               # Left Side Velocity
        self.vr = vr               # Right Side Velocity
        self.arc_radius = arc_rad  ## Current arc of motion
        self.transl_velocity = v_comm  # Current Translational Velocity
        self.g = g                 # Cost from start to this node
        self.h = h                 # Heuristic (estimated cost to goal)
        self.f = g + h             # Total cost (f = g + h)
        self.parent = parent       # To reconstruct the path
    
    # Priority queue comparison based on f value
    def __lt__(self, other):
        return self.f < other.f

class A_Star_Planner:
    def __init__(self, start_x, start_y, goal_x, goal_y, other_args):
        '''
        initialization of the map and hyperparameters
        start_x, start_y: initial position
        goal_x, goal_y: goal position
        dt: time jump between driving steps
        rad: wheel radius
        width: axle width
        wheelbase: length from front axle to the rear
        wgt_heur: weighting for weighted-A* search
        goal_radius: valid radius for termination at goal
        th_gain: for velocity controller, make non-zero to replicate real robot
        '''
        self.start_x = start_x
        self.start_y = start_y
        self.goal_x = goal_x
        self.goal_y = goal_y

        self.dt = other_args[0]
        self.rad = other_args[1]
        self.width = other_args[2]
        self.wheelbase = other_args[3]
        self.wgt_heur = other_args[4]
        self.goal_radius = other_args[5]
        self.th_gain = other_args[6]

    def heuristic(self, x, y):
        '''
        Squared Euclidean Distance to Goal
        '''
        return pow(x-self.goal_x, 2)+pow(y-self.goal_y, 2)

    def velocity_control(self, rad_comm, velo_comm, th_prev):
        '''
        From "Control of a Passively Steered Rover using 3-D Kinematics" 
        - Control of left and right velocities given all steer axes are coplanar
        - Inputs:
            rad_comm: Commanded Arc Radius of Motion
            velo_comm: Commanded Scalar Velocity of Motion
            th_prev: The previous heading angle
        - Outputs:
            velos: (np.array) contains left- and right-side velocities
        '''
        th_comm = np.arctan(self.wheelbase/2/rad_comm)
        ang_velo = velo_comm/rad_comm
        phys_term = np.array([[1/np.cos(th_comm), -self.width/2],[1/np.cos(th_comm), self.width/2]])@np.array([[velo_comm, ang_velo]]).T
        gain_term = self.th_gain*np.array([[-(th_comm-th_prev), th_comm-th_prev]]).T
        velos = phys_term+gain_term
        return velos

    def a_star(self):
        #### A-Star Search over Implicitly Defined Graph (successors based on arc radius + velocity)
        open_list = []
        closed_list = set()
        open_set = {}

        #### Possible Rotations and Velocities (please change if not functional)
        poss_R = np.arange(-10,10,3)
        poss_vel = np.arange(1,10,.5)

        #### Initiate Search
        start_node = Node(self.start_x, self.start_y, 0, 0, 0, 0, self.heuristic(self.start_x, self.start_y),0,0, parent=None)
        open_set[(self.start_x, self.start_y, 0, 0, 0)] = start_node  # Track in open set
        heapq.heappush(open_list, start_node)
        while open_list:
            curr = heapq.heappop(open_list)
            
            # Return the path by back-tracking
            print(curr.x, curr.y, curr.h)
            if curr.h <= self.goal_radius:
                path = []
                while curr:
                    path.append((curr.x, curr.y, curr.theta, curr.arc_radius, curr.transl_velocity))
                    curr = curr.parent
                return path[::-1] 

            closed_list.add((curr.x, curr.y, curr.theta))

            #### Sweep through Possible Arcs + Velocities for Expansion
            for arc_rad in poss_R:
                for v_comm in poss_vel:
                    comm_velos = self.velocity_control(arc_rad, v_comm, curr.theta)
                    vl = comm_velos[0][0]
                    vr = comm_velos[1][0]

                    # Four Wheel Differential Steering Calcs
                    velo = (self.rad/2)*(vl+vr)
                    ang_velo = (self.rad/self.wheelbase)*(vr-vl)
                    next_th = curr.theta+ang_velo*self.dt

                    next_x = curr.x+(velo*-np.sin(curr.theta))*(ang_velo*self.dt)
                    next_y = curr.y+(velo*np.cos(curr.theta))*(ang_velo*self.dt)

                    # Skip if the neighbor is outside the grid or blocked (note that this defined boundaries)
                    if not (0 <= next_x < 5 and 0 <= next_y < 5):
                        continue
                    # Skip if theta is not within rational bounds
                    if next_th >= 2*np.pi or next_th <= 0:
                        continue
                    # Skip if the neighbor is already in the closed list
                    if (next_x, next_y, next_th) in closed_list:
                        continue

                    # Calculate g, h, and f
                    g_cost = curr.g + 5*(np.abs(curr.vl-vl)+np.abs(curr.vr-vr))  # Decided to make cost function exponential higher for changing wheel velos
                    h_cost = wgt_heur*self.heuristic(next_x, next_y)
                    neighbor_node = Node(next_x, next_y, next_th, vl, vr, g_cost, h_cost,v_comm, arc_rad, curr)

                    # If the neighbor is not in the open list, add
                    if (next_x, next_y, next_th, vl, vr) not in open_set:
                        heapq.heappush(open_list, neighbor_node)
                        open_set[(next_x, next_y, next_th, vl, vr)] = neighbor_node

        return None

if __name__ == '__main__':
    dt = 2                      # Time step
    rad = 0.325                 # Wheel Radius
    width = 1.64                # Axle Width
    wheelbase = 1.91            # Wheelbase of ZOE2
    wgt_heur = 5                # A star weighting hyperparameter
    goal_radius = np.sqrt(.005) # Goal Radius for Termination
    gain = 0                    # Controller Gain

    start_x, start_y = 0, 0    # Start position
    goal_x, goal_y = 4, 4      # Goal position

    new_planner = A_Star_Planner(start_x, start_y, goal_x, goal_y, [dt, rad, width, wheelbase, wgt_heur, goal_radius, gain]) 
    plan = new_planner.a_star()  # Plan A-star
    print(plan)

    




        