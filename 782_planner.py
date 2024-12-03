import numpy as np
import heapq
import math, time
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

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
    def __init__(self, start_x, start_y,goal_x,goal_y,other_args):
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
        self.map_bounds = other_args[7]

        self.path = None

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
        poss_R = np.arange(-10,10,.5)
        poss_vel = np.arange(.25,2,.25)

        #### Initiate Search
        start_node = Node(self.start_x, self.start_y, 0, 0, 0, 0, self.heuristic(self.start_x, self.start_y),0,0, parent=None)
        open_set[(self.start_x, self.start_y, 0, 0, 0)] = start_node  # Track in open set
        heapq.heappush(open_list, start_node)
        while open_list:
            curr = heapq.heappop(open_list)
            
            # Return the path by back-tracking
            # print(curr.x, curr.y, curr.h)
            if curr.h <= self.goal_radius:
                path = []
                while curr:
                    path.append((curr.x, curr.y, curr.theta, curr.arc_radius, curr.transl_velocity))
                    curr = curr.parent
                self.path = path[::-1]
                return path[::-1] 

            closed_list.add((curr.x, curr.y, curr.theta))

            #### Sweep through Possible Arcs + Velocities for Expansion
            for arc_rad in poss_R:
                if arc_rad == 0:
                    continue
                for v_comm in poss_vel:
                    comm_velos = self.velocity_control(arc_rad, v_comm, curr.theta)
                    vl = comm_velos[0][0]
                    vr = comm_velos[1][0]

                    # Four Wheel Differential Steering Calcs [NEW]
                    ang_velo = v_comm/arc_rad
                    next_th = curr.theta-ang_velo*self.dt

                    # Clip Theta [NEW]
                    if next_th >= 2*np.pi:
                        next_th = math.remainder(next_th, 2*np.pi)
                    if next_th < 0:
                        next_th = 2*np.pi - math.remainder(next_th, 2*np.pi)

                    # Calculate Next Point [NEW]
                    if arc_rad > 0:
                        next_x = curr.x + arc_rad*(np.cos(-next_th)-np.cos(-curr.theta))
                        next_y = curr.y + arc_rad*(np.sin(-next_th)-np.sin(-curr.theta))
                    else:
                        next_x = curr.x + (-arc_rad)*(np.cos(np.pi-next_th)-np.cos(np.pi-curr.theta))
                        next_y = curr.y + (-arc_rad)*(np.sin(np.pi-next_th)-np.sin(np.pi-curr.theta))

                    # Skip if the neighbor is outside the grid or blocked (note that this defined boundaries) [NEW]
                    if not (self.map_bounds[0] <= next_x < self.map_bounds[2] and self.map_bounds[1] <= next_y < self.map_bounds[3]):
                        continue

                    # Check if in closed list
                    if (next_x, next_y, next_th) in closed_list:
                        continue

                    # Check if any path points are out of bounds [NEW]
                    in_bounds = True
                    for angv_step in np.linspace(0,ang_velo, 5):
                        next_th = curr.theta-angv_step*self.dt
                        if arc_rad > 0:
                            next_x = curr.x + arc_rad*(np.cos(-next_th)-np.cos(-curr.theta))
                            next_y = curr.y + arc_rad*(np.sin(-next_th)-np.sin(-curr.theta))
                        else:
                            next_x = curr.x + (-arc_rad)*(np.cos(np.pi-next_th)-np.cos(np.pi-curr.theta))
                            next_y = curr.y + (-arc_rad)*(np.sin(np.pi-next_th)-np.sin(np.pi-curr.theta))
                        if not (self.map_bounds[0] <= next_x < self.map_bounds[2] and self.map_bounds[1] <= next_y < self.map_bounds[3]):
                            in_bounds = False
                            break
                    if not in_bounds:
                        continue
                            
                    # Calculate g, h, and f
                    # g_cost = curr.g + 10*(np.abs(curr.transl_velocity-v_comm)+np.abs(curr.arc_radius-arc_rad))  # Decided to make cost function exponential higher for changing wheel velos
                    g_cost = curr.g + 5*(np.abs(curr.vl-vl)+np.abs(curr.vr-vr))
                    h_cost = wgt_heur*self.heuristic(next_x, next_y)
                    neighbor_node = Node(next_x, next_y, next_th, vl, vr, g_cost, h_cost,v_comm, arc_rad, curr)

                    # If the neighbor is not in the open list, add
                    if (next_x, next_y, next_th, vl, vr) not in open_set:
                        heapq.heappush(open_list, neighbor_node)
                        open_set[(next_x, next_y, next_th, vl, vr)] = neighbor_node

        return None
    
    def plot_path(self):
        path_points = []
        for i in range(len((self.path))):
            if i == 0:
                continue
            
            init_x = self.path[i-1][0]
            init_y = self.path[i-1][1]
            init_th = self.path[i-1][2]
            arc_rad = self.path[i][3]
            v_comm = self.path[i][4]
            ang_velo = v_comm/arc_rad
            for angv_step in np.linspace(0,ang_velo, 100):
                next_th = init_th-angv_step*self.dt
                steer_angle = np.pi/2-np.arcsin(np.abs(arc_rad)/np.sqrt(arc_rad**2+(self.wheelbase/2)**2))
                if arc_rad > 0:
                    next_x = init_x + arc_rad*(np.cos(-next_th)-np.cos(-init_th))
                    next_y = init_y + arc_rad*(np.sin(-next_th)-np.sin(-init_th))
                    steer_angle /= -1
                else:
                    next_x = init_x + (-arc_rad)*(np.cos(np.pi-next_th)-np.cos(np.pi-init_th))
                    next_y = init_y + (-arc_rad)*(np.sin(np.pi-next_th)-np.sin(np.pi-init_th))
                path_points.append([next_x, next_y, next_th, steer_angle])
        path_points = np.array(path_points)

        plt.figure(0)
        plt.xlim((self.map_bounds[0],self.map_bounds[2]))
        plt.ylim((self.map_bounds[1],self.map_bounds[3]))
        plt.title("Scatter Plot of Path")
        plt.scatter(path_points[:,0], path_points[:,1], label="Path Points")
        plt.legend()
        plt.grid(True)
        plt.show() 

        fig, ax = plt.subplots()
        ax.set_xlim((self.map_bounds[0],self.map_bounds[2]))
        ax.set_ylim((self.map_bounds[1],self.map_bounds[3]))
        ax.grid(True)
        ax.set_aspect('equal') 

        car, = ax.plot([], [], 'k-', lw=2, label="Zoe2")  # Initial empty plot for the car
        front_axle, = ax.plot([], [], 'ko-', lw=2)  
        rear_axle, = ax.plot([], [], 'ko-', lw=2) 
        center_point, = ax.plot([], [], 'ro', markersize=4, label="Zoe2 Centroid")
        ax.plot([self.start_x], [self.start_y], 'mo', markersize=4, label="Initial Position")
        ax.plot([self.goal_x], [self.goal_y], 'co', markersize=10, label="Goal Radius") 
        plt.title("Zoe2 Rover - A-Star Plan")
        plt.xlabel("X (in m)")
        plt.ylabel("Y (in m)")

        def init():
            car.set_data([], [])
            front_axle.set_data([], [])
            rear_axle.set_data([], [])
            center_point.set_data([], [])
            ax.legend(loc='upper left')  # Customize location as needed
            return car, front_axle, rear_axle, center_point

        # Set up the animation
        def update(frame):
            # Get the current position and heading
            car_length = self.wheelbase  # length of the "car" or rod
            axle_length = self.width  # Length of the axles
            
            cx = path_points[:,0][frame]
            cy = path_points[:,1][frame]
            angle = path_points[:,2][frame]
            steer_angle = path_points[:,3][frame]
            
            # Calculate the car's orientation as a line (from the center to the front)
            dx = car_length * np.cos(np.pi/2-angle)
            dy = car_length * np.sin(np.pi/2-angle)
            
            # Update the car's position and orientation (as a line with an arrowhead)
            car.set_data([cx - dx/2, cx + dx/2], [cy - dy/2, cy + dy/2])

            rear_axle_x1 = cx - dx/2 - axle_length/2 * np.sin(np.pi/2-angle+steer_angle)
            rear_axle_y1 = cy - dy/2 + axle_length/2 * np.cos(np.pi/2-angle+steer_angle)
            rear_axle_x2 = cx - dx/2 + axle_length/2 * np.sin(np.pi/2-angle+steer_angle)
            rear_axle_y2 = cy - dy/2  - axle_length/2 * np.cos(np.pi/2-angle+steer_angle)
            
            # Front axle: located at the front end of the car (same direction as the heading)
            front_axle_x1 = cx + dx/2 - axle_length/2 * np.sin(np.pi/2-angle-steer_angle)
            front_axle_y1 = cy + dy/2 + axle_length/2 * np.cos(np.pi/2-angle-steer_angle)
            front_axle_x2 = cx + dx/2 + axle_length/2 * np.sin(np.pi/2-angle-steer_angle)
            front_axle_y2 = cy + dy/2 - axle_length/2 * np.cos(np.pi/2-angle-steer_angle)
            
            # Update the axles' positions
            front_axle.set_data([front_axle_x1, front_axle_x2], [front_axle_y1, front_axle_y2])
            rear_axle.set_data([rear_axle_x1, rear_axle_x2], [rear_axle_y1, rear_axle_y2])

            center_point.set_data(cx, cy)
            
            return car, front_axle, rear_axle, center_point
        
        ani = FuncAnimation(fig, update, frames=len(path_points[:,0]), init_func=init, blit=True, interval=50)
        # ani.save('car_animation.gif', writer='pillow', fps=20)
        plt.show()

# Show the plot
plt.show()

if __name__ == '__main__':
    dt = 2                      # Time step
    rad = 0.325                 # Wheel Radius
    width = 1.64                # Axle Width
    wheelbase = 1.91            # Wheelbase of ZOE2
    wgt_heur = 5               # A star weighting hyperparameter
    goal_radius = np.sqrt(.005) # Goal Radius for Termination
    gain = 0                    # Controller Gain

    start_x, start_y = 1, 0    # Start position
    goal_x, goal_y = 8, 8      # Goal position
    map_bounds = [-2,-2,10,10]      # min_x, min_y, max_x, max_y

    new_planner = A_Star_Planner(start_x, start_y,goal_x, goal_y, [dt, rad, width, wheelbase, wgt_heur, goal_radius, gain, map_bounds]) 
    start = time.time()
    plan = new_planner.a_star()  # Plan A-star
    print('Time to Complete', time.time()-start, 'seconds')
    print(plan)
    new_planner.plot_path()

    




        