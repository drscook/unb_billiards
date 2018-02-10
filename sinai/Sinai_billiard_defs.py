import numpy as np

abs_tol = 1e-5
rel_tol = 0.01

x_range = side - particle_radius
y_range = side - particle_radius
scatter_range = scatter_radius + particle_radius
hole_range = hole - particle_radius

normals = [np.array([1,0])
           ,np.array([-1,0])
           ,np.array([0,1])
           ,np.array([0,-1])
          ]

### Add checks and features
params = np.array([side, scatter_radius, particle_radius, x_range-scatter_range, y_range-scatter_range])
if np.any(params <= abs_tol):
    print(params)
    raise Exception('Parameters must all be positive')
if hole <= particle_radius:
    print('Hole is too small for particle to escape')
boundary_length = 2*2*x_range + 2*2*y_range + 2*np.pi*scatter_range
hole_length = 2*hole_range
bdy_hole_ratio = boundary_length / hole_length
rnd = np.random.RandomState(seed=seed)

def check_position(pos):
    ok = True
    if abs(pos[0]) > (1+rel_tol) * x_range:
        ok = False
    if abs(pos[1]) > (1+rel_tol) * y_range:
        ok = False
    if pos.dot(pos) < (1-rel_tol) * scatter_range**2:
        ok = False
    return ok

def error_message(step,pos,vel):
    raise Exception('Particle escaped at step {} \n pos = {} \n vel = {}'.format(step, pos, vel))

def solve_quadratic(a, b, c):
    small, big = np.inf, np.inf
    if abs(a) > abs_tol:
        d = b**2 - 4*a*c
        if d > abs_tol:
            e = np.sqrt(d)
            small = (-b - e) / (2*a)
            big   = (-b + e) / (2*a)
            if b > 0:
                small, big = big, small
    else:
        if abs(b) > abs_tol:
            t = -c / b
            small, big = t, t
    return small, big

### Main function
def run_trial(x=None, y=None, speed=1.0, theta=None, max_escapes=1):
    # Initialize position
    # Detect whether x or y was passed in.  If not, we randomly pick.
    # We randomly choose values for the unspecified cooridantes and the check if
    # it give a legal position.  If not, try again.
    rand_x = x is None
    rand_y = y is None
    
    max_attempts = 50
    for attempt in range(max_attempts):
        if rand_x == True:
            x = rnd.uniform(low=-(1-rel_tol)*x_range, high=(1-rel_tol)*x_range)
        if rand_y == True:
            y = rnd.uniform(low=-(1-rel_tol)*y_range, high=(1-rel_tol)*y_range)
        pos = np.array([x,y])
        if check_position(pos) == True:
            break
    if check_position(pos) == False:
        raise Exception('Could not initialize position')

    if theta is None:
        theta = rnd.uniform(low=0, high=2*np.pi)
    vel = speed * np.array([np.cos(theta), np.sin(theta)])

    t = 0.0
    escapes = 0
    t_hist = [t]
    pos_hist = [pos.copy()]
    vel_hist = [vel.copy()]
    escape_hist = [escapes]
    
    dts = np.zeros(num_walls)
    prior_collision = num_walls
    for step in range(max_steps):
            error_message(step,pos,vel)
        dts[:] = np.inf

        if abs(vel[0]) > abs_tol:
            dts[0] = (-x_range - pos[0]) / vel[0]
            dts[1] = ( x_range - pos[0]) / vel[0]

        if abs(vel[1]) > abs_tol:
            dts[2] = (-y_range - pos[1]) / vel[1]
            dts[3] = ( y_range - pos[1]) / vel[1]
        if prior_collision in range(0,4):
            dts[prior_collision] = np.inf

        a = vel.dot(vel)
        if a  > abs_tol:
            b = 2 * pos.dot(vel)
            c = pos.dot(pos) - scatter_range**2
            small, big = solve_quadratic(a, b, c)
            if prior_collision == 4:
                dts[4] = big
            else:
                dts[4] = small

        dts[dts<abs_tol] = np.inf
        if np.all(np.isinf(dts)):
            error_message(step,pos,vel)
        col_wall = np.argmin(dts)
        dt = dts[col_wall]
        t = t + dt
        pos = pos + vel*dt

        if col_wall in range(0,4):
            n = normals[col_wall]
        elif col_wall == 4:
            n = pos.copy()
            l = np.sqrt(n.dot(n))
            n = n / l

        vel = vel - 2 * (vel.dot(n)) * n
        prior_collision = col_wall
        
        if col_wall == 2:
            if abs(pos[0]) < hole_range:
                escapes += 1

        t_hist.append(t)
        pos_hist.append(pos.copy())
        vel_hist.append(vel.copy())
        escape_hist.append(escapes)
        
        if escapes >= max_escapes:
            break

    num_col = len(t_hist)-1
    results = [num_col, np.array(t_hist), np.array(pos_hist), np.array(vel_hist), np.array(escape_hist)]
    return results

import matplotlib.pyplot as plt
import ipywidgets as widgets
def draw_hist(pos, t=None):
    dpos = np.diff(pos,axis=0)
    max_steps = dpos.shape[0]
    if t is None:
        t = np.arange(max_steps)
    
    h = hole_range
    x = x_range
    y = y_range
    side_bdy = np.array([(h,-y), (x,-y), (x,y), (-x,y), (-x,-y), (-h,-y)])
    thetas = np.linspace(0, 2*np.pi, 100)
    scatter_bdy = scatter_range * np.array([np.cos(thetas), np.sin(thetas)]).T

    def draw(steps=1):
        print('steps = {}, time = {:.2f}'.format(steps, t[steps]))
        fig, ax = plt.subplots(figsize=[5,5])
        ax.plot(side_bdy[:,0], side_bdy[:,1])
        ax.fill(scatter_bdy[:,0], scatter_bdy[:,1])
        ax.quiver(pos[:steps,0], pos[:steps,1], dpos[:steps,0], dpos[:steps,1], angles='xy', scale_units='xy', scale=1)
        ax.set_aspect('equal')
        plt.show()

    
    step_slider = widgets.IntSlider(min=1, max=max_steps, value=1)
    step_text = widgets.BoundedIntText(min=1, max=max_steps, value=1)
    #link = widgets.jslink((step_slider, 'value'), (step_text, 'value'))
    im = widgets.interactive(draw, steps=step_slider)
    display(widgets.VBox([step_slider, step_text, im]))