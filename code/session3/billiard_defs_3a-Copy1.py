#######################################################################################################
###  Contents ###

# Dynamical Functions and Class - The main simulation code
# Supporting Function - All commands used by simulation not from a standard python package are defined here.
#######################################################################################################


#######################################################################################################
###  Dynamical Functions and Classes ###
#######################################################################################################

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import ipywidgets as widgets
import itertools as it

abs_tol = 1e-4
rnd = np.random.RandomState(seed)

class Wall():
    # default values that apply to all geometries
    Wall_defaults = {'dim':2, 'gap_pad':0.0, 'wp_collision_law':'wp_specular'}

    def wp_specular_law(self, part, p):
        nu = self.normal(part.pos[p])
        part.vel[p] -= 2 * part.vel[p].dot(nu) * nu
        
    # particle wraps around to the opposite side of the billiard cell by flipping sign on dim d
    def wp_wrap_law(self, part, p):
        d = self.wrap_dim   # which dim will have sign flip
        s = np.sign(part.pos[p, d]).astype(int)  # is it at + or -
        part.cell_offset[p, d] += s   # tracks cell position for each particle
        part.pos[p, d] *= -1   # flips sign of dime d
        part.wp_mask[self.idx, p] = False
        part.wp_mask[self.wrap_wall, p] = True

    def resolve_collision(self, part, p):
        if self.wp_collision_law == 'wp_specular':
            self.wp_specular_law(part, p)
        elif self.wp_collision_law == 'wp_wrap':
            self.wp_wrap_law(part, p)
        
class FlatWall(Wall):
    def __init__(self, **kwargs):
        # convient way to combine Wall defaults, FlatWall defaults, and user specified attributes
        params = self.Wall_defaults.copy()
        params.update(kwargs)
    
        params['normal_static'] = make_unit(params.pop('normal'))  # renames normal -> normal-static and makes unit vector
        for key, val in params.items():  # saves params into the wall objeect
            if isinstance(val, list):
                val = np.asarray(val)  # converts lists to arrays
            setattr(self, key, val)
    
        self.wp_gap_min = self.gap_pad
        self.get_mesh()
    
    def get_mesh(self):
        self.mesh = flat_mesh(self.tangents) + self.base_point
        
    def normal(self, pos):  # normal does not depend on collision point
        return self.normal_static

    def get_wp_col_time(self, mask=None):
        t = np.full(part.num, np.inf)  # default in np.inf
        nu = self.normal_static
        dx = part.pos - self.base_point
        c = self.wp_gap_min - dx.dot(nu)
        b = part.vel.dot(nu)
        idx = np.abs(b) >= abs_tol  # prevents divide by zero
        t[idx] = c[idx] / b[idx]
        if mask is not None:
            t[mask] = np.inf
        t[t<0] = np.inf  #np.inf for negative times
        return t
    
class SphereWall(Wall):
    def __init__(self, **kwargs):
        # convient way to combine Wall defaults, FlatWall defaults, and user specified attributes
        params = self.Wall_defaults.copy()
        params.update(kwargs)
    
        for key, val in params.items():  # saves params into the wall objeect
            if isinstance(val, list):
                val = np.asarray(val)  # converts lists to arrays
            setattr(self, key, val)
    
        self.wp_gap_min = self.radius + self.gap_pad
        self.get_mesh()

    def get_mesh(self):
        self.mesh = sphere_mesh(self.dim, self.radius) + self.base_point

    def normal(self, pos): # normal depends on collision point
        dx = pos - self.base_point
        return make_unit(dx)  # see below for make_unit

    def get_wp_col_time(self, mask=None):
        dx = part.pos - self.base_point
        dv = part.vel
        a =   (dv*dv).sum(axis=-1)
        b = 2*(dv*dx).sum(axis=-1)
        c =   (dx*dx).sum(axis=-1) - self.wp_gap_min**2
        t_small, t_big = solve_quadratic(a, b, c, mask)
        t = np.fmin(t_small, t_big)
        return t
        
class Particles():
    def __init__(self, **kwargs):
        params = {'max_steps':50, 'dim':2, 'num':1, 'radius':[1.0], 'mass':[1.0], 'pp_collision_law':'pp_specular', 'gamma':'uniform'}
        params.update(kwargs)
        
        if(params['gamma'] == 'uniform'):
            params['gamma'] = np.sqrt(2/(2+params['dim']))
        elif(params['gamma'] == 'shell'):
            params['gamma'] = np.sqrt(2/params['dim'])
        elif(params['gamma'] == 'point'):
            params['gamma'] = 0

        # Each parameter list must be num_particles long.  If not, this will extend by filling with the last entry
        constants = ['radius', 'mass']
        for const in constants:
            c = listify(params[const])  #listify defined at bottom of this file
            for p in range(len(c), params['num']):
                c.append(c[-1])
            params[const] = np.asarray(c).astype(float)
        
        for key, val in params.items():
            if isinstance(val, list):
                val = np.asarray(val)  # converts lists to arrays
            setattr(self, key, val)
        self.mom_inert = self.mass * (self.gamma * self.radius)**2
        self.get_mesh()
        
        self.wp_dt = np.zeros([len(wall), self.num], dtype='float')
        self.wp_mask = self.wp_dt.copy().astype(bool)
        
        self.t = 0.0
        self.cell_offset = np.zeros([self.num, self.dim], dtype=int)  # tracks which cell the particle is in
        self.col = {}
        self.t_hist = []
        self.col_hist = []
        self.pos_hist = []
        self.vel_hist = []
        
        # Color particles (helpful for the future when we have many particles)
        cm = plt.cm.gist_rainbow
        idx = np.linspace(0, cm.N-1 , self.num).round().astype(int)
        self.clr = [cm(i) for i in idx]
        
    def get_mesh(self):
        self.mesh = []
        for p in range(self.num):
            R = self.radius[p]
            M = sphere_mesh(dim=self.dim, radius=R)
            self.mesh.append(M)
        self.mesh = np.asarray(self.mesh)

    def record_state(self):
        self.t_hist.append(self.t)
        self.pos_hist.append(self.pos_to_global())
        self.vel_hist.append(self.vel.copy())
        #self.cell_offset_hist.append(self.cell_offset.copy())

    def get_KE(self):
        KE = self.mass * np.sum(self.vel**2, axis=-1)
        return np.sum(KE) / 2
        
    def pos_to_global(self):
        # self.pos is local to current cell.  This return the global position by adding cell offset.
        return self.pos + self.cell_offset * self.cell_size * 2

def next_state(wall, part):
    for (i,w) in enumerate(wall):
        part.wp_dt[i] = w.get_wp_col_time(part.wp_mask[i])
    part.dt = np.min(part.wp_dt)

    part.t += part.dt
    part.pos += part.vel * part.dt

    part.wp_mask = (part.wp_dt - part.dt) < 1e-8
    w, p = np.nonzero(part.wp_mask)
    
    # later, we will need to protect against "complex collisions" where a particle make 
    # multiple simultaneous collisions.  For simplicity of the code, we'll ignore this
    # for now, knowing that we may get unexpected behavior if this occurs.
    w, p = w[0], p[0]
    part.col = {'w':w, 'p':p}
    wall[w].resolve_collision(part, p)
    #if np.abs(part.get_KE() - part.KE_init) > abs_tol:
    #    raise Exception('Energy was not conserved')
    part.record_state()

def clean_up(part):
    part.t_hist = np.asarray(part.t_hist)
    #part.cell_offset_hist = np.asarray(part.cell_offset_hist)
    part.pos_hist = np.asarray(part.pos_hist)
    part.vel_hist = np.asarray(part.vel_hist)
    print('Done!! Steps = {}, Time = {:4f}'.format(len(part.t_hist)-1, part.t_hist[-1]))
    

#######################################################################################################    
#######################################################################################################
###  Support Functions ###
#######################################################################################################
#######################################################################################################


#######################################################################################################
###  Graphics Functions ###
#######################################################################################################

def draw_background(pos):
    M = [[(np.min(pos[:,:,d])/(2*part.cell_size[d])).round()
         ,(np.max(pos[:,:,d])/(2*part.cell_size[d])).round()
         ] for d in range(part.dim)]
    cell_range = [2 * part.cell_size[d] * np.arange(M[d][0], M[d][1]+1) for d in range(part.dim)]
    translates = it.product(*cell_range)
    ax = plt.gca()
    if part.dim == 2:
        for trans in translates:
            for w in wall:
                ax.plot(*(w.mesh + trans).T, color='black')

def draw_state(num_frames=-1):
    max_frames = part.pos_hist.shape[0]-1
    if (num_frames == -1) or (num_frames > max_frames):
        num_frames = max_frames
        
    pos = part.pos_hist[:num_frames+1]
    fig, ax = plt.subplots()
    draw_background(pos)
    for p in range(part.num):
        ax.plot(pos[:,p,0], pos[:,p,1], color=part.clr[p])
        ax.plot(*(part.mesh[p]+pos[-1,p]).T, color=part.clr[p])
    ax.set_aspect('equal')
    plt.show()

def interactive_plot(num_frames=-1):
    max_frames = part.pos_hist.shape[0]-1
    if (num_frames == -1) or (num_frames > max_frames):
        num_frames = max_frames

    pos = part.pos_hist[:num_frames+1]
    dpos = np.diff(pos, axis=0)  # position change    
    def update(s):
        fig, ax = plt.subplots(figsize=[8,8]);
        ax.set_aspect('equal')
        plt.title('s={} t={:.2f}'.format(s,part.t_hist[s]))
        draw_background(pos[:s+1])
        for p in range(part.num):
            ax.plot(pos[:s+1,p,0], pos[:s+1,p,1], color=part.clr[p])
            ax.plot(*(part.mesh[p] + pos[s,p]).T, color=part.clr[p])
        plt.show()

    l = widgets.Layout(width='150px')
    step_text = widgets.BoundedIntText(min=0, max=num_frames, value=0, layout=l)
    step_slider = widgets.IntSlider(min=0, max=num_frames, value=0, readout=False, continuous_update=False, layout=l)
    widgets.jslink((step_text, 'value'), (step_slider, 'value'))

    play_button = widgets.Play(min=0, max=num_frames, interval=500, layout=l)
    widgets.jslink((step_text, 'value'), (play_button, 'value'))

    img = widgets.interactive_output(update, {'s':step_text})
    display(widgets.HBox([widgets.VBox([step_text, step_slider, play_button]), img]))

def flat_mesh(tangents):
    pts = 100
    N, D = tangents.shape
    grid = [np.linspace(-1, 1, pts) for n in range(N)]
    grid = np.meshgrid(*grid)
    grid = np.asarray(grid)
    mesh = grid.T.dot(tangents)
    return mesh

def sphere_mesh(dim, radius):
    pts = 100
    grid = [np.linspace(0, np.pi, pts) for d in range(dim-1)]
    grid[-1] *= 2
    grid = np.meshgrid(*grid)                           
    mesh = []
    for d in range(dim):
        w = radius * np.ones_like(grid[0])
        for j in range(d):
            w *= np.sin(grid[j])
        if d < dim-1:
            w *= np.cos(grid[d])
        mesh.append(w)
    return np.asarray(mesh).T    
 
    
#######################################################################################################
### No-Slip Collision Functions ###
#######################################################################################################
def spin_matrix_from_vector(v):
    # Converts spin vector to spin matrix
    # https://en.wikipedia.org/wiki/Rotation_matrix#Exponential_map
                     
    l = len(v)
    # l = d(d-1) -> d**2 - d - 2l = 0
    d = (1 + np.sqrt(1 + 8*l)) / 2
    if d % 1 != 0:
        raise Exception('vector {} of length {} converts to dim = {:.2f}.  Not integer.'.format(v,l,d))
    d = int(d)
    M = np.zeros([d,d])
    idx = np.triu_indices_from(M,1)
    s = (-1)**(np.arange(len(v))+1)
    w = v * s
    w = w[::-1]
    M[idx] = w
    M = make_symmetric(M, skew=True)
    return M

def spin_vector_from_matrix(M):
    idx = np.triu_indices_from(M,1)
    w = M[idx]
    s = (-1)**(np.arange(len(w))+1)
    w = w[::-1]    
    v = w * s
    return v
   
def wedge(a,b):
    return np.outer(b,a)-np.outer(a,b)

def Pi_nu(v, nu):
    return v.dot(nu) * nu

def Pi(v, nu):
    w = Pi_nu(v ,nu)
    return v - w

def Lambda_nu(U, nu):
    return wedge(nu, U.dot(nu))

def E_nu(v, nu):
    return wedge(nu, v)

def Gamma_nu(U, nu):
    return U.dot(nu)
    
#######################################################################################################
###  Random Functions ###
#######################################################################################################

def random_uniform_sphere(num=1, dim=2, radius=1.0):
    pos = rnd.normal(size=[num, dim])
    pos = make_unit(pos, axis=1)
    return abs(radius) * pos


def random_uniform_ball(num=1, dim=2, radius=1.0):
    pos = random_uniform_sphere(num, dim, radius)
    r = rnd.uniform(size=[num, 1])
    return r**(1/dim) * pos
    
#######################################################################################################
###  Basic Math Functions ###
#######################################################################################################
def make_symmetric(A, skew=False):
    """
    Returns symmetric or skew-symmatric matrix by copying upper triangular onto lower.
    """
    A = np.asarray(A)
    U = np.triu(A,1)
    if skew == True:
        return U - U.T
    else:
        return np.triu(A,0) + U.T    
    
def make_unit(A, axis=-1):
    # Normalizes along given axis.  This means that Thus, np.sum(A**2, axis) gives a matrix of all 1's.
    #In other words, if you pick values for all indices except axis and sum the squares, you get 1.  
    A = np.asarray(A, dtype=float)
    M = np.linalg.norm(A, axis=axis, keepdims=True)
    return A / M

def solve_quadratic(a, b, c, mask=None):
    small = np.full_like(a, np.inf)  
    d = b**2 - 4*a*c  #discriminant
    lin = (abs(a) < abs_tol) & (abs(b) >= abs_tol)  #linear 
    quad = (abs(a) >= abs_tol) & (d >= abs_tol)  #quadratic
    
    small[lin] = -1 * c[lin] / b[lin]
    big = small.copy()
    
    d[quad] = np.sqrt(d[quad])
    small[quad] = (-b[quad] - d[quad]) / (2*a[quad])
    big[quad] = (-b[quad] + d[quad]) / (2*a[quad])
    swap = (b >= abs_tol)  # We want the solutions ordered (small, big), so we swap where needed
    small[swap], big[swap] = big[swap], small[swap]
    if mask is not None:
        small[mask] = np.inf
        big[mask] = np.inf
    small[small<0] = np.inf
    big[big<0] = np.inf
    return small, big



def listify(X):
    """
    Convert X to list if it's not already
    """
    if (X is None) or (X is np.nan):
        return []
    elif isinstance(X,str):
        return [X]
    else:
        try:
            return list(X)
        except:
            return [X]
