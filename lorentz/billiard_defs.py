#######################################################################################################
###  Contents ###

# Dynamical Functions and Class - The main simulation code
# Supporting Function - All commands used by simulation not from a standard python package are defined here.
#######################################################################################################



#######################################################################################################
###  Dynamical Functions and Class ###
#######################################################################################################

import numpy as np
import itertools as it

abs_tol = 1e-5
rel_tol = 1 + 1e-5
rnd = np.random.RandomState(seed=seed)

class Wall():
    Wall_defaults = {'dim':2, 'gap_pad':0.0, 'collision_law':'specular', 'wrap_wall':None, 'wrap_dim':None}
    
    def pw_specular_law(self, part, p):
        nu = self.normal(part.pos[p])
        part.vel[p] -= 2 * proj(part.vel[p], nu)

    def pw_wrap_law(self, part, p):
        #part.record_state()
        d = self.wrap_dim
        s = np.sign(part.pos[p, d]).astype(int)
        part.cell[p, d] += s
        part.pos[p, d] *= -1
        part.pw_mask[self.idx, p] = False
        part.pw_mask[self.wrap_wall, p] = True

    def resolve_collision(self, part, p):
        if self.collision_law == 'specular':
            self.pw_specular_law(part, p)
        elif self.collision_law == 'wrap':
            self.pw_wrap_law(part, p)
        else:
            raise Exception('Unknown pw collision law {}'.format(self.collision_law))
    
class FlatWall(Wall):
    def __init__(self, **kwargs):
        params = self.Wall_defaults.copy()
        params.update(kwargs)
        dim = params['dim']
        FlatWall_defaults = {'base_point': np.zeros(dim), 'normal':np.ones(dim), 'tangents':None}
        params.update(FlatWall_defaults)
        params.update(kwargs)
        
        params['normal_static'] = params.pop('normal')
        for key, val in params.items():
            setattr(self, key, val)
        self.base_point = np.asarray(self.base_point, dtype=float)

        if self.tangents is None:
            O = self.normal_static
            self.onb = make_onb(O)
            self.tangents = self.onb[1:].copy()
        else:
            O = np.vstack([self.normal_static, self.tangents])
            self.tangents = O[1:].copy()
            self.onb = make_onb(O)
        self.normal_static = self.onb[0].copy()

        self.pw_gap_min = self.gap_pad
        self.make_mesh()
        
    def normal(self, pos):
        return self.normal_static
        
    def get_pw_gap(self):
        dx = part.pos - self.base_point
        self.pw_gap = dx.dot(self.normal_static) - self.pw_gap_min
        return self.pw_gap
        
    def get_pw_col_time(self):
        t = np.full(part.num, np.inf)
        nu = self.normal_static
        dx = part.pos - self.base_point
        b = -1 * part.vel.dot(nu)
        c = dx.dot(nu) - self.pw_gap_min
        idx = b > abs_tol
        t[idx] = c[idx] / b[idx]
        mask = part.pw_mask[self.idx]
        t[mask] = np.inf
        t[t<0] = np.inf
        return t

    def make_mesh(self):
        mesh = flat_mesh(tangents=self.tangents)
        self.mesh = mesh + self.base_point
    
class SphereWall(Wall):
    def __init__(self, **kwargs):
        params = self.Wall_defaults.copy()
        params.update(kwargs)
        dim = params['dim']
        SphereWall_defaults = {'radius':1.0, 'base_point': np.zeros(dim)}
        params.update(SphereWall_defaults)
        params.update(kwargs)

        for key, val in params.items():
            setattr(self, key, val)

        self.base_point = np.asarray(self.base_point, dtype=float)
        self.pw_gap_min = self.radius + self.gap_pad
        self.make_mesh()
       
    def normal(self, pos):
        dx = pos - self.base_point
        return make_unit(dx)

    def get_pw_gap(self):
        dx = part.pos - self.base_point
        self.pw_gap = np.linalg.norm(dx, axis=-1) - self.pw_gap_min
        return self.pw_gap 

    def get_pw_col_time(self):
        dx = part.pos - self.base_point
        dv = part.vel
        a =   (dv*dv).sum(axis=-1)
        b = 2*(dv*dx).sum(axis=-1)
        c =   (dx*dx).sum(axis=-1) - self.pw_gap_min**2
        t_small, t_big = solve_quadratic(a, b, c, mask=part.pw_mask[self.idx], non_negative=True)
        t = np.fmin(t_small, t_big)
        return t

    def make_mesh(self):
        mesh = sphere_mesh(dim=self.dim, radius=self.radius)
        self.mesh = mesh + self.base_point

        


class Particles():
    def __init__(self, **kwargs):
        params = {'max_steps':50, 'dim':2, 'num':1, 'radius':1.0, 'mass':1.0, 'pos_init':None, 'vel_init':None
                 ,'collision_law':'specular', 'cell_size':None}
        params.update(kwargs)
        
        for key, val in params.items():
            setattr(self, key, val)
        
        self.mass = self.expand(self.mass)
        self.radius = self.expand(self.radius)
        self.pp_gap_min = cross_subtract(self.radius, -self.radius)
        np.fill_diagonal(self.pp_gap_min, -1)

        if self.cell_size is None:
            self.cell_size = np.full(self.dim, np.inf)
        self.cell_size = np.asarray(self.cell_size, dtype=float)
        self.cell = np.zeros([self.num, self.dim], dtype=int)
        
        self.pos = np.full([self.num, self.dim], np.inf)
        self.vel = self.pos.copy()
        
        self.dt_pp = np.zeros([self.num, self.num], dtype='float')
        self.dt_pw = np.zeros([len(wall), self.num], dtype='float')
        self.pp_mask = self.dt_pp.copy().astype(bool)
        self.pw_mask = self.dt_pw.copy().astype(bool)

        self.make_mesh()
        
        self.t = 0.0
        self.col = {}
        self.t_hist = []
        self.pos_hist = []
        self.cell_hist = []
        self.vel_hist = []
        self.col_hist = []

    def pos_to_global(self):
        return self.pos + self.cell * self.cell_size * 2

    def get_pp_gap(self):
        dx = cross_subtract(self.pos_to_global())
        self.pp_gap = np.linalg.norm(dx, axis=-1) - self.pp_gap_min
        return self.pp_gap 
    
    def check_gap(self, soft=False, p=None):
        tol = abs_tol
        if soft == True:
            tol = -abs_tol
        self.get_pp_gap()
        self.pw_gap = np.array([w.get_pw_gap() for w in wall])
        pw_check = self.pw_gap > tol
        pp_check = self.pp_gap > tol
        if p is not None:
            pw_check = pw_check[:,p]
            pp_check = pp_check[:,p]
        return np.all(pw_check) and np.all(pp_check)

    def get_pp_col_time(self, mask=None):
        dx = cross_subtract(self.pos_to_global())
        dv = cross_subtract(self.vel)
        a =   (dv*dv).sum(axis=-1)
        b = 2*(dv*dx).sum(axis=-1)
        c =   (dx*dx).sum(axis=-1) - self.pp_gap_min**2
        t_small, t_big = solve_quadratic(a, b, c, mask=self.pp_mask, non_negative=True)
        t = np.fmin(t_small, t_big)
        return t

    def make_mesh(self):
        self.mesh = []
        for p in range(self.num):
            self.mesh.append(sphere_mesh(dim=self.dim, radius=self.radius[p]))
        self.mesh = np.asarray(self.mesh)

    def expand(self, X):
        Y = listify(X)
        fill = Y[-1]
        for i in range(len(Y), self.num):
            Y.append(fill)
        return np.asarray(Y, dtype=float)

    def set_given(self, target, given):
        if given is None:
            n = 0
        else:
            given = np.asarray(given)
            if given.ndim < target.ndim:
                given = given[np.newaxis]
            n, d = given.shape[:2]
            if d == self.dim:
                target[:n] = given.copy()
            else:
                print('invalid shape - ignoring')
        return n

    def set_vel_init(self, init=None):
        n = self.set_given(self.vel, init)
        self.vel[n:] = random_uniform_sphere(num=self.num-n, dim=self.dim, radius=1.0)
    
    def set_pos_init(self, init=None):
        n = self.set_given(self.pos, init)
        with np.errstate(invalid='ignore'):
            for p in range(n, self.num):
                    self.randomize_pos(p)
    
    def randomize_pos(self, p):
        r = self.radius[p]
        max_attempts = 50
        for attempt in range(max_attempts):
            for d in range(self.dim):
                self.pos[p,d] = rnd.uniform(-self.cell_size[d], self.cell_size[d])
            if self.check_gap(p=p) == True:
                break
        if self.check_gap(p=p) == False:
            raise Exception('Could not randomize position of particle {}'.format(p))

    def resolve_complex(self):
        part.record_state()
        part_involved = part.pw_mask.sum(axis=0) + part.pp_mask.sum(axis=0)
        for p in np.nonzero(part_involved)[0]:
            part.randomize_pos(p)
        part.pw_mask[:] = False
        part.pp_mask[:] = False
        
    def pp_specular_law(self, p1, p2):
        m1, m2 = self.mass[p1], self.mass[p2]
        M = m1 + m2
        nu = self.pos[p2] - self.pos[p1]
        dv = self.vel[p2] - self.vel[p1]
        w = proj(dv, nu)
        self.vel[p1] += 2 * (m2/M) * w
        self.vel[p2] -= 2 * (m1/M) * w
    
    def resolve_collision(self, p1, p2):
        if self.collision_law == 'specular':
            self.pp_specular_law(p1, p2)
        else:
            raise Exception('Unknown pp collision law {}'.format(self.collision_law))
    
    def record_state(self):
        self.t_hist.append(self.t)
        self.pos_hist.append(self.pos.copy())
        self.cell_hist.append(self.cell.copy())
        self.vel_hist.append(self.vel.copy())
        self.col_hist.append(self.col.copy())

            
def check():
    N = part.num
    D = part.dim
    if any([w.dim != D for w in wall]):
        raise Exception('Not all wall and part dimensions agree')
    if (part.pos.shape != (N,D)) or (part.vel.shape != (N,D)):
        raise Exception('Some dynamical variables do not have correct shape')
    if part.check_gap() == False:
        raise Exception('Particle escaped')

def run_trial(wall, part):
    check()
    part.record_state()
    for step in range(part.max_steps):
        if part.check_gap(soft=True) == False:
            raise Exception('A particle escaped')

        part.dt_pp = part.get_pp_col_time()
        for (i,w) in enumerate(wall):
            part.dt_pw[i] = w.get_pw_col_time()

        part.dt = min(part.dt_pp.min(), part.dt_pw.min())
        if np.isinf(part.dt):
            raise Exception("No future collisions detected")

        part.t += part.dt
        part.pos += part.vel * part.dt

        part.pp_mask = (part.dt_pp - part.dt) < abs_tol
        part.pw_mask = (part.dt_pw - part.dt) < abs_tol

        pw_events = part.pw_mask.sum()
        pp_events = part.pp_mask.sum() / 2
        if pw_events + pp_events > 1:
            print('Complex event - re-randomizing position of particles involved')
            part.resolve_complex()
        elif pw_events == 1:
            w, p = np.nonzero(part.pw_mask)
            w, p = w[0], p[0]
            part.col = {'w':w, 'p':p}
            wall[w].resolve_collision(part, p)
        elif pp_events == 1:
            p1, p2 = np.nonzero(part.pp_mask)
            p1, p2 = p1[0], p2[0]
            part.col = {'p1':p1, 'p2':p2}
            part.resolve_collision(p1, p2)
        else:
            raise Exception('Something bizarre happened')

        part.record_state()
        
    part.t_hist = np.asarray(part.t_hist)
    part.cell_hist = np.asarray(part.cell_hist)
    part.pos_hist = np.asarray(part.pos_hist)
    part.vel_hist = np.asarray(part.vel_hist)


    
#######################################################################################################
###  Graphics Functions ###
#######################################################################################################
import ipywidgets as widgets
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def draw_hist(wall, part):
    pos = 2 * part.cell_hist * part.cell_size + part.pos_hist
    dpos = np.diff(pos, axis=0)
    t = part.t_hist
    max_steps = t.shape[0]-1

    def draw(steps=1):        
        cell = part.cell_hist[:steps].T
        cell_range = [2 * part.cell_size[d] * np.arange(cell[d].min(), cell[d].max()+1) for d in range(part.dim)]
        translates = it.product(*cell_range)

        fig = plt.figure(figsize=[10,10])
        if part.dim == 2:
            ax = fig.gca()
            for trans in translates:
                for w in wall:
                    ax.plot(*(w.mesh + trans).T, color='black')

        cm = plt.cm.gist_rainbow
        idx = np.linspace(0, cm.N-1 , part.num).round().astype(int)
        clr = [cm(i) for i in idx]
        
        x = pos[:steps]
        dx = dpos[:steps]
        for p in range(part.num):
            ax.quiver(x[:,p,0], x[:,p,1], dx[:,p,0], dx[:,p,1], angles='xy', scale_units='xy', scale=1, color=clr[p])
            ax.plot(*(part.mesh[p] + pos[steps, p, :]).T, color=clr[p])
        ax.set_aspect('equal')
        plt.title('time = {:.4f}'.format(t[steps]))
        plt.show()

    l = widgets.Layout(width='150px')
    step_interval = 1
    step_text = widgets.BoundedIntText(min=1, max=max_steps, value=1, continuous_update=False, layout=l)
    step_slider = widgets.IntSlider(min=1, max=max_steps, value=1, readout=False, continuous_update=False, layout=l)
    play_button = widgets.Play(min=1, max=max_steps, interval=step_interval*10, layout=l)
    widgets.jslink((step_text, 'value'), (step_slider, 'value'))
    widgets.jslink((step_text, 'value'), (play_button, 'value'))
    
    img = widgets.interactive_output(draw, {'steps':step_text})
#     rept = widgets.interactive_output(report, {'steps':step_text})
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
###  Mathematical Functions ###
#######################################################################################################

def solve_quadratic(a, b, c, mask=None, non_negative=False):
    small = np.full_like(a, np.inf)
    d = b**2 - 4*a*c
    lin = (abs(a) < abs_tol) & (abs(b) >= abs_tol)
    quad = (abs(a) >= abs_tol) & (d >= abs_tol)
    
    small[lin] = - c[lin] / b[lin]
    big = small.copy()
    
    d[quad] = np.sqrt(d[quad])
    small[quad] = (-b[quad] - d[quad]) / (2*a[quad])
    big[quad] = (-b[quad] + d[quad]) / (2*a[quad])
    swap = (b >= abs_tol)
    small[swap], big[swap] = big[swap], small[swap]
    
    if mask is not None:
        small[mask] = np.inf
        big[mask] = np.inf
    if non_negative == True:
        small[small<0] = np.inf
        big[big<0] = np.inf
    return small, big

def random_uniform_sphere(num=1, dim=2, radius=1.0):
    pos = rnd.normal(size=[num, dim])
    pos = make_unit(pos, axis=1)
    return abs(radius) * pos

def random_uniform_ball(num=1, dim=2, radius=1.0):
    pos = random_uniform_sphere(num, dim, radius)
    r = rnd.uniform(size=[num, 1])
    return r**(1/dim) * pos
    
#######################################################################################################
### Linear Algebra Functions ###
#######################################################################################################

def proj(a, b):
    return (a.dot(b) / b.dot(b)) * b

def cross_subtract(u, v=None):
    if v is None:
        v=u.copy()
    w = u[:,np.newaxis] - v[np.newaxis,:]
    return w

def make_unit(A, axis=-1):
    A = np.asarray(A, dtype=float)
    M = np.linalg.norm(A, axis=axis, keepdims=True)
    return A / M

def make_onb(A):
    """
    Converts the rows of A into an orthonormal basis, adding extra vectors if necessary.
    """
    if A.ndim == 1:
        A = A[np.newaxis,:]

    # np.linalg.qr implents QR-factorization via Gram-Schmidt.  It does almost all of the work for us.
    # Option 'complete' will add extra vectors if necessary to complete the basis.    
    Q, R = np.linalg.qr(A.T, 'complete')
    
    # We are almost done - the columns of Q are almost the ONB we seek.
    # But we may need to multiply some columns by -1.  The lines below handle this. 
    S = R.diagonal()  # Negative entries correspond to the columns of Q that must be flipped
    idx = (abs(S) < abs_tol)
    S = np.sign(S)
    S[idx] = 1  # To prevent multiplying by 0
    W = np.ones(len(Q)-len(S))  # appending 1's if S is too short
    S = np.append(S,W)
    
    U = (Q*S).T  # Flips columns of Q that need it and then transpose so ONB is written in rows
    return U

#######################################################################################################
### Utility Functions ###
#######################################################################################################

def fill_nan(A, fill=np.inf):
    idx = np.isnan(A)
    A[idx] = fill

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
