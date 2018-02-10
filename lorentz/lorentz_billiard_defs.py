import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import itertools as it

abs_tol = 1e-5
rel_tol = 1 + 1e-5
rnd = np.random.RandomState(seed=seed)

def proj(a, b):
    return (a.dot(b) / b.dot(b)) * b


def solve_quadratic(a, b, c):
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
    return small, big



#def solve_quadratic(a, b, c):
#    small, big = np.inf, np.inf
#    if abs(a) < abs_tol:  #linear
#        if abs(b) > abs_tol:
#            t = - c / b
#            small, big = t, t
#    else:  #quadratic
#        d = b**2 - 4*a*c
#        if d > abs_tol:
#            e = np.sqrt(d)
#            small = (-b - e) / (2*a)
#            big   = (-b + e) / (2*a)
#            if b > 0:
#                small, big = big, small
#    return small, big
#solve_quadratic = np.vectorize(solve_quadratic)

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

def cross_subtract(u, v=None):
    if v is None:
        v=u.copy()
    w = u[:,np.newaxis] - v[np.newaxis,:]
    return w

def make_unit(X, axis=-1):
    X = np.asarray(X, dtype=float)
    M = np.linalg.norm(X, axis=axis, keepdims=True)
    return X / M

def uniform_sphere(num=1, dim=2, radius=1.0):
    pos = rnd.normal(size=[num,dim])
    pos = make_unit(pos, axis=1)
    return abs(radius) * pos

def uniform_ball(num=1, dim=2, radius=1.0):
    pos = uniform_sphere(num, dim, radius)
    r = rnd.uniform(size=[num,1])
    return r**(1/dim) * pos

def make_onb(A):
    """
    Converts the rows of A into an orthonormal basis, adding extra vectors if necessary.
    """
    B = np.asarray(A, dtype=float)
    if B.ndim == 1:
        B = B[np.newaxis,:]

    # np.linalg.qr implents QR-factorization via Gram-Schmidt.  It does almost all of the work for us.
    # Option 'complete' will add extra vectors if necessary to complete the basis.    
    Q, R = np.linalg.qr(B.T, 'complete')
    
    # We are almost done - the columns of Q are almost the ONB we seek.
    # But we may need to multiply some columns by -1.  The lines below handle this. 
    S = R.diagonal()  # Negative entries correspond to the columns of Q that must be flipped
    idx = np.isclose(S, 0)
    S = np.sign(S)
    S[idx] = 1  # To prevent multiplying by 0
    W = np.ones(len(Q)-len(S))  # appending 1's if S is too short
    S = np.append(S,W)
    
    U = (Q*S).T  # Flips columns of Q that need it and then transpose so ONB is written in rows
    return U



class Wall():
    def __init__(self, dim=2, base_point=None, collision_law='specular'):
        self.dim = dim
        if base_point is None:
            base_point = np.zeros(dim)
        self.base_point = np.asarray(base_point, dtype=float)
        self.collision_law = collision_law

    def pw_specular_law(self, part, p):
        nu = self.normal(part.pos[p])
        part.vel[p] -= 2 * part.vel[p].dot(nu) * nu
    
    def resolve_collision(self, part, p):
        if self.collision_law == 'specular':
            self.pw_specular_law(part, p)
    
    
class FlatWall(Wall):
    def __init__(self, dim=2, base_point=None, normal=None, tangents=None, collision_law='specular'):
        Wall.__init__(self, dim, base_point, collision_law)
        
        if tangents is None:
            O = normal
        else:
            O = np.vstack([normal, tangents])
            self.tangents = O[1:].copy()
        self.onb = make_onb(O)       
        self.normal_static = self.onb[0].copy()
        if tangents is None:
            self.tangents = self.onb[1:].copy()
        self.make_mesh()
        
    def normal(self, pos):
        return self.normal_static
        
    def make_mesh(self):
        pts = 100
        N, D = self.tangents.shape
        grid = [np.linspace(-1, 1, pts) for n in range(N)]
        grid = np.meshgrid(*grid)
        grid = np.asarray(grid)
        dx = grid.T.dot(self.tangents)
        self.mesh = (dx + self.base_point).T
    
    def get_pw_gap(self):
        dx = part.pos - self.base_point
        self.pw_gap = dx.dot(self.normal_static) - part.radius
        return self.pw_gap
        
    def get_pw_col_time(self, mask=None):
        t = np.full(part.num, np.inf)
        nu = self.normal_static
        dx = part.pos - self.base_point
        b = -1 * part.vel.dot(nu)
        c = dx.dot(nu) - part.radius
        idx = b > abs_tol
        t[idx] = c[idx] / b[idx]
        if mask is not None:
            t[mask] = np.inf
        t[t<0] = np.inf
        return t
    

class SphereWall(Wall):
    def __init__(self, dim=2, base_point=None, radius=1.0, collision_law='specular'):
        Wall.__init__(self, dim, base_point)
        self.radius = radius
        self.make_mesh()        
       
    def normal(self, pos):
        dx = pos - self.base_point
        return make_unit(dx)

    def make_mesh(self):
        pts = 100
        D = self.dim
        grid = [np.linspace(0, np.pi, pts) for d in range(D-1)]
        grid[-1] *= 2
        grid = np.meshgrid(*grid)                           
        dx = []
        for d in range(D):
            w = (self.radius + particle_radius) * np.ones_like(grid[0])
            for j in range(d):
                w *= np.sin(grid[j])
            if d < D-1:
                w *= np.cos(grid[d])
            dx.append(w)
        dx = np.asarray(dx).T
        self.mesh = (dx + self.base_point).T
        
    def get_pw_gap(self):
        dx = part.pos - self.base_point
        self.pw_gap = np.linalg.norm(dx, axis=-1) - (part.radius + self.radius)
        return self.pw_gap 

    def get_pw_col_time(self, mask=None):
        dx = part.pos - self.base_point
        dv = part.vel
        a =   (dv*dv).sum(axis=-1)
        b = 2*(dv*dx).sum(axis=-1)
        c =   (dx*dx).sum(axis=-1) - (part.radius + self.radius)**2
        t_small, t_big = solve_quadratic(a, b, c)
        if mask is not None:
            t_small[mask] = np.inf
        t_small[t_small<0] = np.inf
        t_big[t_big<0] = np.inf
        t = np.fmin(t_small, t_big)
        return t

        
class Particles():
    def __init__(self, dim=2, num=1, radius=1.0, mass=1.0, pos_init=None, vel_init=None, collision_law='specular'):
        self.dim = dim
        self.num = num
        self.radius = self.expand(radius)
        self.pp_gap_min = cross_subtract(self.radius, -self.radius)
        np.fill_diagonal(self.pp_gap_min, -1)
        self.mass = self.expand(mass)
        self.collision_law = collision_law
    
    def set_init(self, target, init):
        if init is None:
            n = 0
        else:
            init = np.asarray(init)
            if init.ndim < target.ndim:
                init = init[np.newaxis]
            n, d = init.shape[:2]
            if d == self.dim:
                target[:n] = init.copy()
            else:
                print('invalid shape - ignoring')
        return n

    
    def set_pos_init(self, pos_init=None):
        self.pos = np.full([self.num, self.dim], np.inf)
        n = self.set_init(self.pos, pos_init)
        for p in range(n, self.num):
            self.randomize_pos(p)

    def set_vel_init(self, vel_init=None):
        self.vel = np.full([self.num, self.dim], np.inf)
        n = self.set_init(self.vel, vel_init)
        self.vel[n:] = uniform_sphere(num=self.num-n, dim=self.dim, radius=1.0)
            
    def expand(self, X):
        Y = listify(X)
        fill = Y[-1]
        for i in range(len(Y), self.num):
            Y.append(fill)
        return np.asarray(Y, dtype=float)

    def get_pp_gap(self):
        dx = cross_subtract(self.pos)
        self.pp_gap = np.linalg.norm(dx, axis=-1) - self.pp_gap_min
        return self.pp_gap 

    
    def check_gap(self, soft=False):
        tol = abs_tol
        if soft == True:
            tol -= 1        
        pp_check = self.get_pp_gap() > tol
        pw_check = [w.get_pw_gap() > tol for w in wall]
        
        #self.pw_gap = np.asarray([w.dist(self.pos) for w in wall])

        #self.get_gaps()
        #pw_check = self.pw_gap > self.pw_gap_min * rel_tol
        #pp_check = self.pp_gap > self.pp_gap_min * rel_tol
        return np.all(pw_check) and np.all(pp_check)
    
    def randomize_pos(self, p):
        r = self.radius[p]
        max_attempts = 50
        for attempt in range(max_attempts):
            for d in range(self.dim):
                self.pos[p,d] = rnd.uniform(bounding_box[d][0]+r, bounding_box[d][1]-r)
            if self.check_gap() == True:
                attempt += 1
                break
        if self.check_gap() == False:
            raise Exception('Could not randomize position of particle {}'.format(p))
            
    def get_pp_col_time(self, mask=None):
        dx = cross_subtract(self.pos)        
        dv = cross_subtract(self.vel)
        a =   (dv*dv).sum(axis=-1)
        b = 2*(dv*dx).sum(axis=-1)
        c =   (dx*dx).sum(axis=-1) - self.pp_gap_min**2
        t_small, t_big = solve_quadratic(a, b, c)        
        if mask is not None:
            t_small[mask] = np.inf
        t_small[t_small<0] = np.inf
        t_big[t_big<0] = np.inf
        t = np.fmin(t_small, t_big)
        return t
        
    def pp_specular_law(self, p1, p2):
        m1, m2 = self.mass[p1], self.mass[p2]
        M = m1 + m2
        #nu = make_unit(self.pos[p2] - self.pos[p1])
        nu = self.pos[p2] - self.pos[p1]
        dv = self.vel[p2] - self.vel[p1]
        w = proj(dv, nu)
        self.vel[p1] += 2 * (m2/M) * w
        self.vel[p2] -= 2 * (m1/M) * w
    
    def resolve_collision(self, p1, p2):
        if self.collision_law == 'specular':
            self.pp_specular_law(p1, p2)


            
            
def check():
    N = part.num
    D = part.dim
    if any([w.dim != D for w in wall]):
        raise Exception('Not all wall and part dimensions agree')
    if (part.pos.shape != (N,D)) or (part.vel.shape != (N,D)):
        raise Exception('Some dynamical variable does not have correct shape')
    if part.check_gap() == False:
        raise Exception('Particle escaped')

        
        
        
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
