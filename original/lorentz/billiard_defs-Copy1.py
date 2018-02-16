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
rnd = np.random.RandomState(seed=seed)   # We all get the same random values if we use the same see


### Classes are containers - they keep all the information (attributes) and functions (methods) for a component of our system to
### one place.  It make the code easier and cleaner.  We have a class for particles and for walls.
### We make sub-class for each wall shape for the geometry dependent information.


### wp_mask and pp_mask track the previous collision so that we don't detect that collision again.
### Without this check, we can get stuck detecting the same collision over and over.
### If the previous event was wall 2 and particle 4, wp_mask[2,4]=True and all other entries of wp_mask and pp_mask are false.  
### If the previous event was particles 1 and 3 pp_mask[1,3]=True, pp_mask[3,1]=True and all other entries of wp_mask and pp_mask are false.

### To handle unbounded domains (such as the Lorentz gas), we partition the domain into cells.  We track local position within
### a cell in attribute part.pos and cell index in part.cell.  We provide pos_to_global to convert from local to global position.
### In this case, use wp_wrap_law with gap_pad=0.

### no_slip laws implement the no-slip collision from Broomhead-Gutkin and Cox-Feres.
### BS gave explicit expressions for dim  2 https://github.com/drscook/unb_billiards/blob/master/references/no%20slip%20collisions/broomhead_gutkin_1992.pdf
### CF gave a general classification for dim n in https://arxiv.org/pdf/1501.06536.pdf
### but this was not explicit enough to code.
## We use an explicit dim n expression from private correspondence with CF to appear in a future paper.  See last pages of:
### https://github.com/drscook/unb_billiards/blob/master/references/no%20slip%20collisions/feres_N_dim_no_slip_law_2017.pdf


### Each wall can compute the distance to each particle's center using wp_gap.  User may also specify gap_pad - the minimum
### gap between particle and wall.  Often we set this equal to the particles radius, but not for "wrap around" motion on a torus.

### Each wall and particle has "make_mesh" for later visualization.

### Walls and particles have methods to compute time to next collision.  We use np.inf as the default.  For example, 2 particle moving
### away from wall 1 has a negative time to next impact.  We record this as wall dt_wp[1,2] = np.inf.

### Main wall class with information that is not geomtry depended
class Wall():
    # default values that apply to all geometries
    Wall_defaults = {'dim':2, 'gap_pad':0.0, 'wp_collision_law':'specular', 'wrap_wall':None, 'wrap_dim':None}
    
    # usual angle in-angle out law
    def wp_specular_law(self, part, p):
        nu = self.normal(part.pos[p])
        part.vel[p] -= 2 * proj(part.vel[p], nu)

    # particle wraps around to the opposite side of the billiard cell by flipping sign on dim d
    def wp_wrap_law(self, part, p):
        d = self.wrap_dim   # which dim will have sign flip
        s = np.sign(part.pos[p, d]).astype(int)  # is it at + or -
        part.cell[p, d] += s   # tracks cell position for each particle
        part.pos[p, d] *= -1   # flips sign of dime d
        part.wp_mask[self.idx, p] = False
        part.wp_mask[self.wrap_wall, p] = True

    
    # Particle-wall no-slip law in any dimension from private correspondence with Cox and Feres.
    #See last pages of: https://github.com/drscook/unb_billiards/blob/master/references/no%20slip%20collisions/feres_N_dim_no_slip_law_2017.pdf
    # Uses functions like Lambda_nu defined at the end of this file
    def wp_no_slip_law(self, part, p):  
        nu = self.normal(part.pos[p])
        m = part.mass[p]
        g = part.gamma[p]
        r = part.radius[p]
        d = (2*m*g**2)/(1+g**2)
        
        U_in = part.spin[p]
        v_in = part.vel[p]
        U_out = U_in - (d/(m*g**2) * Lambda_nu(U_in, nu)) + (d/(m*r*g**2)) * E_nu(v_in, nu)
        v_out = (r*d/m) * Gamma_nu(U_in, nu) + v_in - 2 * Pi_nu(v_in, nu) - (d/m) * Pi(v_in,nu)

        part.spin[p] = U_out
        part.vel[p] = v_out

    # applies specified collision law
    def resolve_collision(self, part, p):
        if self.wp_collision_law == 'specular':
            self.wp_specular_law(part, p)
        elif self.wp_collision_law == 'wrap':
            self.wp_wrap_law(part, p)
        elif self.wp_collision_law == 'no_slip':
            self.wp_no_slip_law(part, p)
        else:
            raise Exception('Unknown wp collision law {}'.format(self.collision_law))
    
class FlatWall(Wall):
    def __init__(self, **kwargs):
        # convient way to combined geomtery independent defaults, geomtery dependent defaults, and user specified attributes
        params = self.Wall_defaults.copy()
        params.update(kwargs)
        dim = params['dim']
        FlatWall_defaults = {'base_point': np.zeros(dim), 'normal':np.ones(dim), 'tangents':None}
        params.update(FlatWall_defaults)
        params.update(kwargs)
        
        params['normal_static'] = params.pop('normal')  # renames normal -> normal-static
        for key, val in params.items():  # saves params into the wall objeect
            setattr(self, key, val)
        self.base_point = np.asarray(self.base_point, dtype=float)

        if self.tangents is None: # if tangents are not specified, create them using make_onb defined below
            O = self.normal_static
            self.onb = make_onb(O)
            self.tangents = self.onb[1:].copy()
        else: # if tangents are specified, make sure they form an orthonormal basis
            O = np.vstack([self.normal_static, self.tangents])
            self.tangents = O[1:].copy()
            self.onb = make_onb(O)
        self.normal_static = self.onb[0].copy()

        self.wp_gap_min = self.gap_pad
        self.make_mesh()
        
    def normal(self, pos):  # normal does not depend on collision point
        return self.normal_static
        
    # computes wp spacing
    def get_wp_gap(self):
        dx = part.pos - self.base_point
        self.wp_gap = dx.dot(self.normal_static) - self.wp_gap_min
        return self.wp_gap
        
    # computes time to next collision with each particle via proj (x+vt) onto normal = wp_gap_main
    def get_wp_col_time(self):
        t = np.full(part.num, np.inf)  # default in np.inf
        nu = self.normal_static
        dx = part.pos - self.base_point
        b = -1 * part.vel.dot(nu)
        c = dx.dot(nu) - self.wp_gap_min
        idx = b > abs_tol  # prevents divide by zero
        t[idx] = c[idx] / b[idx]
        mask = part.wp_mask[self.idx]  #  np.inf for previous collision
        t[mask] = np.inf
        t[t<0] = np.inf  #np.inf for negative times
        return t

    def make_mesh(self):
        mesh = flat_mesh(tangents=self.tangents)
        self.mesh = mesh + self.base_point
    
class SphereWall(Wall):
    def __init__(self, **kwargs):
        ### see comments in FlatWall
        params = self.Wall_defaults.copy()
        params.update(kwargs)
        dim = params['dim']
        SphereWall_defaults = {'radius':1.0, 'base_point': np.zeros(dim)}
        params.update(SphereWall_defaults)
        params.update(kwargs)

        for key, val in params.items():
            setattr(self, key, val)

        self.base_point = np.asarray(self.base_point, dtype=float)
        self.wp_gap_min = self.radius + self.gap_pad
        self.make_mesh()
       
    def normal(self, pos): # normal depends on collision point
        dx = pos - self.base_point
        return make_unit(dx)

    def get_wp_gap(self):
        dx = part.pos - self.base_point
        self.wp_gap = np.linalg.norm(dx, axis=-1) - self.wp_gap_min
        return self.wp_gap 

    # computes time to next collision with each particle via (x+vt-center) dot (x+vt-center) = wp_gap_main^2
    # Gives quadatric in t
    def get_wp_col_time(self):
        dx = part.pos - self.base_point
        dv = part.vel
        a =   (dv*dv).sum(axis=-1)
        b = 2*(dv*dx).sum(axis=-1)
        c =   (dx*dx).sum(axis=-1) - self.wp_gap_min**2
        t_small, t_big = solve_quadratic(a, b, c, mask=part.wp_mask[self.idx], non_negative=True)
        t = np.fmin(t_small, t_big)
        return t

    def make_mesh(self):
        mesh = sphere_mesh(dim=self.dim, radius=self.radius)
        self.mesh = mesh + self.base_point

        

### Orientation and spin recorded as d x d matrices.  Orient must be orthogonal (element of Lie group SO(d).  Spin must
### be skew-symmetric (element of Lie algebra so(d)).
### https://en.wikipedia.org/wiki/Rotation_matrix
### Update rule: a particle leaving collision with orientation R and spin S will have orientiation Re^(dt*S) at time dt later

### Goal: allow user to specify attributes for a subset of particles and use reasonable defaults for the remainer.
### This requires some extra machinery that is not interesting from a mathematical perspective.
### Please don't focus on this initialization machinery.
class Particles():
    def __init__(self, **kwargs):
        # See comments in FlatWall
        params = {'max_steps':50, 'dim':2, 'num':1, 'radius':1.0, 'mass':1.0, 'gamma':'uniform'
                 ,'pos_init':None, 'vel_init':None, 'pp_collision_law':'specular', 'cell_size':None}
        params.update(kwargs)
        
        for key, val in params.items():
            setattr(self, key, val)
        
        self.mass = self.expand(self.mass)
        self.radius = self.expand(self.radius)
        
        # Controls radial mass distribution.  Mass is concentrated at center for small gamma and around outside for large gamma.
        # See p.3 of https://arxiv.org/pdf/1612.03355.pdf
        if(self.gamma == 'uniform'):
            self.gamma = np.sqrt(2/(2+self.dim))
        elif(self.gamma == 'shell'):
            self.gamma = np.sqrt(2/self.dim)
        self.gamma = self.expand(self.gamma)
        self.mom_inert = self.mass * (self.gamma * self.radius)**2

        if self.cell_size is None:
            self.cell_size = np.full(self.dim, 100*self.radius[0])
        self.cell_size = np.asarray(self.cell_size, dtype=float)
        self.cell = np.zeros([self.num, self.dim], dtype=int)  # tracks which cell the particle is in
        
        self.pp_gap_min = cross_subtract(self.radius, -self.radius)
        np.fill_diagonal(self.pp_gap_min, -1)  # no gap between a particle and itself

        # records time to next collision between all pp and wp pairs, with default np.inf
        self.dt_pp = np.zeros([self.num, self.num], dtype='float')
        self.dt_wp = np.zeros([len(wall), self.num], dtype='float')
        
        # track previous collision event and ignore the associated dt
        self.pp_mask = self.dt_pp.copy().astype(bool)
        self.wp_mask = self.dt_wp.copy().astype(bool)

        self.make_mesh()
        
        # records all states of the system for future use
        self.t = 0.0
        self.col = {}
        self.t_hist = []
        self.pos_hist = []
        self.vel_hist = []
        self.orient_hist = []
        self.spin_hist = []
        self.cell_hist = []
        self.col_hist = []

### Goal: allow user to specify attributes for a subset of particles and use reasonable defaults for the remainer.
### This requires some extra machinery that is not interesting from a mathematical perspective.
### Please don't focus on this initialization machinery.
    
    def expand(self, X):
        # Copies given parameters to a list of length self.num - one value for each particle
        Y = listify(X)
        fill = Y[-1]
        for i in range(len(Y), self.num):
            Y.append(fill)
        return np.asarray(Y, dtype=float)

    def set_given(self, target, given):
        # Write given as first entries of target
        if given is None:
            n = 0
        else:
            given = np.asarray(given)
            if given.ndim < target.ndim:
                given = given[np.newaxis]
            n = given.shape[0]
            target[:n] = given.copy()
        return n
    
    def randomize_pos(self, p):
        r = self.radius[p]
        max_attempts = 50
        # Draw random position and checks if valid.  Repeats until valid is found.  Gives up after max_attempts.
        for attempt in range(max_attempts):
            for d in range(self.dim):
                self.pos[p,d] = rnd.uniform(-self.cell_size[d], self.cell_size[d])
            if self.check_gap(p=p) == True:
                break
        if self.check_gap(p=p) == False:
            raise Exception('Could not randomize position of particle {}'.format(p))

    def set_pos_init(self, init=None):
        self.pos = np.full([self.num, self.dim], np.inf)
        n = self.set_given(self.pos, init)
        # Draw unspecified positions randomly
        with np.errstate(invalid='ignore'): # Suppresses warnings from np.inf of particles not yet positioned
            for p in range(n, self.num):
                    self.randomize_pos(p)

    def set_vel_init(self, init=None):
        self.vel = np.full([self.num, self.dim], np.inf)
        n = self.set_given(self.vel, init)
        # Draw unspecified velocities randomly
        for p in range(n, self.num):
            self.vel[p] = random_uniform_sphere(num=1, dim=self.dim, radius=1.0)
    
    def set_orient_init(self, init=None):
        self.orient = np.full([self.num, self.dim, self.dim], np.inf)
        n = 0
        # Insist that all orientation start at the identity matrix
        for p in range(n, self.num):
            self.orient[p] = np.eye(self.dim, self.dim)

    def set_spin_init(self, init=None):
        m = int(self.dim * (self.dim - 1) / 2)
        S = np.full([self.num, m], np.inf)
        n = self.set_given(S, init)
        # Draw unspecified spins randomly
        for p in range(n, self.num):
            S[p] = random_uniform_sphere(num=1, dim=m, radius=1.0)
        # Reshapes spin vector into skew-symmetric matrix.  See functions below.
        self.spin = np.asarray([spin_matrix_from_vector(s) for s in S])
                
    def check_angular(self, p=Ellipsis):
        orient_det = np.abs(np.linalg.det(self.orient[p]))-1
        orient_det_check = np.abs(orient_det) < abs_tol
        S = self.spin[p]
        spin_skew = np.abs(S + np.swapaxes(S, -2, -1))
        spin_skew = spin_skew.sum(axis=-1).sum(axis=-1)
        spin_skew_check = np.abs(spin_skew) < abs_tol
        return np.all(orient_det_check) and np.all(spin_skew_check)


    def pos_to_global(self):
        # self.pos is local to current cell.  This return the global position by adding cell offset.
        return self.pos + self.cell * self.cell_size * 2

    def get_pp_gap(self):
        dx = cross_subtract(self.pos_to_global())  #cross_subtract defined below
        self.pp_gap = np.linalg.norm(dx, axis=-1) - self.pp_gap_min
        return self.pp_gap 
    
    def check_gap(self, soft=False, p=Ellipsis):
        # if p is specified, checks gap for particles in list p.  Else, checks all.
        if soft == True:
            tol = -abs_tol
        else:
            tol = abs_tol
        self.wp_gap = np.array([w.get_wp_gap() for w in wall])
        wp_check = self.wp_gap > tol
        wp_check = wp_check[:,p]
        if self.pp_collision_law == 'ignore':
            pp_check = True
        else:
            self.get_pp_gap()
            pp_check = self.pp_gap > tol
            pp_check = pp_check[:,p]
        return np.all(wp_check) and np.all(pp_check)

    # Computes time to next collision with for each p-p  pair via (x1+v1*t-x2-v1*t) dot (x1+v1*t-x2-v1*t) = (r1+r2)^2
    # Gives quadatric in t
    def get_pp_col_time(self, mask=None):
        dx = cross_subtract(self.pos_to_global())
        dv = cross_subtract(self.vel)
        a =   (dv*dv).sum(axis=-1)
        b = 2*(dv*dx).sum(axis=-1)
        c =   (dx*dx).sum(axis=-1) - self.pp_gap_min**2
        t_small, t_big = solve_quadratic(a, b, c, mask=self.pp_mask, non_negative=True)
        t = np.fmin(t_small, t_big)
        return t

    def get_KE(part):
        lin_KE = part.mass * (part.vel**2).sum(axis=-1)
        ang_KE = part.mom_inert * (np.triu(part.spin,1)**2).sum(axis=-1).sum(axis=-1)
        KE = lin_KE + ang_KE
        return KE / 2

    def make_mesh(self):
        self.mesh = []
        for p in range(self.num):
            R = self.radius[p]
            M = sphere_mesh(dim=self.dim, radius=R)
            if self.dim == 2:
                M = np.vstack([M, [-R,0]])  # draw equator
            self.mesh.append(M)
        self.mesh = np.asarray(self.mesh)

    def resolve_complex(self):
        #part.record_state()
        part_involved = part.wp_mask.sum(axis=0) + part.pp_mask.sum(axis=0)
        for p in np.nonzero(part_involved)[0]:
            part.randomize_pos(p)
        part.wp_mask[:] = False
        part.pp_mask[:] = False

    def pp_specular_law(self, p1, p2):
        m1, m2 = self.mass[p1], self.mass[p2]
        M = m1 + m2
        nu = self.pos[p2] - self.pos[p1]
        dv = self.vel[p2] - self.vel[p1]
        w = proj(dv, nu)
        self.vel[p1] += 2 * (m2/M) * w
        self.vel[p2] -= 2 * (m1/M) * w

    # Particle-particle no-slip law in any dimension from private correspondence with Cox and Feres.
    # See last pages of: https://github.com/drscook/unb_billiards/blob/master/references/no%20slip%20collisions/feres_N_dim_no_slip_law_2017.pdf
    # Uses functions like Lambda_nu defined at the end of this file
    def pp_no_slip_law(self, p1, p2):
        print('Using pp no slip')
        m1 = part.mass[p1]
        m2 = part.mass[p2]
        M = m1 + m2
        g1 = part.gamma[p1]
        g2 = part.gamma[p2]
        r1 = part.radius[p1]
        r2 = part.radius[p2]        

        d = 2/((1/m1)*(1+1/g1**2) + (1/m2)*(1+1/g2**2))
        dx = part.pos[p2] - part.pos[p1]    
        nu = munit(dx)
        U1_in = part.spin[p1]
        U2_in = part.spin[p2]
        v1_in = part.vel[p1]
        v2_in = part.vel[p2]

        U1_out = (U1_in-d/(m1*g1**2) * Lambda_nu(U1_in, nu)) \
                    + (-d/(m1*r1*g1**2)) * E_nu(v1_in, nu) \
                    + (-r2/r1)*(d/(m1*g1**2)) * Lambda_nu(U2_in, nu) \
                    + d/(m1*r1*g1**2) * E_nu(v2_in, nu)

        v1_out = (-r1*d/m1) * Gamma_nu(U1_in, nu) \
                    + (v1_in - 2*m2/M * Pi_nu(v1_in, nu) - (d/m1) * Pi(v1_in, nu)) \
                    + (-r2*d/m1) * Gamma_nu(U2_in, nu) \
                    + (2*m2/M) * Pi_nu(v2_in, nu) + (d/m1) * Pi(v2_in, nu)

        U2_out = (-r1/r2)*(d/(m2*g2**2)) * Lambda_nu(U1_in, nu) \
                    + (-d/(m2*r2*g2**2)) * E_nu(v1_in, nu) \
                    + (U2_in - (d/(m2*g2**2)) * Lambda_nu(U2_in, nu)) \
                    + (d/(m2*r2*g2**2)) * E_nu(v2_in, nu)

        v2_out = (r1*d/m2) * Gamma_nu(U1_in, nu) \
                    + (2*m1/M) * Pi_nu(v1_in, nu) + (d/m2) * Pi(v1_in, nu) \
                    + (r2*d/m2) * Gamma_nu(U2_in, nu) \
                    + v2_in - (2*m1/M) * Pi_nu(v2_in, nu) - (d/m2) * Pi(v2_in,nu)
        part.spin[p1] = U1_out
        part.spin[p2] = U2_out
        part.vel[p1] = v1_out
        part.vel[p2] = v2_out    
    
    def resolve_collision(self, p1, p2):
        if self.pp_collision_law == 'specular':
            self.pp_specular_law(p1, p2)
        elif self.pp_collision_law == 'no_slip':
            self.pp_no_slip_law(self, p1, p2)
        elif self.pp_collision_law == 'ignore':
            pass
        else:
            raise Exception('Unknown pp collision law {}'.format(self.collision_law))
    
    def record_state(self):
        self.t_hist.append(self.t)
        self.pos_hist.append(self.pos.copy())        
        self.vel_hist.append(self.vel.copy())
        self.orient_hist.append(self.orient.copy())
        self.spin_hist.append(self.spin.copy())
        self.cell_hist.append(self.cell.copy())
        self.col_hist.append(self.col.copy())

            
def check():
    N = part.num
    D = part.dim
    if any([w.dim != D for w in wall]):
        raise Exception('Not all wall and part dimensions agree')
    if (part.pos.shape != (N,D)) or (part.vel.shape != (N,D)):
        raise Exception('Some dynamical variables do not have correct shape')
    if np.any((part.gamma < 0) | (part.gamma > np.sqrt(2/part.dim))):
        raise Exception('illegal mass distribution parameter {}'.format(gamma))
    if part.check_gap(soft=True) == False:
        raise Exception('A particle escaped')
    if part.check_angular() == False:
        raise Exception('A particle has invalid orintation or spin matrix')
    if np.abs(part.get_KE().sum() - part.KE_init) > abs_tol:
        raise Exception('Total kinetic energy was not conserved')

        
                     
def run_trial(wall, part):
    part.KE_init = part.get_KE().sum()
    for step in range(part.max_steps):
        check()
        part.record_state()
        
        for (i,w) in enumerate(wall):
            part.dt_wp[i] = w.get_wp_col_time()
            
        if part.pp_collision_law == 'ignore':
                part.dt_pp
        part.dt_pp = part.get_pp_col_time()
        part.dt = np.min([np.min(part.dt_pp), np.min(part.dt_wp)])
        if np.isinf(part.dt):
            raise Exception("No future collisions detected")

        part.t += part.dt
        part.pos += part.vel * part.dt

        part.pp_mask = (part.dt_pp - part.dt) < abs_tol
        part.wp_mask = (part.dt_wp - part.dt) < abs_tol

        wp_events = part.wp_mask.sum()
        pp_events = part.pp_mask.sum() / 2
        if wp_events + pp_events > 1:
            print('Complex event - re-randomizing position of particles involved')
            part.resolve_complex()
        elif wp_events == 1:
            w, p = np.nonzero(part.wp_mask)
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
    part.orient_hist = np.asarray(part.orient_hist)
    part.spin_hist = np.asarray(part.spin_hist)


    
#######################################################################################################
###  Graphics Functions ###
#######################################################################################################
import ipywidgets as widgets
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import animation
from mpl_toolkits.mplot3d import Axes3D
import scipy.linalg
from IPython.display import HTML
matplotlib.rc('animation', html='html5')



# Interpolate between collisions AND compute correct orientation.  Note that, for efficiency, orientation was
# not computed during simulation because it was not needed (all collision laws are orientation invariant).  
def smoother(part, min_frames=None):
    t, c, x, v, o, s = part.t_hist, part.cell_hist, part.pos_hist, part.vel_hist, part.orient_hist, part.spin_hist
    dts = np.diff(t)
    if (min_frames is None):
        ddts = dts
        num_frames = np.ones_like(dts).astype(int)
    else:
        short_step = (dts < abs_tol)
        nominal_frame_length = np.min(dts[~short_step]) / min_frames
        num_frames = np.round(dts / nominal_frame_length).astype(int) # Divide each step into pieces of length as close to nominal_frame_length as possible
        num_frames[short_step] = 1
        ddts = dts / num_frames  # Compute frame length within each step

    # Now interpolate.  re_x denotes the interpolated version of x
    re_t, re_c, re_x, re_v, re_o, re_s = [t[0]], [c[0]], [x[0]], [v[0]], [o[0]], [s[0]]
    for (i, ddt) in enumerate(ddts):
        re_t[-1] = t[i]
        re_c[-1] = c[i]
        re_x[-1] = x[i] + c[i] * part.cell_size * 2
        re_v[-1] = v[i]
        re_s[-1] = s[i]        
        dx = re_v[-1] * ddt
        do = [scipy.linalg.expm(ddt * A) for A in re_s[-1]] # incremental rotatation during each frame

        for f in range(num_frames[i]):
            re_t.append(re_t[-1] + ddt)
            re_c.append(re_c[-1])
            re_x.append(re_x[-1] + dx)
            re_v.append(re_v[-1])
            re_s.append(re_s[-1])
            U = [R.dot(O) for (R,O) in zip(re_o[-1], do)] # rotates each particle the right amount
            re_o.append(np.array(U))

    part.re_t = np.asarray(re_t)
    part.re_cell = np.asarray(re_c)
    part.re_pos = np.asarray(re_x)
    part.re_vel = np.asarray(re_v)
    part.re_orient = np.asarray(re_o)
    part.re_spin = np.asarray(re_s)

    
def animate_new(num_frames=None, duration=None, fps=None):
    if (num_frames is None) + (duration is None) + (fps is None) != 1:
        print('Must specify exactly 2 of num_frames, duration, and fps')
        return None
    elif (num_frames is None):
        num_frames = int(np.round(fps * duration))
    elif (fps is None):
        fps = num_frames / duration
    duration = num_frames / fps

    fig, ax = plt.subplots(figsize=[10,10]);
    ax.set_aspect('equal')

    pos = part.re_pos[:num_frames+1]
    orient = part.re_orient[:num_frames+1]
    cell = part.re_cell[:num_frames+1].T
    cell_range = [2 * s * np.arange(np.min(c), np.max(c)+1) for (s,c) in zip(part.cell_size, cell)]
    translates = it.product(*cell_range)
    if part.dim == 2:
        for trans in translates:
            for w in wall:
                ax.plot(*(w.mesh + trans).T, color='black')

    time_label = fig.text(0.5, 0.90, 's=0 t=0.00', ha='center', va='top', fontsize=10)
    
    path = []
    bdy = []
    for p in range(part.num):
        path.append(ax.plot([], [])[0])    
        bdy.append(ax.plot([], [])[0])

    def update(i):
        s = max(i,1)
        time_label.set_text('s={} t={:.2f}'.format(s,part.re_t[s]))
        for p in range(part.num):
            path[p].set_data(pos[:s+1,p,0], part.re_pos[:s+1,p,1])            
            bdy[p].set_data(*(part.mesh[p].dot(part.re_orient[s,p].T) + part.re_pos[s,p]).T)
        return path + bdy
    plt.close()
    
    anim = animation.FuncAnimation(fig, update, frames=num_frames+1, interval=1000/fps, blit=True, repeat=False)
    
    return 
    
    
    
    
    
def draw(steps):
    
    
    
    fig, ax = plt.subplots(figsize=[10,10]);
    ax.set_aspect('equal')

    pos = part.re_pos[:num_frames+1]
    orient = part.re_orient[:num_frames+1]
    cell = part.re_cell[:num_frames+1].T
    cell_range = [2 * s * np.arange(np.min(c), np.max(c)+1) for (s,c) in zip(part.cell_size, cell)]
    translates = it.product(*cell_range)
    if part.dim == 2:
        for trans in translates:
            for w in wall:
                ax.plot(*(w.mesh + trans).T, color='black')
    return fig, ax

                
def draw_hist_old(wall, part, duration=10):
    dpos = np.diff(part.re_pos, axis=0)
    max_steps = dpos.shape[0]

    def draw(steps=1):        
        cell = part.re_cell[:steps+1].T
        cell_range = [2 * s * np.arange(np.min(c), np.max(c)+1) for (s,c) in zip(part.cell_size, cell)]
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
        
        x = part.re_pos[:steps+1]
        R = part.re_orient[steps]
        dx = dpos[:steps]
        for p in range(part.num):
            ax.quiver(x[:-1,p,0], x[:-1,p,1], dx[:,p,0], dx[:,p,1], angles='xy', scale_units='xy', scale=1, color=clr[p])#, headwidth=1)
            ax.plot(*(part.mesh[p].dot(R[p].T) + x[-1,p]).T, color=clr[p])
        ax.set_aspect('equal')
        plt.title('time = {:.4f}'.format(part.re_t[steps]))
        plt.show()

    l = widgets.Layout(width='150px')
    step_interval = 1000*duration / max_steps
    step_text = widgets.BoundedIntText(min=1, max=max_steps, value=1, continuous_update=True, layout=l)
    step_slider = widgets.IntSlider(min=1, max=max_steps, value=1, readout=False, continuous_update=True, layout=l)
    play_button = widgets.Play(min=1, max=max_steps, interval=step_interval, layout=l)
    widgets.jslink((step_text, 'value'), (step_slider, 'value'))
    widgets.jslink((step_text, 'value'), (play_button, 'value'))
    
    img = widgets.interactive_output(draw, {'steps':step_text})
#     rept = widgets.interactive_output(report, {'steps':step_text})
    display(widgets.HBox([widgets.VBox([step_text, step_slider, play_button]), img]))
                
    
def draw_hist_old(wall, part, duration=10):
    dpos = np.diff(part.re_pos, axis=0)
    max_steps = dpos.shape[0]

    def draw(steps=1):        
        cell = part.re_cell[:steps+1].T
        cell_range = [2 * s * np.arange(np.min(c), np.max(c)+1) for (s,c) in zip(part.cell_size, cell)]
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
        
        x = part.re_pos[:steps+1]
        R = part.re_orient[steps]
        dx = dpos[:steps]
        for p in range(part.num):
            ax.quiver(x[:-1,p,0], x[:-1,p,1], dx[:,p,0], dx[:,p,1], angles='xy', scale_units='xy', scale=1, color=clr[p])#, headwidth=1)
            ax.plot(*(part.mesh[p].dot(R[p].T) + x[-1,p]).T, color=clr[p])
        ax.set_aspect('equal')
        plt.title('time = {:.4f}'.format(part.re_t[steps]))
        plt.show()

    l = widgets.Layout(width='150px')
    step_interval = 1000*duration / max_steps
    step_text = widgets.BoundedIntText(min=1, max=max_steps, value=1, continuous_update=True, layout=l)
    step_slider = widgets.IntSlider(min=1, max=max_steps, value=1, readout=False, continuous_update=True, layout=l)
    play_button = widgets.Play(min=1, max=max_steps, interval=step_interval, layout=l)
    widgets.jslink((step_text, 'value'), (step_slider, 'value'))
    widgets.jslink((step_text, 'value'), (play_button, 'value'))
    
    img = widgets.interactive_output(draw, {'steps':step_text})
#     rept = widgets.interactive_output(report, {'steps':step_text})
    display(widgets.HBox([widgets.VBox([step_text, step_slider, play_button]), img]))
    
    
def animate(num_frames=None, duration=None, fps=None):
    if (num_frames is None) + (duration is None) + (fps is None) != 1:
        print('Must specify exactly 2 of num_frames, duration, and fps')
        return None
    elif (num_frames is None):
        num_frames = int(np.round(fps * duration))
    elif (fps is None):
        fps = num_frames / duration
    duration = num_frames / fps

    fig, ax = plt.subplots(figsize=[10,10]);
    ax.set_aspect('equal')

    pos = part.re_pos[:num_frames+1]
    orient = part.re_orient[:num_frames+1]
    cell = part.re_cell[:num_frames+1].T
    cell_range = [2 * s * np.arange(np.min(c), np.max(c)+1) for (s,c) in zip(part.cell_size, cell)]
    translates = it.product(*cell_range)
    if part.dim == 2:
        for trans in translates:
            for w in wall:
                ax.plot(*(w.mesh + trans).T, color='black')

    time_label = fig.text(0.5, 0.90, 's=0 t=0.00', ha='center', va='top', fontsize=10)
    
    
    path = []
    bdy = []
    for p in range(part.num):
        path.append(ax.plot([], [])[0])    
        bdy.append(ax.plot([], [])[0])

    def update(i):
        s = max(i,1)
        time_label.set_text('s={} t={:.2f}'.format(s,part.re_t[s]))
        for p in range(part.num):
            path[p].set_data(pos[:s+1,p,0], part.re_pos[:s+1,p,1])            
            bdy[p].set_data(*(part.mesh[p].dot(part.re_orient[s,p].T) + part.re_pos[s,p]).T)
        return path + bdy
    plt.close()
    return animation.FuncAnimation(fig, update, frames=num_frames+1, interval=1000/fps, blit=True, repeat=False)
    

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
    """
    Quadratic equation solver.  True entries in mask return infinity.
    """
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
### No-Slip Collision Functions ###
#######################################################################################################
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
### Linear Algebra Functions ###
#######################################################################################################
def wedge(a,b):
    return np.outer(b,a)-np.outer(a,b)

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
