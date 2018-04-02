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
import scipy.linalg
from scipy.interpolate import InterpolatedUnivariateSpline


BOLTZ = 1.0
abs_tol = 1e-5
try:
    rnd = np.random.RandomState(seed)
except:
    rnd = np.random.RandomState(42)
np_dtype = np.float32


##############################################################################################################
##############################################################################################################

# Functions are listed in roughly in order of importance, but order of execution

#######################################################################################################
###  Dynamics ###
#######################################################################################################
def next_state(wall, part):
    part.get_col_time(wall)
    if np.isinf(part.dt):
        raise Exception("No future collisions detected")
    
    # We update positions on the CPU now but delay GPU updating until the end.
    # We choose not to update orient during simulation because it does not affect the dynamics
    # and would slow us down.  We compute it later in smoother if needed.
    part.t += part.dt
    part.pos_loc += part.vel * part.dt
    part.pos_glob += part.vel * part.dt
    
    pw_events = (part.pw_dt - part.dt) < abs_tol
    pw_counts = np.sum(pw_events, axis=1)
    pw_tot = np.sum(pw_counts)
    if isinstance(part.pp_collision_law, PP_IgnoreLaw) == False:
        pp_counts = (part.pp_dt - part.dt) < abs_tol
        pp_tot = np.sum(pp_counts)
    else:
        pp_counts = []
        pp_tot = 0
    
    if (pw_tot == 0) & (pp_tot == 2):
        p, q = np.nonzero(pp_counts)[0]
        part.col = {'p':p, 'q':q}
        part.pw_mask = []
        part.pp_mask = ((p,q), (q,p))
        part.resolve_collision(p, q)
        part.p_updated = [p, q]
    elif (pw_tot == 1) & (pp_tot == 0):
        p = np.argmax(pw_counts)
        w = np.argmax(pw_events[p])
        part.col = {'p':p, 'w':w}
        part.pw_mask = ((p,w))
        part.pp_mask = []
        wall[w].resolve_collision(part, p)
        part.p_updated = [p]
    else:
        part.p_updated = []
        for p in np.nonzero(pp_counts + pw_counts)[0]:
            part.rand_pos(p)
            part.p_updated.append(p)
        print('COMPLEX COLLISION DETECTED. Re-randomizing positions of particles {}'.format(part.p_updated))
    part.update_gpu()
    
def check():
    if part.check_pos() == False:
        raise Exception('A particle escaped')
    if part.check_angular() == False:
        raise Exception('A particle has invalid orintation or spin matrix')
    if part.check_KE() == False:
        raise Exception('Total kinetic energy was not conserved')

def init(wall, part):
    for (i, w) in enumerate(wall):
        w.idx = i
        w.pw_gap_min = w.gap_m * part.radius + w.gap_b
    
    if np.all([w.dim == part.dim for w in wall]) == False:
        raise Exception('Not all wall and part dimensions agree')
    if np.all((part.gamma >= 0) & (part.gamma <= np.sqrt(2/part.dim))) == False:
        raise Exception('illegal mass distribution parameter {}'.format(gamma))
        
    part.init_pos()
    part.init_vel()
    part.init_orient()
    part.init_spin()
        
    # check types
    arr = [part.pos_loc, part.pos_glob, part.vel, part.orient, part.spin, part.pp_dt, part.pw_dt , part.vel_update]
    for A in arr:
        A = np.asarray(A, dtype=np_dtype)
        
    # check init
    arr = [part.pos_loc, part.pos_glob, part.vel, part.orient, part.spin]
    for A in arr:
        if np.all(is_set(A)) == False:
            raise Exception('Some dyanamical variable have not been initialized')
    
    part.KE_init = np.sum(part.get_KE())
    part.record_state()
    check()
    part.load_gpu()



#######################################################################################################
###  Setup Functions ###
#######################################################################################################

def check_keys(params, required_keys):
    given_keys = set(params.keys())
    missing_keys = required_keys.difference(given_keys)
    invalid_keys = given_keys.difference(required_keys)
    if (len(missing_keys) > 0) | (len(invalid_keys) > 0):
        print('Required parameters missing: {}'.format(missing_keys))
        print('Invalid parameters given {}'.format(invalid_keys))
        raise Exception()
        
def record_params(obj, params):
    for key, val in params.items():
        if isinstance(val, list):
            val = np.asarray(val, dtype=np_dtype)  # converts lists to arrays
        setattr(obj, key, val)

def extend_param(param, length):
    param = listify(param)  #listify defined at bottom of this file
    for p in range(len(param), length):
        param.append(param[-1])
    return param

def is_set(A):
    return ~np.isinf(contract(A*A))

#######################################################################################################
###  Wall Classes ###
#######################################################################################################

class Wall():
    Wall_defaults = {'dim':2, 'gap_m':1.0, 'gap_b':0.0, 'pw_collision_law':'pw_specular', 'temp':1, 'misc':{}}

    def resolve_collision(self, part, p):
        self.pw_collision_law.resolve_collision(self, part, p)

    def get_pw_gap(self, p=Ellipsis):        
        return self.get_pw_col_coefs(p=p, gap_only=True)

    @staticmethod
    def normal(pos):
        raise Exception('You should implement the method normal() in a subclass.')

    @staticmethod
    def get_mesh():
        raise Exception('You should implement the method get_mesh() in a subclass.')

    @staticmethod
    def get_pw_col_coefs(self, part):
        raise Exception('You should implement the method get_pw_col_time() in a subclass.')


class FlatWall(Wall):
    def __init__(self, **kwargs):
        params = self.Wall_defaults.copy()
        params.update(kwargs)
        params['name'] = 'FlatWall'
        params['gap_b'] = 0.0
        if isinstance(params['pw_collision_law'], PW_WrapLaw):
            params['gap_m'] = 0.0
        else:
            params['gap_m'] = 1.0
        params['normal_static'] = make_unit(params.pop('normal'))  # renames normal -> normal-static and makes unit vector

        required_keys = {'name', 'dim', 'pw_collision_law', 'gap_m', 'gap_b', 'base_point', 'normal_static', 'tangents', 'temp', 'misc'}
        check_keys(params, required_keys)
        record_params(self, params)
        self.get_mesh()

    def get_mesh(self):
        self.mesh = flat_mesh(self.tangents) + self.base_point

    def normal(self, pos):
        # normal does not depend on collision point
        return self.normal_static

    def get_pw_col_coefs(self, p=Ellipsis, gap_only=False):
        if p is Ellipsis:
            dx = part.pos_loc - self.base_point
            dv = part.vel
            gap = self.pw_gap_min
        else:
            dx = part.pos_loc[p] - self.base_point
            dv = part.vel[p]
            gap = self.pw_gap_min[p]
        nu = self.normal_static
        c = dx.dot(nu) - gap
        if gap_only == True:
            return c
        b = dv.dot(nu)
        a = np.zeros(b.shape, dtype=b.dtype)
        return a, b, c

class SphereWall(Wall):
    def __init__(self, **kwargs):
        params = self.Wall_defaults.copy()
        params.update(kwargs)
        params['name'] = 'SphereWall'
        params['gap_b'] = params['radius']
        if isinstance(params['pw_collision_law'], PW_WrapLaw):
            params['gap_m'] = 0.0
        else:
            params['gap_m'] = 1.0
            
        required_keys = {'name', 'dim', 'pw_collision_law', 'gap_m', 'gap_b', 'base_point', 'radius', 'temp', 'misc'}
        check_keys(params, required_keys)
        record_params(self, params)
        self.get_mesh()

    def get_mesh(self):
        self.mesh = sphere_mesh(self.dim, self.radius) + self.base_point

    def normal(self, pos): # normal depends on collision point
        dx = pos - self.base_point
        return make_unit(dx)  # see below for make_unit

    def get_pw_col_coefs(self, p=Ellipsis, gap_only=False):
        if p is Ellipsis:
            dx = part.pos_loc - self.base_point
            dv = part.vel
            gap = self.pw_gap_min
        else:
            dx = part.pos_loc[p] - self.base_point
            dv = part.vel[p]
            gap = self.pw_gap_min[p]
        c = contract(dx*dx)
        if gap_only == True:
            return np.sqrt(c) - gap
        c -= gap**2
        b = contract(dx*dv) * 2
        a = contract(dv*dv)
        return a, b, c
        
#######################################################################################################
###  Particle Class ###
#######################################################################################################

class Particles():
    
### Setup ###
    mode = 'serial'
    Particle_defaults = {'max_steps':50, 'dim':2, 'num':1, 'cell_size':None, 'pp_collision_law':'pp_specular', 'radius':[1.0], 'mass':[1.0], 'gamma':'uniform', 'temp':1, 'misc':{}}
    
    def __init__(self, **kwargs):
        params = self.Particle_defaults.copy()
        params.update(kwargs)
        
        if(params['gamma'] == 'uniform'):
            params['gamma'] = np.sqrt(2/(2+params['dim']))
        elif(params['gamma'] == 'shell'):
            params['gamma'] = np.sqrt(2/params['dim'])
        elif(params['gamma'] == 'point'):
            params['gamma'] = 0

        # Each parameter list must be num_particles long.  This extends by filling with the last entry
        for const in ['radius', 'mass', 'gamma', 'temp']:
            params[const] = extend_param(params[const], params['num'])

        required_keys = {'max_steps', 'dim', 'num', 'cell_size', 'pp_collision_law', 'radius', 'mass', 'gamma', 'temp', 'misc'}
        check_keys(params, required_keys)
        record_params(self, params)
        self.get_mesh()

        self.pp_gap_min = cross_subtract(self.radius, -self.radius)
        np.fill_diagonal(self.pp_gap_min, -1*np.diag(self.pp_gap_min))  # no gap between a particle and itself
        
        self.dim_spin = int(self.dim * (self.dim - 1) / 2)
        self.mom_inert = self.mass * (self.gamma * self.radius)**2
        
        self.sigma_lin = np.sqrt(BOLTZ * self.temp / self.mass)
        self.sigma_spin = np.sqrt(BOLTZ * self.temp / self.mom_inert)

        self.pos_loc = np.full([self.num, self.dim], np.inf)
        self.pos_glob = self.pos_loc.copy()
        self.vel = self.pos_loc.copy()
        self.orient = np.full([self.num, self.dim, self.dim], np.inf)
        self.spin_vec = np.full([self.num, self.dim_spin], np.inf)
        
        self.t = 0.0
        self.dt = -1.0
        self.vel_update = np.full([2,self.dim+1], np.inf)
        
        self.pw_dt = np.full([self.num, len(wall)], np.inf)
        self.pp_dt = np.full(self.num, np.inf)
        self.pw_mask = []
        self.pp_mask = []
        self.pp_gap = np.full([self.num, self.num], np.inf)
        self.pw_gap = np.full([self.num, len(wall)], np.inf)
        
        self.p_updated = []
        

        self.cell_offset = np.zeros([self.num, self.dim], dtype=int)  # tracks which cell the particle is in
        self.col = {}
        self.t_hist = []
        self.col_hist = []
        self.pos_hist = []
        self.vel_hist = []
        self.spin_hist = []
        
        # Color particles (helpful for the future when we have many particles)
        cm = plt.cm.gist_rainbow
        idx = np.linspace(0, cm.N-1 , self.num).round().astype(int)
        self.clr = [cm(i) for i in idx]
        
    def get_mesh(self):
        self.mesh = []
        for p in range(self.num):
            R = self.radius[p]
            M = sphere_mesh(dim=self.dim, radius=R)
            if self.dim == 2:
                M = np.vstack([M, [-R,0]])  # draw equator
            self.mesh.append(M)
        self.mesh = np.asarray(self.mesh)
        
    def rand_pos(self, p):
        set_idx = is_set(self.pos_loc)
        max_attempts = 100
        for k in range(max_attempts):            
            self.pos_loc[p] = [rnd.uniform(-R, R) for R in self.cell_size]
            self.pos_glob[p] = self.pos_loc[p].copy()
            set_idx[p] = True
            if self.check_pos(set_idx, tol=abs_tol) == True:
                break
            else:
                set_idx[p] = False
        if set_idx[p] == False:
            raise Exception('Could not place particle {}'.format(p))
        
    def init_pos(self):
        self.pos_glob = self.pos_loc.copy()
        set_idx = is_set(self.pos_loc)
        if self.check_pos(set_idx, tol=abs_tol) == False:
            raise Exception('Illegal initial position specified by YOU, Mr./Ms./Mrs./Dr. User')
        for p in range(self.num):
            if set_idx[p] == False:
                 self.rand_pos(p)

    def init_orient(self):
        set_idx = is_set(self.orient)
        for p in range(self.num):
            if set_idx[p] == False:
                self.orient[p] = np.eye(self.dim, self.dim)

    def init_vel(self):
        set_idx = is_set(self.vel)
        for p in range(self.num):
            if set_idx[p] == False:
                self.vel[p] = [rnd.normal(0.0, self.sigma_lin[p]) for d in range(self.dim)]

    def init_spin(self):
        set_idx = is_set(self.spin_vec)
        for p in range(self.num):
            if set_idx[p] == False:
                self.spin_vec[p] = [rnd.normal(0.0, self.sigma_spin[p]) for d in range(self.dim_spin)]
        self.spin = np.full([self.num, self.dim, self.dim], np.inf)
        for p in range(self.num):
            self.spin[p] = spin_mat_from_vec(self.spin_vec[p])
        
### Dynamics ###

    def get_pos_glob(self):
        # self.pos_loc is local to current cell.  This computes the global position by adding cell offset.
        if self.cell_size is None:
            self.pos_glob = self.pos_loc
        else:        
            self.pos_glob = self.pos_loc + self.cell_offset * self.cell_size * 2
        return self.pos_glob
        
    def get_pp_col_coefs(self, p=Ellipsis, gap_only=False):
        if p is Ellipsis:
            x = self.pos_glob
            v = self.vel
            gap = self.pp_gap_min
        else:
            x = self.pos_glob[p]
            v = self.vel[p]
            gap = self.pp_gap_min[p][:,p]        
        dx = cross_subtract(x)
        c =   np.einsum('pqd,pqd->pq', dx, dx)
        if gap_only == True:
            return np.sqrt(c) - gap
        c -= gap**2
        dv = cross_subtract(v)
        b = 2*np.einsum('pqd,pqd->pq', dv, dx)
        a =   np.einsum('pqd,pqd->pq', dv, dv)
        return a, b, c
        
    def get_pp_gap(self, p=Ellipsis):
        return self.get_pp_col_coefs(p=p, gap_only=True)

    def get_col_time(self, wall):
        self.get_col_time_cpu(wall)

    def get_col_time_cpu(self, wall):
        if isinstance(self.pp_collision_law, PP_IgnoreLaw) == False:
            self.get_pp_col_time_cpu(wall)
        self.get_pw_col_time_cpu(wall)
        self.dt = min([np.min(self.pp_dt), np.min(self.pw_dt)])
        return self.dt

    def get_pp_col_time_cpu(self, wall):
        print('in pp cpu')
        a, b, c = self.get_pp_col_coefs()        
#         self.pp_a, self.pp_b, self.pp_c = a.copy(), b.copy(), c.copy()

        self.pp_dt_full = solve_quadratic(a, b, c, mask=self.pp_mask)        
        self.pp_dt = np.min(self.pp_dt_full, axis=1)
        return a, b, c
        
    def get_pw_col_time_cpu(self, wall):
        a = np.zeros([self.num, len(wall)])
        b = a.copy()
        c = a.copy()
        for (j,w) in enumerate(wall):
            a[:,j], b[:,j], c[:,j] = w.get_pw_col_coefs()
#         self.pw_a, self.pw_b, self.pw_c = a.copy(), b.copy(), c.copy()
        
        self.pw_dt = solve_quadratic(a, b, c, mask=self.pw_mask)        
        return a, b, c
        
    def resolve_collision(self, p1, p2):
        self.pp_collision_law.resolve_collision(self, p1, p2)

### Checks ###
        
    def check_pos(self, p=Ellipsis, tol=-1*abs_tol):
        self.pw_gap = np.array([w.get_pw_gap(p) for w in wall]).T
        pw_check = (self.pw_gap >= tol)
        if isinstance(self.pp_collision_law, PP_IgnoreLaw):
            pp_check = [True]
        else:
            self.pp_gap = self.get_pp_gap(p)
            pp_check = (self.pp_gap >= tol)
        return np.all(pw_check) and np.all(pp_check)

    def get_KE(self, p=Ellipsis):
        if p is Ellipsis:
            m = self.mass
            v = self.vel
            s = self.spin
            i = self.mom_inert
        else:
            m = self.mass[p]
            v = self.vel[p]
            s = self.spin[p]
            i = self.mom_inert[p]
        lin_KE = m * contract(v*v)
        ang_KE = i * contract(s*s) / 2
        KE = lin_KE + ang_KE
        return KE / 2

    def check_KE(self, p=Ellipsis, tol=abs_tol):
        return abs((np.sum(self.get_KE(p)) / self.KE_init[p]) - 1) < tol
    
    def check_angular(self, p=Ellipsis, tol=abs_tol):
        if p is Ellipsis:
            o = self.orient
            s = self.spin
        else:
            o = self.orient[p]
            s = self.spin[p]
        o_det = np.abs(np.linalg.det(o))-1
        orient_check = np.abs(o_det) < tol        
        skew = s + np.swapaxes(s, -2, -1)
        spin_check = contract(skew*skew) < tol
        return np.all(orient_check) and np.all(spin_check)

### Admin ###
    
    def record_state(self):
        self.t_hist.append(self.t)
        self.pos_hist.append(self.pos_glob.copy())
        self.vel_hist.append(self.vel.copy())
        self.spin_hist.append(self.spin.copy())
        # we compute orient later in smoother
        #self.cell_offset_hist.append(self.cell_offset.copy())
        self.col_hist.append(self.col.copy())

    def clean_up(self):
        part.t_hist = np.asarray(part.t_hist)
        #part.cell_offset_hist = np.asarray(part.cell_offset_hist)
        part.pos_hist = np.asarray(part.pos_hist)
        part.vel_hist = np.asarray(part.vel_hist)
        part.spin_hist = np.asarray(part.spin_hist)
        print('Done!! #Particles = {}, Steps = {}, Time = {:4f}'.format(part.num, len(part.t_hist)-1, part.t_hist[-1]))
        
    def update_gpu(self):
        pass # for parallel version
        
    def load_gpu(self):
        pass # for parallel version

    def check_sync(self):
        pass # for parallel version

#######################################################################################################
###  Collision Laws ###
#######################################################################################################

class PW_CollisionLaw:
    @staticmethod
    def resolve_collision(self, wall, part, p):
        raise Exception('You should implement the method resolve_collision() in a subclass.')

class PP_CollisionLaw:
    @staticmethod
    def resolve_collision(self, part, p1, p2):
        raise Exception('You should implement the method resolve_collision() in a subclass.')

##############################################################################################################

class PW_IgnoreLaw(PW_CollisionLaw):
    name = 'PW_IgnoreLaw'
    def resolve_collision(self, wall, part, p):
        pass

class PP_IgnoreLaw(PP_CollisionLaw):
    name = 'PP_IgnoreLaw'
    def resolve_collision(self, part, p1, p2):
        pass

##############################################################################################################

class PW_SpecularLaw(PW_CollisionLaw):
    name = 'PW_SpecularLaw'
    def resolve_collision(self, wall, part, p):
        nu = wall.normal(part.pos_loc[p])
        part.vel[p] -= 2 * part.vel[p].dot(nu) * nu

class PP_SpecularLaw(PP_CollisionLaw):
    name = 'PP_SpecularLaw'        
    def resolve_collision(self, part, p1, p2):
        m1, m2 = part.mass[p1], part.mass[p2]
        M = m1 + m2
        nu = make_unit(part.pos_glob[p2] - part.pos_glob[p1])
        dv = part.vel[p2] - part.vel[p1]
        w = dv.dot(nu) * nu
        part.vel[p1] += 2 * (m2/M) * w
        part.vel[p2] -= 2 * (m1/M) * w    

##############################################################################################################
        
class PW_TerminateLaw(PW_CollisionLaw):
    name = 'PW_TerminateLaw'
    def resolve_collision(self, wall, part, p):
        raise Exception('particle {} hit termination wall {}'.format(p, wall.idx))

class PP_TerminateLaw(PP_CollisionLaw):
    name = 'PP_TerminateLaw'
    def resolve_collision(self, part, p1, p2):
        raise Exception('particle-particle collision caused termination')

##############################################################################################################

class PW_PeriodicLaw(PW_CollisionLaw):
    name = 'PW_PeriodicLaw'
    def __init__(self, wrap_dim, wrap_wall):
        self.wrap_dim = wrap_dim
        self.wrap_wall = wrap_wall

    def resolve_collision(self, wall, part, p):
        d = self.wrap_dim   # which dim will have sign flip
        s = np.sign(part.pos_loc[p, d]).astype(int)  # is it at + or -        
        part.pos_loc[p, d] *= -1   # flips sign of dim d
        part.pw_mask = [(p, self.wrap_wall)]
        part.cell_offset[p, d] += s

class PW_WrapLaw(PW_PeriodicLaw):
    name = 'PW_WrapLaw'
    def resolve_collision(self, wall, part, p):
        super().resolve_collision(wall, part, p)
        part.pos_glob[p, d] *= -1   # flips sign of dim d
        part.cell_offset[p, d] = 0   # no offset for wrap

##############################################################################################################

#No-slip law in any dimension from private correspondence with Cox and Feres.
#See last pages of: https://github.com/drscook/unb_billiards/blob/master/references/no%20slip%20collisions/feres_N_dim_no_slip_law_2017.pdf
# Uses functions like Lambda_nu defined at the end of this file
class PW_NoSlipLaw(PW_CollisionLaw):
    name = 'PW_NoSlipLaw'
    def resolve_collision(self, wall, part, p):
        nu = wall.normal(part.pos_loc[p])
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

class PP_NoSlipLaw(PP_CollisionLaw):
    name = 'PP_NoSlipLaw'
    def resolve_collision(self, part, p1, p2):
        m1 = part.mass[p1]
        m2 = part.mass[p2]
        M = m1 + m2
        g1 = part.gamma[p1]
        g2 = part.gamma[p2]
        r1 = part.radius[p1]
        r2 = part.radius[p2]        

        d = 2/((1/m1)*(1+1/g1**2) + (1/m2)*(1+1/g2**2))
        dx = part.pos_glob[p2] - part.pos_glob[p1]
        nu = make_unit(dx)
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



        
        
    

#######################################################################################################    
#######################################################################################################
###  Support Functions ###
#######################################################################################################
#######################################################################################################


def solve_quadratic(a, b, c, mask=None):
    tol = 1e-12
    small = np.full(a.shape, np.inf)
    big = small.copy()
    b *= -1    
    lin = (np.abs(a) < tol) & (np.abs(b) >= tol)  #linear 
    small[lin] = c[lin] / b[lin]
    
    d = b**2 - 4*a*c  #discriminant
    quad = (np.abs(a) >= tol) & (d >= 0)  #quadratic
    d[quad] = np.sqrt(d[quad])
    small[quad] = (b[quad] - d[quad]) / (2 * a[quad])
    big[quad] = (b[quad] + d[quad]) / (2 * a[quad])
    
    # We want the solutions ordered small -> big by abs val, so we swap where needed
    swap = quad & (b < 0)
    small[swap], big[swap] = big[swap], small[swap]

    small[mask] = big[mask]
    big[mask] = np.inf
    
    small_idx = small < 0
    big_idx = big < 0
    clear_idx = small_idx & big_idx
    small[clear_idx] = np.inf
#     big[clear_idx] = np.inf
    swap_idx = small_idx & ~big_idx
    small[swap_idx] = big[swap_idx]
#     big[swap_idx] = np.inf
    return small#, big


#######################################################################################################
###  Graphics Functions ###
#######################################################################################################

def analyze_diffusion(start=1, anomalous=False):
    x0 = part.pos_hist[0]
    x = part.pos_hist[start:]
    t = part.t_hist[start:]

    displace = (x - x0).T
    delta = (displace[np.newaxis,:] * displace[:,np.newaxis])
    delta = np.mean(delta, axis=2)    
    if anomalous == False:
        #print('not anomalous')
        D_matrix = delta / (2 *t)
    else:
        #print('anomalous')
        D_matrix = delta / (2 * t * np.log(t))

    trace = np.einsum('dds->s', D_matrix)
    D_const = trace / part.dim
    
    fig, ax = plt.subplots(figsize=[8,8])
    for d in range(dim):
        for e in range(d+1):
            ax.plot(t, D_matrix[d,e]) 
    ax.plot(t, D_const, '.')
    plt.show()
    
    l = np.round(t.shape[0]/4).astype(int)
    D = np.mean(D_const[-l:])
    #D = D_const[-1]
    print(D)
    def norm_pdf(x,mu=0,var=1):
        return np.exp(-(x-mu)**2/(2*var)) / np.sqrt(2*np.pi*var)

    def interactive_plot(num_frames=-1):
        max_frames = x.shape[0]-1
        if (num_frames == -1) or (num_frames > max_frames):
            num_frames = max_frames
        s= t[:,np.newaxis,np.newaxis]
        if anomalous == False:
            #print('not anomalous')
            data = x / np.sqrt(s)
        else:
            #print('anomalous')
            data = x / np.sqrt(s * np.log(s))
        def update(s):
            fig, ax = plt.subplots()
            for d in range(part.dim):
                q = np.histogram(data[s,:,d], density=True)            
                h = q[1]
                dh = np.diff(h) / 2
                h = h[:-1] + dh
                hnew = np.linspace(h.min(),h.max(),300)
                v = InterpolatedUnivariateSpline(h,q[0])(hnew)
                if anomalous == False:
                    #print('not anomalous')
                    n = norm_pdf(hnew, mu=0, var=2*D)
                else:
                    #print('anomalous')
                    n = norm_pdf(hnew, mu=0, var=2*D)
                ax.plot(hnew,v)
                ax.plot(hnew,n)

        l = widgets.Layout(width='150px')
        step_text = widgets.BoundedIntText(min=2, max=num_frames, value=0, layout=l)
        step_slider = widgets.IntSlider(min=2, max=num_frames, value=0, readout=False, continuous_update=False, layout=l)
        widgets.jslink((step_text, 'value'), (step_slider, 'value'))

        play_button = widgets.Play(min=2, max=num_frames, step=1, interval=50, layout=l)
        widgets.jslink((step_text, 'value'), (play_button, 'value'))

        img = widgets.interactive_output(update, {'s':step_text})
        display(widgets.HBox([widgets.VBox([step_text, step_slider, play_button]), img]))
    interactive_plot()



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
    max_frames = part.re_pos.shape[0]-1
    if (num_frames == -1) or (num_frames > max_frames):
        num_frames = max_frames
        
    pos = part.re_pos[:num_frames+1]
    orient = part.re_orient[:num_frames+1]
    fig, ax = plt.subplots()
    draw_background(pos)
    for p in range(part.num):
        ax.plot(pos[:,p,0], pos[:,p,1], color=part.clr[p])
        ax.plot(*(part.mesh[p].dot(orient[-1,p].T) + pos[-1,p]).T, color=part.clr[p])
    ax.set_aspect('equal')
    plt.show()

def interactive_plot(num_frames=-1, duration=10):
    max_frames = part.re_pos.shape[0]-1
    if (num_frames == -1) or (num_frames > max_frames):
        num_frames = max_frames

    pos = part.re_pos[:num_frames+1]
    orient = part.re_orient[:num_frames+1]
    dpos = np.diff(pos, axis=0)  # position change    
    def update(s):
        fig, ax = plt.subplots(figsize=[8,8]);
        ax.set_aspect('equal')
        plt.title('s={} t={:.2f}'.format(s,part.re_t[s]))
        draw_background(pos[:s+1])
        for p in range(part.num):
            ax.plot(pos[:s+1,p,0], pos[:s+1,p,1], color=part.clr[p])
            ax.plot(*(part.mesh[p].dot(orient[s,p].T) + pos[s,p]).T, color=part.clr[p])
        plt.show()

    l = widgets.Layout(width='150px')
    step_text = widgets.BoundedIntText(min=0, max=num_frames, value=0, layout=l)
    step_slider = widgets.IntSlider(min=0, max=num_frames, value=0, readout=False, continuous_update=False, layout=l)
    widgets.jslink((step_text, 'value'), (step_slider, 'value'))
    
    
    play_button = widgets.Play(min=0, max=num_frames, interval=1000*duration/num_frames, layout=l)
    widgets.jslink((step_text, 'value'), (play_button, 'value'))

    img = widgets.interactive_output(update, {'s':step_text})
    display(widgets.HBox([widgets.VBox([step_text, step_slider, play_button]), img]))

def smoother(part, min_frames=None, orient=True):
    print('smoothing')
    t, x, v, s = part.t_hist, part.pos_hist, part.vel_hist, part.spin_hist
    dts = np.diff(t)
    if (min_frames is None):
        ddts = dts
        num_frames = np.ones(dts.shape, dtype=int)
    else:
        short_step = dts < abs_tol
        nominal_frame_length = np.percentile(dts[~short_step], 25) / min_frames
        num_frames = np.round(dts / nominal_frame_length).astype(int) # Divide each step into pieces of length as close to nominal_frame_length as possible
        num_frames[num_frames<1] = 1
        ddts = dts / num_frames  # Compute frame length within each step

    # Now interpolate.  re_x denotes the interpolated version of x
    re_t, re_x, re_v, re_s = [t[0]], [x[0]], [v[0]], [s[0]]
    re_o = [part.orient]
    for (i, ddt) in enumerate(ddts):
        re_t[-1] = t[i]
        re_x[-1] = x[i]
        re_v[-1] = v[i]
        re_s[-1] = s[i]
        dx = re_v[-1] * ddt
        if orient == True:
            do = [scipy.linalg.expm(ddt * U) for U in re_s[-1]] # incremental rotatation during each frame

        for f in range(num_frames[i]):
            re_t.append(re_t[-1] + ddt)
            re_x.append(re_x[-1] + dx)
            re_v.append(re_v[-1])
            re_s.append(re_s[-1])
            if orient == True:
                #B = [A.dot(Z) for (A,Z) in zip(re_o[-1], do)] # rotates each particle the right amount
                B = np.einsum('pde,pef->pdf', re_o[-1], do)
                re_o.append(np.array(B))
            else:
                re_o.append(re_o[-1])

    part.re_t = np.asarray(re_t)
    part.re_pos = np.asarray(re_x)
    part.re_vel = np.asarray(re_v)
    part.re_orient = np.asarray(re_o)
    part.re_spin = np.asarray(re_s)    
    
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
def spin_mat_from_vec(v):
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

def spin_vec_from_mat(M):
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
def contract(A, keepdims=[0]):
    keepdims = listify(keepdims)
    return np.einsum(A, range(A.ndim), keepdims)

def cross_subtract(u, v=None):
    if v is None:
        v=u.copy()
    w = u[np.newaxis,:] - v[:,np.newaxis]
    return w

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
    A = np.asarray(A, dtype=np_dtype)
    M = np.linalg.norm(A, axis=axis, keepdims=True)
    return A / M

def listify(X):
    """
    Convert X to list if it's not already
    """
    if isinstance(X, list):
        return X
    elif (X is None) or (X is np.nan):
        return []
    elif isinstance(X,str):
        return [X]
    else:
        try:
            return list(X)
        except:
            return [X]
