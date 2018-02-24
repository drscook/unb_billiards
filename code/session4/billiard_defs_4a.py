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

abs_tol = 1e-4
rnd = np.random.RandomState(seed)

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


    
    
class Wall():
    # default values that apply to all geometries
    Wall_defaults = {'dim':2, 'gap_pad':0.0, 'wp_collision_law':'wp_specular'}

    def wp_specular_law(self, part, p):
        nu = self.normal(part.pos[p])
        part.vel[p] -= 2 * part.vel[p].dot(nu) * nu

    def wp_terminate_law(self, part, p):
        raise Exception('particle {} hit termination wall {}'.format(p, self.idx))         
        
    # particle wraps around to the opposite side of the billiard cell by flipping sign on dim d
    def wp_wrap_law(self, part, p):
        d = self.wrap_dim   # which dim will have sign flip
        s = np.sign(part.pos[p, d]).astype(int)  # is it at + or -
        part.cell_offset[p, d] += s   # tracks cell position for each particle
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
        
    def resolve_collision(self, part, p):
        if self.wp_collision_law == 'wp_specular':
            self.wp_specular_law(part, p)
        elif self.wp_collision_law == 'wp_terminate':
            self.wp_terminate_law(part, p)
        elif self.wp_collision_law == 'wp_wrap':
            self.wp_wrap_law(part, p)
        elif self.wp_collision_law == 'wp_no_slip':
            self.wp_no_slip_law(part, p)
        else:
            raise Exception('Unknown collision law')
        
class FlatWall(Wall):
    def __init__(self, **kwargs):
        # convient way to combine Wall defaults, FlatWall defaults, and user specified attributes
        params = self.Wall_defaults.copy()
        # check key in kwargs against list of valid keys, throw error is not subset
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
        nu = self.normal_static
        dx = part.pos - self.base_point
        c = dx.dot(nu) - self.wp_gap_min
        b = part.vel.dot(nu)
        t = solve_linear(b, c, mask)
        return t
               
    # computes wp spacing
    def get_wp_gap(self):
        dx = part.pos - self.base_point
        self.wp_gap = dx.dot(self.normal_static) - self.wp_gap_min
        return self.wp_gap
    
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

    def get_wp_gap(self):
        dx = part.pos - self.base_point
        self.wp_gap = np.linalg.norm(dx, axis=-1) - self.wp_gap_min
        return self.wp_gap 
        
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
        constants = ['radius', 'mass', 'gamma']
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
        
        self.pp_gap_min = cross_subtract(self.radius, -self.radius)
        np.fill_diagonal(self.pp_gap_min, -1)  # no gap between a particle and itself
        
        self.wp_dt = np.zeros([len(wall), self.num], dtype='float')
        self.wp_mask = self.wp_dt.copy().astype(bool)

        if self.pp_collision_law == 'pp_ignore':
            self.pp_dt = np.array([np.inf])
        else:
            self.pp_dt = np.zeros([self.num, self.num], dtype='float')
        self.pp_mask = self.pp_dt.copy().astype(bool)
        
        self.t = 0.0
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

    def record_state(self):
        self.t_hist.append(self.t)
        self.pos_hist.append(self.pos_to_global())
        self.vel_hist.append(self.vel.copy())
        self.spin_hist.append(self.spin.copy())
        # we compute orient later in smoother
        #self.cell_offset_hist.append(self.cell_offset.copy())
        self.col_hist.append(self.col.copy())

    def get_pp_gap(self):
        dx = cross_subtract(self.pos_to_global())  #cross_subtract defined below
        self.pp_gap = np.linalg.norm(dx, axis=-1) - self.pp_gap_min
        return self.pp_gap 

    def check_gap(self, p=Ellipsis):
        # if p is specified, checks gap for particles in list p.  Else, checks all.
        self.wp_gap = np.array([w.get_wp_gap() for w in wall])
        wp_check = self.wp_gap > -abs_tol
        wp_check = wp_check[:,p]
        if self.pp_collision_law == 'pp_ignore':
            pp_check = [True]
        else:
            self.get_pp_gap()
            pp_check = self.pp_gap > -abs_tol
            pp_check = pp_check[:,p]
        return np.all(wp_check) and np.all(pp_check)

    def check_angular(self, p=Ellipsis):
        orient_det = np.abs(np.linalg.det(self.orient[p]))-1
        orient_det_check = np.abs(orient_det) < abs_tol
        S = self.spin[p]
        spin_skew = np.abs(S + np.swapaxes(S, -2, -1))
        spin_skew = spin_skew.sum(axis=-1).sum(axis=-1)
        spin_skew_check = np.abs(spin_skew) < abs_tol
        return np.all(orient_det_check) and np.all(spin_skew_check)
    
    # Computes time to next collision with for each p-p  pair via (x1+v1*t-x2-v1*t) dot (x1+v1*t-x2-v1*t) = (r1+r2)^2
    # Gives quadatric in t
    def get_pp_col_time(self, mask=None):
        dx = cross_subtract(self.pos_to_global())
        dv = cross_subtract(self.vel)
        a =   (dv*dv).sum(axis=-1)
        b = 2*(dv*dx).sum(axis=-1)
        c =   (dx*dx).sum(axis=-1) - self.pp_gap_min**2
        t_small, t_big = solve_quadratic(a, b, c, mask=self.pp_mask)
        t = np.fmin(t_small, t_big)
        return t

    def pp_specular_law(self, p1, p2):
        m1, m2 = self.mass[p1], self.mass[p2]
        M = m1 + m2
        nu = make_unit(self.pos[p2] - self.pos[p1])
        dv = self.vel[p2] - self.vel[p1]
        w = dv.dot(nu) * nu
        self.vel[p1] += 2 * (m2/M) * w
        self.vel[p2] -= 2 * (m1/M) * w    

    def pp_no_slip_law(self, p1, p2):
        m1 = part.mass[p1]
        m2 = part.mass[p2]
        M = m1 + m2
        g1 = part.gamma[p1]
        g2 = part.gamma[p2]
        r1 = part.radius[p1]
        r2 = part.radius[p2]        

        d = 2/((1/m1)*(1+1/g1**2) + (1/m2)*(1+1/g2**2))
        dx = part.pos[p2] - part.pos[p1]    
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
    
    def resolve_collision(self, p1, p2):
        if self.pp_collision_law == 'pp_specular':
            self.pp_specular_law(p1, p2)
        elif self.pp_collision_law == 'pp_no_slip':
            self.pp_no_slip_law(p1, p2)
        elif self.pp_collision_law == 'pp_ignore':
            raise Exception('Should not detect pp collisions')
        else:
            raise Exception('Unknown pp collision law {}'.format(self.collision_law))

    def get_KE(self):
        lin_KE = part.mass * (part.vel**2).sum(axis=-1)
        ang_KE = part.mom_inert * (np.triu(part.spin,1)**2).sum(axis=-1).sum(axis=-1)
        KE = lin_KE + ang_KE
        return KE / 2
        
    def pos_to_global(self):
        # self.pos is local to current cell.  This return the global position by adding cell offset.
        return self.pos + self.cell_offset * self.cell_size * 2

def check():
    N = part.num
    D = part.dim
    if np.any([w.dim != D for w in wall]):
        raise Exception('Not all wall and part dimensions agree')
    if (part.pos.shape != (N,D)) or (part.vel.shape != (N,D)):
        raise Exception('Some dynamical variables do not have correct shape')
    if np.any((part.gamma < 0) | (part.gamma > np.sqrt(2/part.dim))):
        raise Exception('illegal mass distribution parameter {}'.format(gamma))
    if part.check_gap() == False:
        raise Exception('A particle escaped')
    if part.check_angular() == False:
        raise Exception('A particle has invalid orintation or spin matrix')
    if np.abs(part.get_KE().sum() - part.KE_init) > abs_tol:
        raise Exception('Total kinetic energy was not conserved')

    
def next_state(wall, part):
    for (i,w) in enumerate(wall):
        part.wp_dt[i] = w.get_wp_col_time(part.wp_mask[i])
    if part.pp_collision_law != 'pp_ignore':
        part.pp_dt = part.get_pp_col_time(part.pp_mask)
    part.dt = np.min([np.min(part.pp_dt), np.min(part.wp_dt)])
    if np.isinf(part.dt):
        raise Exception("No future collisions detected")

    part.t += part.dt
    part.pos += part.vel * part.dt
    # We choose not to update orient during simulation because it does not affect the dynamics
    # and would slow us down.  We compute it later in smoother if needed.

    part.wp_mask = (part.wp_dt - part.dt) < 1e-8
    part.pp_mask = (part.pp_dt - part.dt) < 1e-8
    
    wp_counts = np.sum(part.wp_mask,axis=0)
    pp_counts = np.sum(part.pp_mask,axis=0)
    total_counts = wp_counts + pp_counts
    if np.any(total_counts > 1):
        raise Exception('Complex event - would re-randomize position of particles involved if implemented.  Until that is complete, simulation simply terminates.')
    else:
        part.col = []
        for (w, p) in zip(*np.nonzero(part.wp_mask)):
            part.col.append({'w':w, 'p':p})
            wall[w].resolve_collision(part, p)
        for (p1, p2) in zip(*np.nonzero(part.pp_mask)):
            if p1 < p2:
                part.col.append({'p':p1, 'q':p2})
                part.resolve_collision(p1, p2)

def clean_up(part):
    part.t_hist = np.asarray(part.t_hist)
    #part.cell_offset_hist = np.asarray(part.cell_offset_hist)
    part.pos_hist = np.asarray(part.pos_hist)
    part.vel_hist = np.asarray(part.vel_hist)
    part.spin_hist = np.asarray(part.spin_hist)
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

def interactive_plot(num_frames=-1):
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

    play_button = widgets.Play(min=0, max=num_frames, interval=50, layout=l)
    widgets.jslink((step_text, 'value'), (play_button, 'value'))

    img = widgets.interactive_output(update, {'s':step_text})
    display(widgets.HBox([widgets.VBox([step_text, step_slider, play_button]), img]))

def smoother(part, min_frames=None, orient=True):
    t, x, v, s = part.t_hist, part.pos_hist, part.vel_hist, part.spin_hist
    dts = np.diff(t)
    if (min_frames is None):
        ddts = dts
        num_frames = np.ones_like(dts).astype(int)
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
    A = np.asarray(A, dtype=float)
    M = np.linalg.norm(A, axis=axis, keepdims=True)
    return A / M

def solve_linear(b, c, mask=None):
    t = np.full(part.num, np.inf)  # default in np.inf    
    idx = np.abs(b) >= abs_tol  # prevents divide by zero
    t[idx] = -1 * c[idx] / b[idx]
    if mask is not None:
        t[mask] = np.inf
    t[t<0] = np.inf  #np.inf for negative times
    return t

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
