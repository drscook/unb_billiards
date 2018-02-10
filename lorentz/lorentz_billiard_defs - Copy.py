import numpy as np
import matplotlib.pyplot as plt
import itertools as it

abs_tol = 1e-5
rel_tol = 0.01
rnd = np.random.RandomState(seed=seed)

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

def mag(X, axis=-1, keepdims=False):
    X = np.asarray(X, dtype=float)
    Y = (X**2).sum(axis=axis, keepdims=keepdims)
    return np.sqrt(Y)

def make_unit(X, axis=-1):
    X = np.asarray(X, dtype=float)
    M = mag(X, axis=axis, keepdims=True)
    return X / M

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
    def __init__(self, dim=2, base_point=None):
        self.dim = dim
        self.base_point = np.asarray(base_point, dtype=float)    
   
    
class FlatWall(Wall):
    def __init__(self, dim=2, base_point=None, normal=None, tangents=None):
        Wall.__init__(self, dim, base_point)
        
        if tangents is None:
            O = normal
            self.tangents = None
        else:
            O = np.vstack([normal, tangents])
            self.tangents = O[1:].copy()
        self.onb = make_onb(O)
        self.normal_static = self.onb[0].copy()
        self.make_mesh()
        
    def normal(self, pos):
        return self.normal_static
        
    def make_mesh(self):
        pts = 100
        n, d = self.tangents.shape
        s = np.linspace(-1, 1, pts)
        grid = it.product(s, repeat=n)
        grid = np.asarray(list(grid), dtype=float)
        dx = grid.dot(self.tangents)
        self.mesh = self.base_point + dx
    
    def dist(self, pos):
        dx = pos - self.base_point
        return dx.dot(self.normal_static)


class SphereWall(Wall):
    def __init__(self, dim=2, base_point=None, radius=1.0):
        Wall.__init__(self, dim, base_point)
        self.radius = radius
        self.make_mesh()        
       
    def normal(self, pos):
        dx = pos - self.base_point
        return make_unit(dx)

    def make_mesh(self):
        pts = 100
        D = self.dim
        s = np.linspace(-1, 1, pts) * np.pi
        grid = it.product(s, repeat=D-1)
        grid = np.asarray(list(grid), dtype=float)
        dx = []
        for d in range(D):
            w = np.ones(pts)
            for j in range(d):
                w = w * np.sin(grid[:,j])
            if d < D-1:
                w = np.cos(grid[:,d])
            dx.append(w)
        dx = self.radius * np.asarray(dx).T
        self.mesh = self.base_point + dx
    
        
    def dist(self, pos):
        dx = pos - self.base_point
        print(pos)
        print(dx)
        print(dx.shape)
        z = mag(dx)
        print(z)
        return z

        
class Particles():
    def __init__(self, dim=2, num=1, radius=1.0):
        self.dim = dim
        self.num = num
        self.radius = np.asarray(self.expand(radius), dtype=float)
        self.pos = np.full([num, dim], np.inf)
        print(self.pos.shape)
        self.vel = self.pos.copy()
        
        for p in range(self.num):
            self.randomize_pos(p)
        
    def expand(self, X):
        Y = listify(X)
        fill = Y[-1]
        for i in range(len(Y), self.num):
            Y.append(fill)
        return Y

    def check_position(self, p):
        wall_check = [w.dist(self.pos[p]) > self.radius[p] * rel_tol for w in wall]
        part_gap = mag(self.pos[p] - self.pos, axis=-1)
        part_check = part_gap > (self.radius[p] + self.radius) * rel_tol
        return all(wall_check) and all(part_check)
    
    def randomize_pos(self, p):
        r = self.radius[p]
        max_attempts = 50
        for attempt in range(max_attempts):
            for d in range(self.dim):
                self.pos[p,d] = rnd.uniform(bounding_box[d][0]+r, bounding_box[d][1]-r)
            if self.check_position(p) is True:
                break
        if self.check_position(p) is True:
            raise Exception('Could not randomize position of particle {}'.format(p))
            
            
            if attempt >= max_attempts:
                print(self.pos)
                raise Exception('Could not place particle {}'.format(p))
            
            #self.pos[p] = lab_frame.dot(c)
            success = self.check_position(p)
            attempt += 1
        return success
