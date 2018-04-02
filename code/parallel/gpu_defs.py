import math
import numba as nb
import numba.cuda as cuda
nb_dtype = nb.float32
threads_per_block_max = 1024
sqrt_threads_per_block_max = int(np.floor(np.sqrt(threads_per_block_max)))

class Parallel(Particles):
    mode = 'parallel'
    def load_gpu(self):
        print('load_gpu')
        
        if part.num < part.dim:
            raise Exception('Can not use parallel processing when part_num < dim')
            
#         self.get_col_time_cpu(wall)
#         self.pp_a_gpu = nb.SmartArray(self.pp_a)
#         self.pp_b_gpu = nb.SmartArray(self.pp_b)
#         self.pp_c_gpu = nb.SmartArray(self.pp_c)
#         self.pp_dt_full_gpu = nb.SmartArray(self.pp_dt_full)
            
        self.pp_mask_gpu = nb.SmartArray(np.array([self.num, self.num]))
        self.pw_mask_gpu = nb.SmartArray(np.array([self.num, len(wall)]))
        self.pp_dt_block_gpu = nb.SmartArray(np.full([self.num, pp_grid_shape[0]], np.inf, dtype=np_dtype))
        
        self.radius_gpu = nb.SmartArray(self.radius)
        self.pos_loc_gpu = nb.SmartArray(self.pos_loc)
        self.pos_glob_gpu = nb.SmartArray(self.pos_glob)
        self.vel_gpu = nb.SmartArray(self.vel)
        self.pw_dt_gpu = nb.SmartArray(self.pw_dt)
        
        self.wall_base_point_gpu = cuda.to_device(np.vstack([w.base_point for w in wall]))
        self.wall_normal_gpu = cuda.to_device(np.vstack([w.normal for w in wall]))
        self.pw_gap_min_gpu = cuda.to_device(np.vstack([w.pw_gap_min for w in wall]))
       
    def update_gpu(self):
        self.update_pos_gpu[[pp_grid_shape[0],1],[pp_block_shape[0],self.dim]](self.dt_gpu, self.pos_glob_gpu, self.vel_gpu)
        for p in self.p_updated:
            self.pos_glob_gpu[p] = self.pos_glob[p]
            self.pos_glob_gpu.mark_changed('host')
            self.vel_gpu[p] = self.vel[p]
            self.vel_gpu.mark_changed('host')

        try:
            self.pp_mask_gpu[:] = self.pp_mask[0]
        except:
            self.pp_mask_gpu[:] = [self.num, self.num]
        self.pp_mask_gpu.mark_changed('host')
        
        try:
            self.pw_mask_gpu[:] = self.pw_mask[0]
        except:
            self.pw_mask_gpu[:] = [self.num, len(wall)]
        self.pw_mask_gpu.mark_changed('host')
            
            
    @cuda.jit
    def update_pos_gpu(dt, pos, vel):
        dim = cuda.threadIdx.y
        q = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
        pos[q, dim] += vel[q, dim] * dt
        

    def check_sync(self):
        sync = (np.allclose(self.pos_glob, self.pos_glob_gpu, atol=abs_tol) & np.allclose(self.vel, self.vel_gpu, atol=abs_tol))
        if sync == False:
            print('pos')
            print(self.pos_glob - self.pos_glob_gpu)
            print('vel')
            print(self.vel - self.vel_gpu)
            raise Exception('not synced')
        else:
            print('In sync')
    
        self.get_pp_col_time_cpu(wall)    
        if np.allclose(self.pp_dt, self.pp_dt_gpu, atol=abs_tol):
            raise Exception('times do not match')
        else:
            print('times match')

    
    def get_col_time(self, wall):
        if isinstance(self.pp_collision_law, PP_IgnoreLaw) == False:
            self.get_pp_col_time_gpu[pp_grid_shape, pp_block_shape](self.pp_dt_block_gpu, self.pos_glob_gpu, self.vel_gpu, self.radius_gpu, self.pp_mask_gpu, self.num)#, self.pp_dt_full_gpu, self.pp_a_gpu, self.pp_b_gpu, self.pp_c_gpu)
            self.pp_dt_gpu = np.min(self.pp_dt_block_gpu.get(), axis=1)

#         self.get_col_time_cpu(wall)
#         if np.allclose(self.pp_dt, self.pp_dt_gpu, atol=abs_tol) == False:
#             raise Exception('times do not match')

        self.get_pw_col_time_cpu(wall)
        self.pw_dt_gpu = self.pw_dt

        self.dt_gpu = min(np.min(self.pw_dt_gpu), np.min(self.pp_dt_gpu))
        self.pp_dt = self.pp_dt_gpu
        self.dt = self.dt_gpu
        
    @cuda.jit
    def get_pp_col_time_gpu(pp_dt, pos, vel, radius, mask, N):#, pp_dt_full, a_gpu, b_gpu, c_gpu):
        pp_dt_shr = cuda.shared.array(shape=(pp_brows, pp_bcols), dtype=nb_dtype)
        tx = cuda.threadIdx.x
        ty = cuda.threadIdx.y
        p = ty + cuda.blockIdx.y * cuda.blockDim.y
        q = tx + cuda.blockIdx.x * cuda.blockDim.x
        if ((p < N) and (q < N)):
            a = 0.0
            b = 0.0
            c = 0.0
            for d in range(dim):
                dx = pos[p,d] - pos[q,d]
                dv = vel[p,d] - vel[q,d]
                a += dv * dv
                b += dx * dv * 2
                c += dx * dx
            c -= (radius[p] + radius[q])**2

            if ((mask[0]==p) & (mask[1]==q)):
                masked = True
            elif ((mask[0]==q) & (mask[1]==p)):
                masked = True
            else:
                masked = False

            pp_dt_shr[ty,tx] = solve_quadratic_gpu(a, b, c, masked)
        else:
            pp_dt_shr[ty,tx] = 987
            a = p
            b = q
            c = N

#         a_gpu[p,q] = a
#         b_gpu[p,q] = b
#         c_gpu[p,q] = c            
#         pp_dt_full[p,q] = pp_dt_shr[ty,tx]

        cuda.syncthreads()
        row_min_gpu(pp_dt_shr)
        pp_dt[p, cuda.blockIdx.x] = pp_dt_shr[ty, 0]
        cuda.syncthreads()

        
        
@cuda.jit(device=True)
def solve_quadratic_gpu(a, b, c, mask=False):
    tol = 1e-16
    small = np.inf
    big = np.inf
    b *= -1
    if abs(a) < tol:
        if abs(b) >= tol:
            small = c / b
    else:
        d = b**2 - 4 * a * c
        if d >= 0:
            d = math.sqrt(d)
            f = 2 * a
            if b >= 0:
                small = (b - d) / f
                big  =  (b + d) / f
            else:
                small = (b + d) / f
                big  =  (b - d) / f
    if mask == True:
        small = np.inf
    if small < 0:
        small = np.inf
    if big < 0:
        big = np.inf
    return min(small, big)

#     if mask == True:
#         small = big
#         big = np.inf
#     if small < 0:
#         if big < 0:
#             small = np.inf
#             big = np.inf
#         else:
#             small = big
#             big = np.inf
#     return min(small, big)


@cuda.jit(device=True)
def row_min_gpu(A):
    tx = cuda.threadIdx.x
    ty = cuda.threadIdx.y
    m = float(cuda.blockDim.x)
    while m > 1:
        n = m / 2
        k = int(math.ceil(n))
        if (tx + k) < m:
            if A[ty,tx] > A[ty,tx+k]:
                A[ty,tx] = A[ty,tx+k]
        m = n
        cuda.syncthreads()

        
        
####### old versions      
        
#     @cuda.jit
#     def get_pp_col_time_gpu(pp_dt, pos, vel, radius, mask):#, a_gpu, b_gpu, c_gpu):
#         tx = cuda.threadIdx.x
#         ty = cuda.threadIdx.y
#         p = ty + cuda.blockIdx.y * cuda.blockDim.y
#         q = tx + cuda.blockIdx.x * cuda.blockDim.x

#         pp_dt_shr = cuda.shared.array(shape=(pp_brows, pp_bcols), dtype=nb_dtype)
#         pos_shr = cuda.shared.array(shape=(pp_bcols, dim), dtype=nb_dtype)
#         vel_shr = cuda.shared.array(shape=(pp_bcols, dim), dtype=nb_dtype)
#         radius_shr = cuda.shared.array(shape=pp_bcols, dtype=nb_dtype)

#         if ty < dim:
#             pos_shr[tx, ty] = pos[q, ty]
#             vel_shr[tx, ty] = vel[q, ty]
#         if ty == 0:
#             radius_shr[tx] = radius[q]
#         cuda.syncthreads()
                
#         a = 0.0
#         b = 0.0
#         c = 0.0
#         x = pos[p]
#         v = vel[p]
#         r = radius[p]
#         for d in range(dim):
#             dx = x[d] - pos_shr[tx, d]
#             dv = v[d] - vel_shr[tx, d]
#             a += dv * dv
#             b += dx * dv * 2
#             c += dx * dx
#         c -= (r + radius_shr[tx])**2

#         if ((mask[0]==p) & (mask[1]==q)):
#             masked = True
#         elif ((mask[0]==q) & (mask[1]==p)):
#             masked = True
#         else:
#             masked = False

#         pp_dt_shr[ty,tx] = solve_quadratic_gpu(a, b, c, masked)
#         cuda.syncthreads()
#         row_min_gpu(pp_dt_shr)
#         pp_dt[p, cuda.blockIdx.x] = pp_dt_shr[ty,0]

#     @cuda.jit
#     def get_pp_col_time_gpu4(pp_dt, pos, vel, radius, mask):#, a_gpu, b_gpu, c_gpu):
#         tx = cuda.threadIdx.x
#         ty = cuda.threadIdx.y
#         p = ty + cuda.blockIdx.y * cuda.blockDim.y
#         q = tx + cuda.blockIdx.x * cuda.blockDim.x

#         pp_dt_shr = cuda.shared.array(shape=(pp_brows, pp_bcols), dtype=nb_dtype)
        
#         pos_shr_row = cuda.shared.array(shape=(pp_brows, dim), dtype=nb_dtype)
#         vel_shr_row = cuda.shared.array(shape=(pp_brows, dim), dtype=nb_dtype)
#         radius_shr_row = cuda.shared.array(shape=pp_brows, dtype=nb_dtype)
        
#         pos_shr_col = cuda.shared.array(shape=(pp_bcols, dim), dtype=nb_dtype)
#         vel_shr_col = cuda.shared.array(shape=(pp_bcols, dim), dtype=nb_dtype)
#         radius_shr_col = cuda.shared.array(shape=pp_bcols, dtype=nb_dtype)

#         if ty < dim:
#             pos_shr_col[tx, ty] = pos[q, ty]
#             vel_shr_col[tx, ty] = vel[q, ty]
#         if ty == 0:
#             radius_shr_col[tx] = radius[q]            
#         if tx < dim:
#             pos_shr_row[ty, tx] = pos[p, tx]
#             vel_shr_row[ty, tx] = vel[p, tx]
#         if tx== 0:
#             radius_shr_row[ty] = radius[p]

#         cuda.syncthreads()
                
#         a = 0.0
#         b = 0.0
#         c = 0.0
# #         x = pos[p]
# #         v = vel[p]
# #         r = radius[p]
#         for d in range(dim):
#             dx = pos_shr_row[ty, d] - pos_shr_col[tx, d]
#             dv = vel_shr_row[ty, d] - vel_shr_col[tx, d]
#             a += dv * dv
#             b += dx * dv * 2
#             c += dx * dx
#         c -= (radius_shr_row[ty] + radius_shr_col[tx])**2

#         if ((mask[0]==p) & (mask[1]==q)):
#             masked = True
#         elif ((mask[0]==q) & (mask[1]==p)):
#             masked = True
#         else:
#             masked = False

#         pp_dt_shr[ty,tx] = solve_quadratic_gpu(a, b, c, masked)
#         cuda.syncthreads()
#         row_min_gpu(pp_dt_shr)
#         pp_dt[p, cuda.blockIdx.x] = pp_dt_shr[ty,0]

        
#     @cuda.jit
#     def get_pp_col_time_gpu2(pp_dt, pos, vel, radius, mask):#, a_gpu, b_gpu, c_gpu):
#         tx = cuda.threadIdx.x
#         ty = cuda.threadIdx.y
#         p = ty + cuda.blockIdx.y * cuda.blockDim.y
#         q = tx + cuda.blockIdx.x * cuda.blockDim.x

#         pp_dt_shr = cuda.shared.array(shape=(pp_brows, pp_bcols), dtype=nb_dtype)
#         pos_shr = cuda.shared.array(shape=(pp_bcols, dim), dtype=nb_dtype)
#         vel_shr = cuda.shared.array(shape=(pp_bcols, dim), dtype=nb_dtype)
#         radius_shr = cuda.shared.array(shape=pp_bcols, dtype=nb_dtype)

#         if ty < dim:
#             pos_shr[tx, ty] = pos[q, ty]
#             vel_shr[tx, ty] = vel[q, ty]
#         if ty == 0:
#             radius_shr[tx] = radius[q]
#         cuda.syncthreads()
                
#         a = 0.0
#         b = 0.0
#         c = 0.0
# #         x = pos[p]
# #         v = vel[p]
# #         r = radius[p]
#         for d in range(dim):
#             dx = pos[p, d] - pos_shr[tx, d]
#             dv = vel[p, d] - vel_shr[tx, d]
#             a += dv * dv
#             b += dx * dv * 2
#             c += dx * dx
#         c -= (radius[p] + radius_shr[tx])**2

#         if ((mask[0]==p) & (mask[1]==q)):
#             masked = True
#         elif ((mask[0]==q) & (mask[1]==p)):
#             masked = True
#         else:
#             masked = False

#         pp_dt_shr[ty,tx] = solve_quadratic_gpu(a, b, c, masked)
#         cuda.syncthreads()
#         row_min_gpu(pp_dt_shr)
#         pp_dt[p, cuda.blockIdx.x] = pp_dt_shr[ty,0]

        
    
#     @cuda.jit
#     def get_pp_col_time_gpu3(pp_dt, pos, vel, radius, mask):#, a_gpu, b_gpu, c_gpu):
#         tx = cuda.threadIdx.x
#         ty = cuda.threadIdx.y
#         p = ty + cuda.blockIdx.y * cuda.blockDim.y
#         q = tx + cuda.blockIdx.x * cuda.blockDim.x

#         pp_dt_shr = cuda.shared.array(shape=(pp_brows, pp_bcols), dtype=nb_dtype)
                
#         a = 0.0
#         b = 0.0
#         c = 0.0
#         for d in range(dim):
#             dx = pos[p,d] - pos[q, d]
#             dv = vel[p,d] - vel[q, d]
#             a += dv * dv
#             b += dx * dv * 2
#             c += dx * dx
#         c -= (radius[p] + radius[q])**2

#         if ((mask[0]==p) & (mask[1]==q)):
#             masked = True
#         elif ((mask[0]==q) & (mask[1]==p)):
#             masked = True
#         else:
#             masked = False
            
#         pp_dt_shr[ty,tx] = solve_quadratic_gpu(a, b, c, masked)
#         cuda.syncthreads()
#         row_min_gpu(pp_dt_shr)
#         pp_dt[p, cuda.blockIdx.x] = pp_dt_shr[ty,0]



# @cuda.jit(device=True)
# def solve_quadratic_gpu(a, b, c, mask=False):
#     tol = 1e-12
#     b *= -1
#     if abs(a) < tol:
#         if abs(b) < tol:
#             small = np.inf
#             big = np.inf
#         else:
#             small = c / b
#             big = np.inf
#     else:
#         d = b**2 - 4 * a * c
#         if d < 0:
#             small = np.inf
#             big = np.inf
#         else:
#             d = math.sqrt(d)
#             f = 2 * a
#             if b >= 0:
#                 small = (b - d) / f
#                 big  =  (b + d) / f
#             else:
#                 small = (b + d) / f
#                 big  =  (b - d) / f
#     if mask == True:
#         small = big
#         big = np.inf
#     if small < 0:
#         if big < 0:
#             return np.inf
#         else:
#             return big
#     else:
#         return small
