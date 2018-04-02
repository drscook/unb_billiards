num_walls = len(wall)
num_part = part.num
pp_bcols = min(num_part, sqrt_threads_per_block_max)
pp_brows = pp_bcols
pp_gcols = int(np.ceil(num_part / pp_bcols))
pp_grows = pp_gcols
pp_block_shape = (pp_bcols, pp_brows)
pp_grid_shape = (pp_gcols, pp_grows)
assert pp_block_shape[1] * pp_grid_shape[1] >= num_part
assert pp_block_shape[0] == pp_block_shape[1]
assert pp_grid_shape[0] == pp_grid_shape[1]

pw_bcols = num_walls
pw_brows = int(np.floor(threads_per_block_max / pw_bcols))
pw_gcols = 1
pw_grows = int(np.ceil(num_part / pw_brows))
pw_block_shape = (pw_bcols, pw_brows)
pw_grid_shape = (pw_gcols, pw_grows)
assert pw_block_shape[1] * pw_grid_shape[1] >= num_part
assert pw_block_shape[0] * pw_grid_shape[0] >= num_walls