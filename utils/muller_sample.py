import numpy as np
n_sample = 5000
coor = np.random.randn(n_sample, 3)
norm = np.linalg.norm(coor, axis=1).reshape(n_sample,1).repeat(3).reshape(n_sample,3)
coor = coor / norm
np.savetxt("../query_vector.xyz", coor)
