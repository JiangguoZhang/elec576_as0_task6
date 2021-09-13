import numpy as np
import scipy.linalg as linalg
a = np.array([[1. ,2. ,3.], [4. ,5. ,6.]])
np.ndim(a)
a.ndim
np.size(a)
a.size
np.shape(a)
a.shape
a.shape[1]
a
b = np.array([[7. ,8.], [9. ,10.]])
c = np.copy([[11. ,12. ,13.], [14. ,15. ,16.]])
d = np.array([[17. ,18.], [19. ,20.]])
np.block([[a, b], [c, d]])
a[-1]
a = np.array([[1. ,2. ,3. ,4. ,5. ,6. ,7. ,8. ,9., 10.],
              [11. ,12. ,13. ,14. ,15. ,16. ,17. ,18. ,19., 20.],
              [21. ,22. ,23. ,24. ,25. ,26. ,27. ,28. ,29., 30.],
              [31. ,32. ,33. ,34. ,35. ,36. ,37. ,38. ,39., 40.],
              [41. ,42. ,43. ,44. ,45. ,46. ,47. ,48. ,49., 50.],
              [51. ,52. ,53. ,54. ,55. ,56. ,57. ,58. ,59., 60.],
              [61. ,62. ,63. ,64. ,65. ,66. ,67. ,68. ,69., 70.],
              [71. ,72. ,73. ,74. ,75. ,76. ,77. ,78. ,79., 80.],
              [81. ,82. ,83. ,84. ,85. ,86. ,87. ,88. ,89., 90.]])
a[1, 4]
a[1]
a[1, :]
a[0:5]
a[:5]
a[0:5, :]
a[-5:]
a[0:3, 4:9]
a[np.ix_([1, 3, 4], [0, 2])]
a[2:21:2,:]
a[ ::2,:]
a[::-1,:]
a[np.r_[:len(a),0]]
a.transpose()
a.T
a = np.array([[1. ,2. ,3.+2j], [4. ,5.-3j ,6.]])
a.conj().transpose()
a.conj().T
a = a.T
b = np.array([[7. ,8.], [9. ,10.]])
a @ b
b = np.array([[7. ,8. ,9.], [10. ,11. ,12]]).T
a * b
a/b
a**3
a = np.array([[.1 ,.4 ,.7 ,1.],
              [.2 ,.5 ,.8 ,1.1],
              [.3 ,.6 ,.9 ,.0]])
(a > 0.5)
np.nonzero(a > 0.5)
v = np.array([.1 ,.4 ,.7 ,1.])
a[:,np.nonzero(v > 0.5)[0]]
a[:, v.T > 0.5]
a[a < 0.5]=0
a
a * (a > 0.5)
a[:] = 3
a
x = np.array([[1. ,2. ,3.], [4. ,5. ,6.]])
y = x.copy()
y
y = x[1, :].copy()
y
y = x.flatten()
y
np.arange(1., 11.)
np.r_[1.:11.]
np.r_[1:10:10j]
np.arange(10.)
np.r_[:10.]
np.r_[:9:10j]
np.arange(1.,11.)[:, np.newaxis]
np.zeros((3, 4))
np.zeros((3, 4, 5))
np.ones((3, 4))
np.eye(3)
np.diag(a)
np.diag(v, 0)
from numpy.random import default_rng
rng = default_rng(42)
rng.random([3, 4])
np.linspace(1,3,4)
np.mgrid[0:9.,0:6.]
np.meshgrid(np.r_[0:9.],np.r_[0:6.])
np.ogrid[0:9.,0:6.]
np.ix_(np.r_[0:9.],np.r_[0:6.])
np.meshgrid([1,2,4],[2,4,5])
np.ix_([1,2,4],[2,4,5])
np.tile(a, (2, 3))
a = np.array([[.1 ,.4 ,.7 ,1.],
              [.2 ,.5 ,.8 ,1.1],
              [.3 ,.6 ,.9 ,.0]])
b = np.array([[.11 ,.32 ,.73 ,1.],
              [.24 ,.45 ,.86 ,1.1],
              [.37 ,.58 ,.99 ,.0]])
np.concatenate((a,b),1)
np.hstack((a,b))
np.column_stack((a,b))
np.c_[a,b]
np.concatenate((a,b))
np.vstack((a,b))
np.r_[a,b]
a.max()
np.nanmax(a)
a.max(0)
a.max(1)
np.maximum(a, b)
np.sqrt(v @ v)
np.linalg.norm(v)
a = np.array([[True, False, False], [False, True, True]])
b = np.array([[True, True, False], [False, True, False]])
np.logical_and(a,b)
np.logical_or(a,b)
a & b
a | b
a = rng.random([5, 5])
a
b = linalg.inv(a)
b
a @ b
b = linalg.pinv(a)
b
np.linalg.matrix_rank(a)
a = rng.random([3, 3])
a
b = rng.random(3)
b
linalg.solve(a, b)
a = rng.random([2, 3])
a
b = rng.random(2)
b
linalg.lstsq(a, b)
a = rng.random([3,3])
a
b = rng.random([3,3])
b
linalg.lstsq(a.T, b.T)
U, S, Vh = linalg.svd(a)
V = Vh.T
U
S
V
a = np.array([[1,-2j],[2j,5]])
c = linalg.cholesky(a)
c
a = rng.random([5,5])
a
D,V = linalg.eig(a)
D
V
b = rng.random([5,5])
b
D,V = linalg.eig(a, b)
D
V
from scipy.sparse.linalg import eigs
D,V = eigs(a, k = 3)
D
V
Q,R = linalg.qr(a)
Q
R
P,L,U = linalg.lu(a)
P
L
U
from scipy.sparse.linalg import cg
b = rng.random(5)
b
cg(a, b)
np.fft.fft(a)
np.fft.ifft(a)
np.sort(a)
a.sort(axis=0)
a
np.sort(a, axis = 1)
a.sort(axis = 1)
a
I = np.argsort(a[:, 0])
I
b = a[I,:]
b
Z = rng.random([6, 6])
Z
x = linalg.lstsq(Z, y)
x
from scipy.signal import resample
x = np.random.randint(0, 255, size=[6, 6])
x
resample(x, int(np.ceil(len(x)/2)))
a = np.array([[1,2,3,3,2,1,4,5,6,6,5,2]])
np.unique(a)
a.squeeze()
a
