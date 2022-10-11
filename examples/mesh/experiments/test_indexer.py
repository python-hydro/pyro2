# some simple tests of the ArrayIndexer


import numpy as np

import mesh.patch as patch

myg = patch.Grid2d(8, 8, ng=2)
a = myg.scratch_array()

print(id(a))
print(a.flags.owndata)

b = a.v()
print(id(b))
print(b.flags.owndata)
print(b.shape)

s = b.shape
b[:, :] = np.arange(s[0]*s[1]).reshape(s)

print(b)
print(" ")

print(a)

print(" ")

c = a.ip(1)
print(id(c))
print(c.flags.owndata)

print(c)


print(" ")

d = a.v(s=2)
print(d)
d[:, :] = 0.0

print(" ")

print(a)
