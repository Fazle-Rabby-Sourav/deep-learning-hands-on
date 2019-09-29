# alpha : a, unit vector : u
# au define a line of point that may be obtained by varrying the value of a (Alpha)
# Derive an expression for the point y that lies on the line that is as close as possible to an arbitary point x

import numpy as np

# values of a specific given point, x
x = np.array([[1.5], [2.0]])
print("x : \n", x)

# Given unit vector, u
u = np.array([[.866], [.5]])
print("u : \n", u)


# we need to find optimal alpha a to compute the desired point, y
alpha = np.dot(x.T, u)
print("Alpha : ", alpha)

y = alpha*u
print("Point y on the line, which is nearest to x: \n", y)