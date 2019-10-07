import numpy as np
import matplotlib.pyplot as plt

# covar matrix must be positive definite, or
# symmetric with an inverse for Gaussian distribution
covarMat = np.array([[1.0, 0.0], [0.0, 6.0]])  # 2x2 array
covarInv = np.linalg.inv(covarMat)


def gaussValue(x, covariance_inv):  # x is 2-element column vec
    return np.exp(-0.5 * np.dot(np.dot(x.T,
                                       covariance_inv),
                                x))


def generateValues(pltRange, _mean, covarianceInv):
    i = 0
    for x in pltRange:
        j = 0
        for y in pltRange:
            pt1 = np.array([[x], [y]])  # column vector. 2x1 array.
            z1[j, i] = gaussValue(pt1-_mean, covarianceInv)
            j = j + 1
        i = i + 1

    return z1


# create a range from -6 to 6 in 0.2 increments. Length is 61.
pltRange = np.linspace(-6, 6, num=61, endpoint=True)

# array to hold the values for a gaussian function
# index z arrays by [i,j)
# indices are from 0 to 60
z1 = np.zeros((61, 61))
_mean = np.array([[0.0], [0.0]])
z1 = generateValues(pltRange, _mean, covarInv)

plt.figure()
CS = plt.contour(pltRange, pltRange, z1)
plt.grid()

# Genrating 100 data points with numpy corresponding function with mean (0,0)
data_points = np.random.multivariate_normal(mean=[0, 0], cov=covarMat, size=100)
x, y = data_points.T
plt.plot(x, y, 'bo')
plt.show()

# -------------------------------------- Step : 3 -------------------------------------- #
# choose eigenvecs compatible # with an ellipse oriented 30 degrees to the x-axis
theta = -0.523599   # equivalent to -30 degree
c, s = np.cos(theta), np.sin(theta)

rotationMat = np.array([[c, -s], [s, c]])
invRotationMat = np.linalg.inv(rotationMat)

# Eigen Decomposition Foumula to find new covariance matrix
newCoverMat = np.dot(rotationMat, np.dot(covarMat, invRotationMat))
newCoverInv = -newCoverMat

# -------------------------------------- Step : 4 -------------------------------------- #
# Generate values
_mean = np.array([[1.0], [2.0]])
z1 = generateValues(pltRange, _mean, newCoverInv)


# -------------------------------------- Step : 5 -------------------------------------- #
plt.figure()
CS = plt.contour(pltRange, pltRange, z1)
plt.grid()
plt.show()

# -------------------------------------- Step : 6 -------------------------------------- #
# Genrating 100 data points with numpy corresponding function with mean (1, 2) and
# with new covariance matrix
data_points = np.random.multivariate_normal(mean=[1,2], cov=newCoverMat, size=200)
x, y = data_points.T

# -------------------------------------- Step : 7 -------------------------------------- #
# only points
plt.figure()
plt.grid()
plt.plot(x, y, 'bo')
plt.show()

# contour with geenrated points for mean and rotated covariance matrix
plt.figure()
CS = plt.contour(pltRange, pltRange, z1)
plt.grid()
plt.plot(x, y, 'bo')
plt.show()