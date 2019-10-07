import numpy as np
import matplotlib.pyplot as plt

file_name = 'raw bloodpressure data.txt'


def read_file():
    with open(file_name) as f:
        # w, h = [int(x) for x in next(f).split()] # read first line

        array = []
        X = []
        y = []

        for line in f:  # read rest of lines
            array.append([int(x) for x in line.split()])

        for item in array:
            X.append((1, item[2]))
            y.append(item[3])

        X = np.array(X)
        y = np.array(y)

    return X, y


X, y = read_file()
shape_X = X.shape

# print("Shape of X : ", X.shape)
# print("Shape of y : ", y.shape)
# print("\n\n")

w = np.dot(np.linalg.inv(np.dot(X.T, X)), np.dot(X.T, y))
print("Optimal Weight w: : ", w)
print("Shape of w : ", w.shape)
print("\n\n")

xx = np.linspace(0, 80, 800)
yy = np.array(w[0] + w[1] * xx)


prediction = np.dot(X, w.T)
# prediction = np.dot(w.T, X)
print("Shape of prediction : ", prediction.shape)
print("Prediction : \n", prediction)

# Plot data, regression line
plt.figure(1)
plt.plot(xx, yy.T, color='#C70039')
plt.scatter(X[:,1], y[:,], color="#1B4F72")
plt.xlabel("Age")
plt.ylabel("Blood Pressure")
plt.show()
save_path = 'Regression_figure'
plt.savefig(save_path, dpi=300)
