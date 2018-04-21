from sklearn import linear_model
import matplotlib.pyplot as plt
import numpy as np

def plot(model, title, test_data_x):
    test_data_y = model.predict(test_data_x)
    print(test_data_y)

    row_idx = 0
    for features in test_data_x:
        format = 'ro'
        if test_data_y[row_idx] == 1:
            format = 'bo'
        
        plt.plot(features[0], features[1], format)
        row_idx += 1

    plt.title(title)
    plt.show()

# Can't fix xor data with linear regression directly
# since xor is not linear itself, We need to choose 
# a feature map which can be done by neural network?
def regression(train_x, train_y):
    regr = linear_model.LinearRegression()
    regr.fit(train_x, train_y)
    print(regr.coef_)
    plot(regr, 'Linear regression', train_x)


if __name__ == '__main__':
    # xor data
    train_data_x = np.array([
        [0, 0], 
        [0, 1],
        [1, 0],
        [1, 1]])

    train_data_y = np.array([0, 1, 1, 0])

    regression(train_data_x, train_data_y)