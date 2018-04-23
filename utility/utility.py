import matplotlib.pyplot as plt
import numpy as np

def predict_and_plot(model, x):
    pred_y = model.predict(x)
    plt.plot(pred_y)