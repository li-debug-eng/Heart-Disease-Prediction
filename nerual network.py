
import sys
sys.path.append(r'/home/mw/input/MLS_C1W15029/')
import numpy as np
import matplotlib.pyplot as plt
from lab_utils_uni import plt_intuition, plt_stationary, plt_update_onclick, soup_bowl
plt.style.use('C:\Users\li\A最新版 吴恩达机器学习Deeplearning.ai\第一课 Supervised Machine Learning Regression and Classification\week1\work\lab_utils_uni.py')
def comepute_model_output(x,w,b):
    m = len(x)
    f_wb = np.zeros(m)
    for i in range(m):
        f_wb[i] = w * x[i]+b
    return f_wb

def compute_cost(x,y,w,b):
    m = len(x)
    f_wb = np.zeros(m)
    cost_sum = 0.0
    for i in range(m):
        f_wb[i] = w * x[i]+b
        cost = (f_wb[i] - y[i])**2
        cost_sum = cost_sum + cost
    j_wb = cost_sum/(2*m)
    return j_wb


# x_train = np.array([1.0,2.0]) #square
# y_train = np.array([300,500]) #price

# # m is the number of training examples
# print(f"x_train.shape: {x_train.shape}")
# m = x_train.shape[0]
# print(f"Number of training examples is: {m}")

# plt.scatter(x_train, y_train,marker='x', c = 'r')
# plt.title('Housing Price')
# plt.xlabel('size (1000 sqft)')
# plt.ylabel('price (in 1000s of dollars)')
# plt.show()

# w = 200
# b = 100
# tmp_f_wb = comepute_model_output(x_train,w,b)

# plt.plot(x_train, tmp_f_wb, c = 'b', label='Our prediction')
# plt.scatter(x_train, y_train,marker='x', c = 'r', label='Actual price')
# plt.title('Housing Price')
# plt.xlabel('size (1000 sqft)')
# plt.ylabel('price (in 1000s of dollars)')
# plt.legend()
# plt.show()

# w = 200 
# b = 100
# x_i =  1.2
# cost_1200sqrt = w * x_i + b
# print(f"{cost_1200sqrt:.0f} thousands of dollars")

x_train = np.array([1.0, 1.7, 2.0, 2.5, 3.0, 3.2])
y_train = np.array([250, 300, 480,  430,   630, 730,])

fig,ax,dyn_items= plt_stationary(x_train, y_train)
updater = plt_update_onclick(fig, ax, dyn_items, x_train, y_train)
soup_bowl()
plt.show()