import numpy as np
import matplotlib.pyplot as plt

# arr=np.array([[1,2,3,4],[6,9,3,4],[7,8,9,0]])
# a1=arr[:,0:2]
# print(a1)
# print(len(arr))

'''fig = plt.figure()
ax = plt.axes(projection="3d")

z_line = np.linspace(0, 15, 1000)
x_line = np.cos(z_line)
y_line = np.sin(z_line)
#ax.plot3D(x_line, y_line, z_line, 'gray')

z_points = 15 * np.random.random(100)
x_points = np.cos(z_points) + 0.1 * np.random.randn(100)
y_points = np.sin(z_points) + 0.1 * np.random.randn(100)
ax.scatter3D(x_points, y_points, z_points, c=z_points, cmap='hsv');

plt.show()'''

fig = plt.figure(num=1, clear=True)
ax = fig.add_subplot(1, 1, 1, projection='3d')

(x, y) = np.meshgrid(np.arange(-2, 2.1, 1), np.arange(-1, 1.1, .25))
z = x + y

ax.plot_surface(x, y, z)
ax.set(xlabel='x', ylabel='y', zlabel='z', title='z = x + y')

fig.tight_layout()
plt.show()