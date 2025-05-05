import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['font.size'] = 14

mu = 3.003489614915e-1  # mu for Sun-Earth = 3.003489614915e-6
r1 = np.array([-mu, 0]) 
r2 = np.array([1 - mu, 0]) 
x = np.linspace(-1.5, 1.5, 400)
y = np.linspace(-1.0, 1.0, 300)
X, Y = np.meshgrid(x, y)

def effective_potential(X, Y, mu):
    r1_dist = np.sqrt((X + mu)**2 + Y**2)
    r2_dist = np.sqrt((X - (1 - mu))**2 + Y**2)
    return - (1 - mu) / r1_dist - mu / r2_dist - 0.5 * (X**2 + Y**2)

Z = effective_potential(X, Y, mu)

def f(x):
    r1 = x + mu
    r2 = x - 1 + mu
    return x - (1 - mu)*r1/abs(r1)**3 - mu*r2/abs(r2)**3

def df(x):
    r1 = x + mu
    r2 = x - 1 + mu
    return 1 - (1 - mu)*(1/abs(r1)**3 - 3*r1**2/abs(r1)**5) - mu*(1/abs(r2)**3 - 3*r2**2/abs(r2)**5)

def newton_raphson(f, df, x0, tol=1e-10, max_iter=100):
    x = x0
    for _ in range(max_iter):
        fx = f(x)
        dfx = df(x)
        if abs(fx) < tol:
            return x
        x -= fx / dfx
    raise ValueError("Root not found")

x_L1 = newton_raphson(f, df, 0.5)
x_L2 = newton_raphson(f, df, 1)
x_L3 = newton_raphson(f, df, -1.0)

L1 = np.array([x_L1, 0])
L2 = np.array([x_L2, 0])
L3 = np.array([x_L3, 0])
L4 = np.array([0.5 - mu, np.sqrt(3)/2])
L5 = np.array([0.5 - mu, -np.sqrt(3)/2])

plt.figure(figsize=(10, 7))
contours = plt.contour(X, Y, Z, levels=np.linspace(-3.0, -1.2, 40))
plt.clabel(contours, inline=1, fontsize=8)

plt.plot(r1[0], r1[1], 'o', label='Mass 1')
plt.plot(r2[0], r2[1], 'o', label='Mass 2')

for point, name in zip([L1, L2, L3, L4, L5], ['L1','L2','L3','L4','L5']):
    plt.plot(point[0], point[1], 'x', markersize=10, label=name)

plt.xlabel('x (normalized)')
plt.ylabel('y (normalized)')
plt.title(r'Effective Potential Contours and Lagrange Points ($\mu = 3.00 \times 10^{-1}$)')
plt.legend()
plt.grid(True)
plt.axis('equal')
plt.show()
