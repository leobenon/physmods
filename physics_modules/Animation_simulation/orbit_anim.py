import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter

# --- integrate simple 2D gravity ---
G, M = 1.0, 1.0
dt = 0.01
r = np.array([1.0, 0.0], float)
v = np.array([0.0, 1.0], float)

rs = []
for _ in range(2000):
    a = -(G*M) * r / (np.linalg.norm(r)**3)
    v += a*dt
    r += v*dt
    rs.append(r.copy())
rs = np.array(rs)  # shape (N, 2)

# --- plotting/animation ---
fig, ax = plt.subplots()
ax.set_aspect('equal')
ax.set_xlim(-1.5, 1.5)
ax.set_ylim(-1.5, 1.5)
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_title("Orbit")

# Sun
ax.plot(0, 0, 'yo', ms=10)

# Planet (as a Line2D with marker-only). IMPORTANT: set_data needs sequences.
planet, = ax.plot([], [], 'o', ms=6)
trail,  = ax.plot([], [], '-', lw=1)

def init():
    planet.set_data([], [])
    trail.set_data([], [])
    return planet, trail

def update(i):
    # For a single point, wrap in lists (or 1D arrays)
    planet.set_data([rs[i, 0]], [rs[i, 1]])
    trail.set_data(rs[:i+1, 0], rs[:i+1, 1])  # these are already 1D sequences
    return planet, trail

ani = FuncAnimation(fig, update, frames=len(rs), init_func=init, blit=True)
ani.save("orbit.mp4", writer=FFMpegWriter(fps=60))
print("Wrote orbit.mp4")
