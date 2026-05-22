import argparse
from pathlib import Path

import numpy as np 
import matplotlib.pylab as plt 
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection
from matplotlib.lines import Line2D
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
# file paths

geom_file = "..."
prop_file = "..."
pert_file = "..."

def parse_args():
    parser = argparse.ArgumentParser(description="Animate D-SPOSE output files")
    parser.add_argument("--basepath", required=True, help="Absolute path of the data folder")
    parser.add_argument("--geom", required=True, help="Relative path to geometry file")
    parser.add_argument("--prop", required=True, help="Relative path to propagation file")
    parser.add_argument("--pert", required=True, help="Relative path to perturbations file")

    parser.add_argument("--outdir", default="animation_output", help="Output directory")
    parser.add_argument("--prefix", default="satellite_animation", help="Output filename prefix")

    parser.add_argument("--save", action="store_true", help="Save animations")
    parser.add_argument("--show", action="store_true", help="Show plots interactively")

    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--dpi", type=int, default=150)

    parser.add_argument("--orbit-trail-len", type=int, default=1000)
    parser.add_argument("--axis-trail-len", type=int, default=300)
    parser.add_argument("--scale", type=float, default=8.0)

    return parser.parse_args()

args = parse_args()
base_path = Path(args.basepath).expanduser()

geom_file = base_path / args.geom
prop_file = base_path / args.prop
pert_file = base_path / args.pert

outdir = base_path / args.outdir
outdir.mkdir(parents=True, exist_ok=True)


# loaders
def load_geometry(path):
    data = np.loadtxt(path, comments="#")
    v1 = data[:,3:6]
    v2 = data[:,6:9]
    v3 = data[:,9:12]
    tris = np.stack([v1,v2,v3], axis=1)
    return tris

def load_propagation(path):
    data = np.loadtxt(path, comments="#")
    t = data[:,0] # time in s 
    v = data[:,1:4] # TEME velocity in m/s
    r = data[:,4:7] # TEME position in m
    w = data[:,7:10] # angular velocity in rad/s
    q = data[:,10:14] # quaternions 
    return t,r,w,q,v

def load_perturbations(path):
    data = np.loadtxt(path,comments="#")
    t = data[:,0]
    T_gg = data[:,10:13]
    T_srp = data[:,31:34]
    return t, T_gg, T_srp


# math

def quat_to_dcm(q):  # calculates C matrix 
    q = np.asarray(q, dtype=float)
    q = q / np.linalg.norm(q)
    q0, q1, q2, q3 = q

    return np.array([
        [1 - 2*(q2*q2 + q3*q3),     2*(q1*q2 - q0*q3),     2*(q1*q3 + q0*q2)],
        [    2*(q1*q2 + q0*q3), 1 - 2*(q1*q1 + q3*q3),     2*(q2*q3 - q0*q1)],
        [    2*(q1*q3 - q0*q2),     2*(q2*q3 + q0*q1), 1 - 2*(q1*q1 + q2*q2)]
    ])

def quat_to_euler321(q):
    q = np.asarray(q, dtype=float)
    q = q / np.linalg.norm(q)

    q0, q1, q2, q3 = q

    # roll (phi)
    phi = np.arctan2(
        2*(q0*q1 + q2*q3),
        1 - 2*(q1*q1 + q2*q2)
    )

    # pitch (theta)
    theta = np.arcsin(
        2*(q0*q2 - q3*q1)
    )

    # yaw (psi)
    psi = np.arctan2(
        2*(q0*q3 + q1*q2),
        1 - 2*(q2*q2 + q3*q3)
    )

    return np.degrees([phi, theta, psi])

def read_initial_params_from_header(path):
    with open(path, "r") as f:
        for line in f:
            if line.startswith("# SPACECRAFT PARAMETERS:"):
                vals = np.array(line.split(":")[1].split(), dtype=float)
                break
        else:
            raise ValueError("Could not find SPACECRAFT PARAMETERS in file header.")


    euler0_deg = vals[11:14]   # [deg]
    frame_flag = int(vals[14])

    if frame_flag == 1:
        frame_name = "IRF"
    elif frame_flag == 2:
        frame_name = "ORF"
    else:
        frame_name = f"unknown frame flag {frame_flag}"

    return euler0_deg, frame_flag, frame_name

def rotate_tris(tris, R):
    return tris @ R.T

def set_equal_axes(ax, pts):
    mins = pts.min(axis=0)
    maxs = pts.max(axis=0)
    c = 0.5 * (mins + maxs)
    radius = 0.5 * np.max(maxs - mins)
    ax.set_xlim(c[0] - radius, c[0] + radius)
    ax.set_ylim(c[1] - radius, c[1] + radius)
    ax.set_zlim(c[2] - radius, c[2] + radius)

def draw_earth(ax, radius_km=6378.0, n=36):
    u = np.linspace(0, 2*np.pi, n)
    v = np.linspace(0, np.pi, n)
    x = radius_km * np.outer(np.cos(u), np.sin(v))
    y = radius_km * np.outer(np.sin(u), np.sin(v))
    z = radius_km * np.outer(np.ones_like(u), np.cos(v))
    ax.plot_surface(x, y, z, alpha=0.2, linewidth=0)


# load data 1 

tris_body = load_geometry(geom_file)
t, r_m, w_body, q, v = load_propagation(prop_file)
t2, T_gg, T_srp = load_perturbations(pert_file)


# --- Draw eclitic and intantaneous orbit plane ----

def plane_disk_from_normal(n, radius=1.0, N=100):
    n = np.asarray(n, dtype=float)
    n = n / np.linalg.norm(n)

    a = np.array([1.0, 0.0, 0.0])
    if abs(np.dot(a, n)) > 0.9:
        a = np.array([0.0, 1.0, 0.0])

    u = np.cross(n, a)
    u = u / np.linalg.norm(u)
    v = np.cross(n, u)

    theta = np.linspace(0, 2*np.pi, N)

    circle = radius * (
        np.outer(np.cos(theta), u)
        + np.outer(np.sin(theta), v)
    )

    return circle

# ---- Turn velocity and position into osculating parameters ----

def state2coe(R, V):
    """
    Convert position and velocity vectors to classical orbital elements.

    Parameters:
        R : array_like (3,) - position vector (m)
        V : array_like (3,) - velocity vector (m/s)

    Returns:
        sma  : semi-major axis (m)
        ecc  : eccentricity
        inc  : inclination (rad)
        raan : right ascension of ascending node (rad)
        aop  : argument of perigee (rad)
        ta   : true anomaly (rad)
    """

    eps = 1e-10
    mu = 3.986004418e14  # Earth's gravitational parameter (m^3/s^2)

    R = np.array(R, dtype=float)
    V = np.array(V, dtype=float)

    r = np.linalg.norm(R)
    v = np.linalg.norm(V)

    vr = np.dot(R, V) / r

    # Angular momentum
    H = np.cross(R, V)
    h = np.linalg.norm(H)

    # Inclination
    inc = np.arccos(H[2] / h)

    # Node vector
    z = np.array([0.0, 0.0, 1.0])
    N = np.cross(z, H)
    no = np.linalg.norm(N)

    # RAAN
    if no != 0:
        raan = np.arccos(N[0] / no)
        if N[1] < 0:
            raan = 2 * np.pi - raan
    else:
        raan = 0.0

    # Eccentricity vector
    E = (1 / mu) * ((v**2 - mu / r) * R - r * vr * V)
    ecc = np.linalg.norm(E)

    # Semi-major axis
    sma = h**2 / mu / (1 - ecc**2)

    # Argument of perigee
    if no != 0:
        if ecc > eps:
            aop = np.arccos(np.dot(N, E) / (no * ecc))
            if E[2] < 0:
                aop = 2 * np.pi - aop
        else:
            aop = 0.0
    else:
        aop = 0.0

    # True anomaly
    if ecc > eps:
        ta = np.arccos(np.dot(E, R) / (ecc * r))
        if vr < 0:
            ta = 2 * np.pi - ta
    else:
        cp = np.cross(N, R)
        if cp[2] >= 0:
            ta = np.arccos(np.dot(N, R) / (no * r))
        else:
            ta = 2 * np.pi - np.arccos(np.dot(N, R) / (no * r))

    return sma, ecc, inc, raan, aop, ta
sma ,ecc , inc , raan , aop , ta = [],[],[],[],[],[]
for i in range(len(t)):
    sma_i, ecc_i, inc_i, raan_i, aop_i, ta_i = state2coe(r_m[i,:],v[i,:])
    sma.append(sma_i)
    ecc.append(ecc_i)
    inc.append(inc_i)
    raan.append(raan_i)
    aop.append(aop_i)
    ta.append(ta_i)
sma = np.array(sma)
ecc = np.array(ecc)
inc = np.array(inc)
raan = np.array(raan)
aop = np.array(aop)
ta = np.array(ta)

def orbit_normal_from_elements(inc, raan):
    return np.array([
        np.sin(inc) * np.sin(raan),
       -np.sin(inc) * np.cos(raan),
        np.cos(inc)
    ])
# ----- Adding figure 3 for animating the orbital and ecliptic plane -----
fig3 = plt.figure(figsize=(8, 8))
ax_planes = fig3.add_subplot(111, projection="3d")

ax_planes.set_title("Ecliptic plane and osculating orbit plane")
ax_planes.set_xlabel("X")
ax_planes.set_ylabel("Y")
ax_planes.set_zlabel("Z")

plane_radius = 1.0

# fixed ecliptic normal
eps = np.deg2rad(23.43928)
n_ecl = np.array([0.0, -np.sin(eps), np.cos(eps)])

ecl_circle = plane_disk_from_normal(n_ecl, radius=plane_radius)
ecl_line, = ax_planes.plot(
    ecl_circle[:,0], ecl_circle[:,1], ecl_circle[:,2],
    color="orange", lw=2, label="Ecliptic plane"
)

orbit_line, = ax_planes.plot([], [], [], color="blue", lw=2, label="Orbit plane")

ecl_normal = ax_planes.quiver(
    0, 0, 0,
    n_ecl[0], n_ecl[1], n_ecl[2],
    color="orange",
    linewidth=2
)

orbit_normal = None

ax_planes.legend()
ax_planes.set_xlim(-1.1, 1.1)
ax_planes.set_ylim(-1.1, 1.1)
ax_planes.set_zlim(-1.1, 1.1)
ax_planes.set_box_aspect([1,1,1])
ax_planes.set_autoscale_on(False)

# -- Draw plane fills 
ecl_fill = Poly3DCollection([], alpha=0.18, color="orange")
orb_fill = Poly3DCollection([], alpha=0.18, color="blue")

ax_planes.add_collection3d(ecl_fill)
ax_planes.add_collection3d(orb_fill)

node_line, = ax_planes.plot([], [], [], color="green", lw=3, label="Line of nodes / RAAN")
ref_line, = ax_planes.plot([-1, 1], [0, 0], [0, 0], "k--", lw=1, label="Reference X-axis")

#  ----- Ecliptic fill -----
ecl_circle = plane_disk_from_normal(n_ecl, radius=plane_radius)

ecl_line.set_data(ecl_circle[:,0], ecl_circle[:,1])
ecl_line.set_3d_properties(ecl_circle[:,2])

ecl_fill.set_verts([ecl_circle])


#Extract initial parameters

t_init, r_m_init, w_body_init, q_init, v_init = t[0], r_m[0]/1000, np.rad2deg(w_body[0]), q[0], v[0]/1000 # [s], [km], [°/s], [], [km/s]

euler0_header, frame_flag, frame_name = read_initial_params_from_header(prop_file)
phi_init = euler0_header[0] # roll
theta_init = euler0_header[1] # pitch
psi_init = euler0_header[2] # yaw

if not np.allclose(t, t2):
    raise ValueError("Propagation and perturbation times do not match.")

r_km = r_m / 1000.0

# body-axis history for precession/nutation traces
e1 = np.array([1.0, 0.0, 0.0])
e2 = np.array([0.0, 1.0, 0.0])
e3 = np.array([0.0, 0.0, 1.0])

axis1_hist = np.zeros((len(t), 3))
axis2_hist = np.zeros((len(t), 3))
axis3_hist = np.zeros((len(t), 3))

for i in range(len(t)):
    R = quat_to_dcm(q[i])
    axis1_hist[i] = R @ e1
    axis2_hist[i] = R @ e2
    axis3_hist[i] = R @ e3

# scale torque arrows so they are visible
gg_mag_max = np.max(np.linalg.norm(T_gg, axis=1))
srp_mag_max = np.max(np.linalg.norm(T_srp, axis=1))

gg_scale = 8.0 / gg_mag_max if gg_mag_max > 0 else 1.0
srp_scale = 8.0 / srp_mag_max if srp_mag_max > 0 else 1.0




# ----------------------------
# figure
# ----------------------------
fig = plt.figure(figsize=(16, 9))
ax_orbit = fig.add_subplot(1, 2, 1, projection="3d")
ax_body  = fig.add_subplot(1, 2, 2, projection="3d")
fig2 = plt.figure(figsize=(15, 9))
ax_body_x  = fig2.add_subplot(1, 3, 1, projection="3d")
ax_body_y  = fig2.add_subplot(1, 3, 2, projection="3d")
ax_body_z  = fig2.add_subplot(1, 3, 3, projection="3d")

# orbit panel

# ----------------------------
# vanishing trail for the orbit 
# ----------------------------

#orbit_trail_len = 5   # number of previous points to show
#orbit_trail, = ax_orbit.plot([], [], [], lw=1)

# ----------------------------
# fading alpha trail for the orbit 
# ----------------------------
orbit_trail_len = args.orbit_trail_len
orbit_segments = Line3DCollection([np.array([[0,0,0],[0,0,0]])], linewidths=1)
ax_orbit.add_collection3d(orbit_segments)


# draw the full orbit over the whole propagation
#ax_orbit.plot(r_km[:, 0], r_km[:, 1], r_km[:, 2], lw=1)


draw_earth(ax_orbit)
ax_orbit.set_title("Orbit")
ax_orbit.set_xlabel("X [km]")
ax_orbit.set_ylabel("Y [km]")
ax_orbit.set_zlabel("Z [km]")
set_equal_axes(ax_orbit, r_km)
ax_orbit.set_autoscale_on(False)

# body panel
legend_elements = [
    Line2D([0],[0], color='r', lw=2, label='Body X'),
    Line2D([0],[0], color='g', lw=2, label='Body Y'),
    Line2D([0],[0], color='b', lw=2, label='Body Z'),
]


mesh = Poly3DCollection(tris_body, alpha=0.7, edgecolors="k", linewidths=0.5)
mesh_x = Poly3DCollection(tris_body, alpha=0.7, edgecolors="k", linewidths=0.5)
mesh_y = Poly3DCollection(tris_body, alpha=0.7, edgecolors="k", linewidths=0.5)
mesh_z = Poly3DCollection(tris_body, alpha=0.7, edgecolors="k", linewidths=0.5)
# body with in components

ax_body_x.add_collection3d(mesh_x)
ax_body_x.set_title("Attitude ")
ax_body_x.legend(handles=[legend_elements[0]], loc='upper right')
ax_body_x.set_xlabel("X")
ax_body_x.set_ylabel("Y")
ax_body_x.set_zlabel("Z")
set_equal_axes(ax_body_x, tris_body.reshape(-1, 3))
ax_body_x.set_autoscale_on(False)

ax_body_y.add_collection3d(mesh_y)
ax_body_y.set_title("Attitude ")
ax_body_y.legend(handles=[legend_elements[1]], loc='upper right')
ax_body_y.set_xlabel("X")
ax_body_y.set_ylabel("Y")
ax_body_y.set_zlabel("Z")
set_equal_axes(ax_body_y, tris_body.reshape(-1, 3))
ax_body_y.set_autoscale_on(False)

ax_body_z.add_collection3d(mesh_z)
ax_body_z.set_title("Attitude ")
ax_body_z.legend(handles=[legend_elements[2]], loc='upper right')
ax_body_z.set_xlabel("X")
ax_body_z.set_ylabel("Y")
ax_body_z.set_zlabel("Z")
set_equal_axes(ax_body_z, tris_body.reshape(-1, 3))
ax_body_z.set_autoscale_on(False)



# body with all directions
ax_body.add_collection3d(mesh)
ax_body.set_title("Attitude")
ax_body.legend(handles=legend_elements, loc='upper right')
ax_body.set_xlabel("X")
ax_body.set_ylabel("Y")
ax_body.set_zlabel("Z")
set_equal_axes(ax_body, tris_body.reshape(-1, 3))
ax_body.set_autoscale_on(False)

# orbit marker
sat_point, = ax_orbit.plot([], [], [], marker="o", linestyle="None")

# small attitude triad in orbit view
triad_lines = [ax_orbit.plot([], [], [], lw=2)[0] for _ in range(3)]
triad_scale_km = 1000.0

# body-axis trails
trail_len = 80
trail_x, = ax_body.plot([], [], [], lw=1)
trail_y, = ax_body.plot([], [], [], lw=1)
trail_z, = ax_body.plot([], [], [], lw=1)

# body-axis trails x direction 
trail_len_x = 80
trail_xx, = ax_body_x.plot([], [], [], lw=1)
trail_yx, = ax_body_x.plot([], [], [], lw=1)
trail_zx, = ax_body_x.plot([], [], [], lw=1)

# body-axis trails x direction 
trail_len_y = 80
trail_xy, = ax_body_y.plot([], [], [], lw=1)
trail_yy, = ax_body_y.plot([], [], [], lw=1)
trail_zy, = ax_body_y.plot([], [], [], lw=1)

# body-axis trails x direction 
trail_len_z = 80
trail_xz, = ax_body_z.plot([], [], [], lw=1)
trail_yz, = ax_body_z.plot([], [], [], lw=1)
trail_zz, = ax_body_z.plot([], [], [], lw=1)

# torque arrows in body view
#gg_quiv = None
#srp_quiv = None

# create texts on the plots
vel_text = ax_orbit.text2D(
    0.02, 0.85, '', 
    transform=ax_orbit.transAxes
)
torque_text = ax_body.text2D(0.02, 0.85, '', transform=ax_body.transAxes)
torque_text_x = ax_body_x.text2D(0.02, 0.85, '', transform=ax_body_x.transAxes)
torque_text_y = ax_body_y.text2D(0.02, 0.85, '', transform=ax_body_y.transAxes)
torque_text_z = ax_body_z.text2D(0.02, 0.85, '', transform=ax_body_z.transAxes)
#initialize quivers for body fixed axes
body_axes = [
    ax_body.quiver(0,0,0, 1,0,0, linewidth=2),
    ax_body.quiver(0,0,0, 0,1,0, linewidth=2),
    ax_body.quiver(0,0,0, 0,0,1, linewidth=2)
]

body_axes_x = [
    ax_body_x.quiver(0,0,0, 1,0,0, linewidth=2),
]
body_axes_y = [
    ax_body_y.quiver(0,0,0, 0,1,0, linewidth=2),
]
body_axes_z = [
    ax_body_z.quiver(0,0,0, 0,0,1, linewidth=2),
]

axis_trail_len = args.axis_trail_len

#x_tip_trail = Line3DCollection([np.array([[0,0,0],[0,0,0]])], linewidths=2)
#y_tip_trail = Line3DCollection([np.array([[0,0,0],[0,0,0]])], linewidths=2)
#z_tip_trail = Line3DCollection([np.array([[0,0,0],[0,0,0]])], linewidths=2)

#ax_body_x.add_collection3d(x_tip_trail)
#ax_body_y.add_collection3d(y_tip_trail)
#ax_body_z.add_collection3d(z_tip_trail)

# optional: also show all three trails on the main attitude plot
#x_tip_trail_all = Line3DCollection([np.array([[0,0,0],[0,0,0]])], linewidths=2)
#y_tip_trail_all = Line3DCollection([np.array([[0,0,0],[0,0,0]])], linewidths=2)
#z_tip_trail_all = Line3DCollection([np.array([[0,0,0],[0,0,0]])], linewidths=2)

#ax_body.add_collection3d(x_tip_trail_all)
#ax_body.add_collection3d(y_tip_trail_all)
#ax_body.add_collection3d(z_tip_trail_all)



def update_axis_trail(collection, axis_hist, i, scale, rgb, trail_len=300):
    j0 = max(0, i - trail_len)
    pts = axis_hist[j0:i+1] * scale

    if len(pts) >= 2:
        segments = np.stack([pts[:-1], pts[1:]], axis=1)

        alphas = np.linspace(0.05, 1.0, len(segments))
        colors = np.zeros((len(segments), 4))
        colors[:, :3] = rgb
        colors[:, 3] = alphas

        collection.set_segments(segments)
        collection.set_color(colors)

x_tip_scatter = ax_body_x.scatter([], [], [], color="r", s=8, alpha=0.8)
y_tip_scatter = ax_body_y.scatter([], [], [], color="g", s=8, alpha=0.8)
z_tip_scatter = ax_body_z.scatter([], [], [], color="b", s=8, alpha=0.8)

x_tip_scatter_all = ax_body.scatter([], [], [], color="r", s=8, alpha=0.8)
y_tip_scatter_all = ax_body.scatter([], [], [], color="g", s=8, alpha=0.8)
z_tip_scatter_all = ax_body.scatter([], [], [], color="b", s=8, alpha=0.8)



# assumes 10 s timestep
trail_minutes = axis_trail_len * 10 / 60

norm = Normalize(vmin=0, vmax=trail_minutes)

sm_x = ScalarMappable(cmap="Reds", norm=norm)
sm_y = ScalarMappable(cmap="Greens", norm=norm)
sm_z = ScalarMappable(cmap="Blues", norm=norm)

sm_x.set_array([])
sm_y.set_array([])
sm_z.set_array([])

fig2.colorbar(
    sm_x,
    ax=ax_body_x,
    shrink=0.65,
    label="Trail age [min]"
)

fig2.colorbar(
    sm_y,
    ax=ax_body_y,
    shrink=0.65,
    label="Trail age [min]"
)

fig2.colorbar(
    sm_z,
    ax=ax_body_z,
    shrink=0.65,
    label="Trail age [min]"
)

def update_axis_scatter(scatter, axis_hist, i, scale, cmap_name, trail_len=300):
    j0 = max(0, i - trail_len)
    pts = axis_hist[j0:i+1] * scale

    scatter._offsets3d = (pts[:, 0], pts[:, 1], pts[:, 2])

    # age parameter: old = 0, recent = 1
    age = np.linspace(0.0, 1.0, len(pts))

    cmap = plt.get_cmap(cmap_name)
    colors = cmap(age)

    # old points faint, recent points strong
    colors[:, 3] = np.linspace(0.05, 1.0, len(pts))

    scatter.set_facecolors(colors)
    scatter.set_edgecolors(colors)

    # IMPORTANT for 3D scatter
    scatter._facecolor3d = colors
    scatter._edgecolor3d = colors


def update(i):
    # compute position, velocities and torques
    v_vec = v[i]
    v_mag = np.linalg.norm(v_vec)

    R = quat_to_dcm(q[i])
    w_inertial = R @ w_body[i]
    w_mag = np.linalg.norm(w_inertial)

    r_vec = r_m[i]          # position in meters
    r_km_vec = r_vec / 1000 # position in km
    r_mag = np.linalg.norm(r_km_vec)
    
    vel_text.set_text(
        f'r = [{r_km_vec[0]:.1f}, {r_km_vec[1]:.1f}, {r_km_vec[2]:.1f}] km\n'
        f'|r| = {r_mag:.1f} km\n\n'
        f'v = [{v_vec[0]/1000:.2f}, {v_vec[1]/1000:.2f}, {v_vec[2]/1000:.2f}] km/s\n'
        f'|v| = {v_mag/1000:.2f} km/s\n\n'
        f'ω = [{w_inertial[0]:.3f}, {w_inertial[1]:.3f}, {w_inertial[2]:.3f}] rad/s\n'
        f'|ω| = {w_mag:.4f} rad/s'
    )

    Tgg = T_gg[i]
    Tsrp = T_srp[i]

    Tgg_mag = np.linalg.norm(Tgg)
    Tsrp_mag = np.linalg.norm(Tsrp)

    torque_text.set_text(
        f'T_GG = [{Tgg[0]:.2e}, {Tgg[1]:.2e}, {Tgg[2]:.2e}] N m\n'
        f'|T_GG| = {Tgg_mag:.2e} N m\n\n'
        f'T_SRP = [{Tsrp[0]:.2e}, {Tsrp[1]:.2e}, {Tsrp[2]:.2e}] N m\n'
        f'|T_SRP| = {Tsrp_mag:.2e} N m'
    )


    # print the magnitudes of torques, positions, velocities
    gg_mag = np.linalg.norm(T_gg[i])
    srp_mag = np.linalg.norm(T_srp[i])
    

    global body_axes, body_axes_x, body_axes_y, body_axes_z, orbit_normal

    # ----------------------------
    # update ecliptic/orbit-plane plot
    # ----------------------------

    if orbit_normal is not None:
        orbit_normal.remove()

    # orbit plane normal from current inclination and RAAN
    n_orb = orbit_normal_from_elements(inc[i], raan[i])

    # update orbit plane circle
    orbit_circle = plane_disk_from_normal(n_orb, radius=plane_radius)

    orbit_line.set_data(orbit_circle[:, 0], orbit_circle[:, 1])
    orbit_line.set_3d_properties(orbit_circle[:, 2])

    # update transparent orbit-plane fill
    orb_fill.set_verts([orbit_circle])

    # update orbit normal arrow
    orbit_normal = ax_planes.quiver(
        0, 0, 0,
        n_orb[0], n_orb[1], n_orb[2],
        color="blue",
        linewidth=2
    )

    # line of nodes / RAAN direction
    n_ref = np.array([0.0, 0.0, 1.0])
    node_vec = np.cross(n_ref, n_orb)

    if np.linalg.norm(node_vec) > 1e-12:
        node_vec = node_vec / np.linalg.norm(node_vec)

        node_line.set_data(
            [-node_vec[0], node_vec[0]],
            [-node_vec[1], node_vec[1]]
        )
        node_line.set_3d_properties(
            [-node_vec[2], node_vec[2]]
        )

    # angle between orbit plane and ecliptic plane
    angle_planes = np.rad2deg(
        np.arccos(np.clip(np.dot(n_ecl, n_orb), -1.0, 1.0))
    )

    # slowly rotate the camera during the animation
    azim = 45 #+ 180 * i / len(t)
    elev = 45
    ax_planes.view_init(elev=elev, azim=azim)

    fig3.suptitle(
        f"t = {t[i]/3600:.2f} h / {t[i]/3600/24:.2f} days\n"
        r"$\Omega$" f" = {np.rad2deg(raan[i]):.2f}°, "
        f"i = {np.rad2deg(inc[i]):.3f}°, \n"
        f"orbit/ecliptic angle = {angle_planes:.3f}°"
    )

    R = quat_to_dcm(q[i])

    #rotate body axes

    # remove old ones
    for ax in body_axes:
        ax.remove()

    # remove old ones x direction 
    for ax in body_axes_x:
        ax.remove()
    # remove old ones y direction 
    for ax in body_axes_y:
        ax.remove()
    # remove old ones z direction 
    for ax in body_axes_z:
        ax.remove()

    # rotated body fixed axes
    x_axis = R @ np.array([1,0,0])
    y_axis = R @ np.array([0,1,0])
    z_axis = R @ np.array([0,0,1])

    scale = args.scale  # adjust to match your satellite size

    #update_axis_trail(x_tip_trail, axis1_hist, i, scale, rgb=(1,0,0), trail_len=axis_trail_len)
    #update_axis_trail(y_tip_trail, axis2_hist, i, scale, rgb=(0,1,0), trail_len=axis_trail_len)
    #update_axis_trail(z_tip_trail, axis3_hist, i, scale, rgb=(0,0,1), trail_len=axis_trail_len)

    #update_axis_trail(x_tip_trail_all, axis1_hist, i, scale, rgb=(1,0,0), trail_len=axis_trail_len)
    #update_axis_trail(y_tip_trail_all, axis2_hist, i, scale, rgb=(0,1,0), trail_len=axis_trail_len)
    #update_axis_trail(z_tip_trail_all, axis3_hist, i, scale, rgb=(0,0,1), trail_len=axis_trail_len)

    update_axis_scatter(x_tip_scatter, axis1_hist, i, scale, "Reds", trail_len=axis_trail_len)
    update_axis_scatter(y_tip_scatter, axis2_hist, i, scale, "Greens", trail_len=axis_trail_len)
    update_axis_scatter(z_tip_scatter, axis3_hist, i, scale, "Blues", trail_len=axis_trail_len)

    update_axis_scatter(x_tip_scatter_all, axis1_hist, i, scale, "Reds", trail_len=axis_trail_len)
    update_axis_scatter(y_tip_scatter_all, axis2_hist, i, scale, "Greens", trail_len=axis_trail_len)
    update_axis_scatter(z_tip_scatter_all, axis3_hist, i, scale, "Blues", trail_len=axis_trail_len)

    body_axes = [
        ax_body.quiver(0,0,0, *(x_axis*scale), color='r', linewidth=2),
        ax_body.quiver(0,0,0, *(y_axis*scale), color='g', linewidth=2),
        ax_body.quiver(0,0,0, *(z_axis*scale), color='b', linewidth=2)
    ]

    body_axes_x = [
        ax_body_x.quiver(0,0,0, *(x_axis*scale), color='r', linewidth=2)
    ]
    body_axes_y = [
        ax_body_y.quiver(0,0,0, *(y_axis*scale), color='g', linewidth=2)
    ]
    body_axes_z = [
        ax_body_z.quiver(0,0,0, *(z_axis*scale), color='b', linewidth=2)
    ]


    # rotate body mesh
    tris_now = rotate_tris(tris_body, R)
    mesh.set_verts(tris_now)
    mesh_x.set_verts(tris_now)
    mesh_y.set_verts(tris_now)
    mesh_z.set_verts(tris_now)

    # orbit marker
    sat_point.set_data([r_km[i, 0]], [r_km[i, 1]])
    sat_point.set_3d_properties([r_km[i, 2]])

    # attitude triad attached to spacecraft in orbit view
    center = r_km[i]
    dirs = np.array([R @ e1, R @ e2, R @ e3]) * triad_scale_km
    for j in range(3):
        p0 = center
        p1 = center + dirs[j]
        triad_lines[j].set_data([p0[0], p1[0]], [p0[1], p1[1]])
        triad_lines[j].set_3d_properties([p0[2], p1[2]])

    # body-axis trails
    #i0 = max(0, i - trail_len)

    #trail_x.set_data(axis1_hist[i0:i+1, 0], axis1_hist[i0:i+1, 1])
    #trail_x.set_3d_properties(axis1_hist[i0:i+1, 2])

    #trail_y.set_data(axis2_hist[i0:i+1, 0], axis2_hist[i0:i+1, 1])
    #trail_y.set_3d_properties(axis2_hist[i0:i+1, 2])

    #trail_z.set_data(axis3_hist[i0:i+1, 0], axis3_hist[i0:i+1, 1])
    #trail_z.set_3d_properties(axis3_hist[i0:i+1, 2])

    # remove previous torque arrows
    #if gg_quiv is not None:
    #    gg_quiv.remove()
    #if srp_quiv is not None:
    #    srp_quiv.remove()

    # torque vectors are already in body-fixed frame
    #gg_vec = T_gg[i] * gg_scale
    #srp_vec = T_srp[i] * srp_scale

    #gg_quiv = ax_body.quiver(
    #    0, 0, 0,
    #    gg_vec[0], gg_vec[1], gg_vec[2],
    #    color = 'yellow' ,linewidth = 2, label = "GG torque"
    #)

    #srp_quiv = ax_body.quiver(
    #    0, 0, 0,
    #    srp_vec[0], srp_vec[1], srp_vec[2],
    #    color = 'black' ,linewidth=2, label = "SRP torque"
    #)

    # ----------------------------
# vanishing trail for the orbit 
# ----------------------------

#orbit_trail.set_data(r_km[j0:i+1, 0], r_km[j0:i+1, 1])
#orbit_trail.set_3d_properties(r_km[j0:i+1, 2])

# ----------------------------
# fading alpha trail for the orbit 
# ----------------------------

    j0 = max(0, i - orbit_trail_len)
    pts = r_km[j0:i+1]

    if len(pts) >= 2:
        segments = np.stack([pts[:-1], pts[1:]], axis=1)

        alphas = np.linspace(0.05, 1.0, len(segments))
        colors = np.zeros((len(segments), 4))
        colors[:, 0] = 1.0      # red
        colors[:, 3] = alphas   # alpha fade

        orbit_segments.set_segments(segments)
        orbit_segments.set_color(colors)



    fig.suptitle(f"t = {t[i]/3600:.2f} h / {t[i]/3600/24:.2f} days")
    fig2.suptitle(
    f"t = {t[i]/3600:.2f} h / {t[i]/3600/24:.2f} days \n"
    f"Initial parameters:\n"
    f"r = [{r_m_init[0]:.2f}, {r_m_init[1]:.2f}, {r_m_init[2]:.2f}] km, "
    f"v = [{v_init[0]:.3f}, {v_init[1]:.3f}, {v_init[2]:.3f}] km/s,"
    f"ω = [{w_body_init[0]:.2f}, {w_body_init[1]:.2f}, {w_body_init[2]:.2f}] °/s,\n "
    rf"$\phi$ = {phi_init:.2f}°, "
    rf"$\theta$ = {theta_init:.2f}°, "
    rf"$\psi$ = {psi_init:.2f}° "
    f"w.r.t {frame_name}",
    fontsize=10
)
    return [
    mesh, mesh_x, mesh_y, mesh_z,
    sat_point, *triad_lines,
    orbit_segments,
    *body_axes, *body_axes_x, *body_axes_y, *body_axes_z,
    x_tip_scatter, y_tip_scatter, z_tip_scatter, x_tip_scatter_all, y_tip_scatter_all, z_tip_scatter_all
] # orbit_trail ,  gg_quiv, srp_quiv
#----skip frames---
#frame_step = 5
#frames = range(0, len(t), frame_step)
#ani = FuncAnimation(fig, update, frames=frames, interval=80, blit=False)


def show_plot(hold=False, save=False):
    if hold:
        hold_frames = 4
        ani = FuncAnimation(fig, update, frames=len(t), interval=hold_frames*60, blit=False)
        ani2 = FuncAnimation(fig2, update, frames=len(t), interval=hold_frames*60, blit=False)
        ani3 = FuncAnimation(fig3, update, frames=len(t), interval=hold_frames*60, blit=False)
    else:
        ani = FuncAnimation(fig, update, frames=len(t), interval=8, blit=False)
        ani2 = FuncAnimation(fig2, update, frames=len(t), interval=8, blit=False)
        #ani3 = FuncAnimation(fig3, update, frames=len(t), interval=8, blit=False)

    plt.tight_layout()

    if save:
        ani.save(outdir / f"{args.prefix}_main.mp4", fps=args.fps, dpi=args.dpi)
        ani2.save(outdir / f"{args.prefix}_axes.mp4", fps=args.fps, dpi=args.dpi)
        ani3.save(outdir / f"{args.prefix}_planes.mp4", fps=args.fps, dpi=args.dpi)

    return ani, ani2, ani3


ani, ani2  = show_plot(hold=False, save=args.save)  # , ani3

if args.show:
    plt.show()




# entire plot of the angular velocity vector
scale = 8.0
time_days = t / 3600 / 24

fig = plt.figure(figsize=(10,8))
ax = fig.add_subplot(111, projection="3d")

# scale trajectories
pts_x = axis1_hist * scale
pts_y = axis2_hist * scale
pts_z = axis3_hist * scale

# X-axis trajectory
scx = ax.scatter(
    pts_x[:,0],
    pts_x[:,1],
    pts_x[:,2],
    c=time_days,
    cmap="Reds",
    s=4,
    label="Body X-axis"
)

# Y-axis trajectory
scy = ax.scatter(
    pts_y[:,0],
    pts_y[:,1],
    pts_y[:,2],
    c=time_days,
    cmap="Greens",
    s=4,
    label="Body Y-axis"
)

# Z-axis trajectory
scz = ax.scatter(
    pts_z[:,0],
    pts_z[:,1],
    pts_z[:,2],
    c=time_days,
    cmap="Blues",
    s=4,
    label="Body Z-axis"
)

# starting points
ax.scatter(
    pts_x[0,0], pts_x[0,1], pts_x[0,2],
    color="darkred", s=60
)

ax.scatter(
    pts_y[0,0], pts_y[0,1], pts_y[0,2],
    color="darkgreen", s=60
)

ax.scatter(
    pts_z[0,0], pts_z[0,1], pts_z[0,2],
    color="darkblue", s=60
)

# ending points
ax.scatter(
    pts_x[-1,0], pts_x[-1,1], pts_x[-1,2],
    color="red", s=80, marker="x"
)

ax.scatter(
    pts_y[-1,0], pts_y[-1,1], pts_y[-1,2],
    color="lime", s=80, marker="x"
)

ax.scatter(
    pts_z[-1,0], pts_z[-1,1], pts_z[-1,2],
    color="cyan", s=80, marker="x"
)

ax.set_title("Body-axis tip trajectories")
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")

ax.set_box_aspect([1,1,1])
ax.legend()

ax.view_init(elev=-45 , azim=45)

# one shared colorbar
cbar = fig.colorbar(scx, ax=ax, shrink=0.7)
cbar.set_label("Time [days]")

plt.tight_layout()
plt.show()

# entire orbit plot 
fig = plt.figure(figsize=(9, 8))
ax = fig.add_subplot(111, projection="3d")

draw_earth(ax)

sc = ax.scatter(
    r_km[:, 0],
    r_km[:, 1],
    r_km[:, 2],
    c=t/3600/24,
    cmap="viridis",
    s=2
)

ax.scatter(
    r_km[0, 0], r_km[0, 1], r_km[0, 2],
    color="green", s=60, label="start"
)

ax.scatter(
    r_km[-1, 0], r_km[-1, 1], r_km[-1, 2],
    color="red", s=60, label="end"
)

ax.set_title("Satellite orbit over full data set")
ax.set_xlabel("X [km]")
ax.set_ylabel("Y [km]")
ax.set_zlabel("Z [km]")

set_equal_axes(ax, r_km)
ax.legend()

cbar = fig.colorbar(sc, ax=ax, shrink=0.7)
cbar.set_label("Time [days]")

plt.tight_layout()
plt.show()

