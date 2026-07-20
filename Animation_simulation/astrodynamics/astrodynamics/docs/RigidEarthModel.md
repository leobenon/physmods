# Rigid Earth Rotation Model

## 1. Scope

This model describes the rotational motion of a rigid, axisymmetric Earth
subject to a simplified lunar gravity-gradient torque.

The implementation is based on the MATLAB files:

- `rot2.m`
- `deqrot2.m`

The initial Python implementation aims to reproduce the MATLAB model before
introducing physical or numerical improvements.

---

## 2. State Vector

The propagated state is

$$
\mathbf y =
\begin{bmatrix}
\omega_1 &
\omega_2 &
\omega_3 &
\psi &
\varepsilon &
\theta
\end{bmatrix}^{T}.
$$

The quantities \(\omega_1,\omega_2,\omega_3\) are the components of the
Earth's angular velocity in the body-fixed principal-axis frame.

The angles \(\psi,\varepsilon,\theta\) describe the Earth's orientation using
the Euler-angle convention of the original MATLAB model.

---

## 3. Time and Angle Constants

$$
1\ \mathrm{day}=86400\ \mathrm{s}
$$

$$
1\ \mathrm{rad}
=
\frac{180}{\pi}\ \mathrm{deg}
=
206264.806\ \mathrm{arcsec}
$$

---

## 4. Earth Parameters

The nominal Earth rotation rate is

$$
\omega_\oplus
=
7.2921151467\times10^{-5}\ \mathrm{rad\,s^{-1}}.
$$

The initial obliquity is

$$
\varepsilon_0 = 23.5^\circ.
$$

The inertia-difference ratios are

$$
\gamma_1=\frac{C-B}{A},
\qquad
\gamma_2=\frac{A-C}{B},
\qquad
\gamma_3=\frac{B-A}{C}.
$$

The model uses

$$
\gamma_1=0.003295669,
\qquad
\gamma_2=-0.003295669,
\qquad
\gamma_3=0.
$$

Because \(\gamma_3=0\), the model assumes

$$
A=B,
$$

and therefore treats Earth as an axisymmetric rigid body.

---

## 5. Moon Parameters

The lunar gravitational parameter is approximated by

$$
\mu_M
=
\frac{398.6\times10^{12}}{81.3}
\ \mathrm{m^3\,s^{-2}}.
$$

The simplified lunar orbit uses

$$
r_M=3.8\times10^8\ \mathrm{m},
$$

$$
n_M=2.661707223\times10^{-6}\ \mathrm{rad\,s^{-1}},
$$

and an orbital inclination

$$
i_M=28^\circ.
$$

The Moon is initially modeled as moving on a circular orbit.

## 6. Elementary Rotation Matrices

The implementation uses active, right-handed rotations acting on column
vectors:

$$
\mathbf v' = \mathbf R\,\mathbf v.
$$

Rotation about the first axis:

$$
\mathbf R_x(\alpha)=
\begin{bmatrix}
1 & 0 & 0 \\
0 & \cos\alpha & -\sin\alpha \\
0 & \sin\alpha & \cos\alpha
\end{bmatrix}.
$$

Rotation about the second axis:

$$
\mathbf R_y(\beta)=
\begin{bmatrix}
\cos\beta & 0 & \sin\beta \\
0 & 1 & 0 \\
-\sin\beta & 0 & \cos\beta
\end{bmatrix}.
$$

Rotation about the third axis:

$$
\mathbf R_z(\gamma)=
\begin{bmatrix}
\cos\gamma & -\sin\gamma & 0 \\
\sin\gamma & \cos\gamma & 0 \\
0 & 0 & 1
\end{bmatrix}.
$$

Every proper rotation matrix satisfies

$$
\mathbf R^\mathsf{T}\mathbf R=\mathbf I,
\qquad
\det(\mathbf R)=+1,
\qquad
\mathbf R^{-1}=\mathbf R^\mathsf{T}.
$$

### Matrices in the original MATLAB model

The lunar-orbit inclination matrix is

$$
\mathbf A=\mathbf R_x(i_M).
$$

The sidereal-angle matrix is

$$
\mathbf B=\mathbf R_z(-\theta).
$$

The lunar position is transformed according to

$$
\mathbf r_M^{\F}
=
\mathbf R_z(-\theta)
\mathbf R_x(i_M)
\mathbf r_M^{\I}.
$$

The superscripts \(I\) and \(F\) denote inertial and Earth-fixed
coordinates, respectively.