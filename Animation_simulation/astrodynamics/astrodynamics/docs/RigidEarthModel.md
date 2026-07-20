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

\[
\mathbf y =
\begin{bmatrix}
\omega_1 &
\omega_2 &
\omega_3 &
\psi &
\varepsilon &
\theta
\end{bmatrix}^{T}.
\]

The quantities \(\omega_1,\omega_2,\omega_3\) are the components of the
Earth's angular velocity in the body-fixed principal-axis frame.

The angles \(\psi,\varepsilon,\theta\) describe the Earth's orientation using
the Euler-angle convention of the original MATLAB model.

---

## 3. Time and Angle Constants

\[
1\ \mathrm{day}=86400\ \mathrm{s}
\]

\[
1\ \mathrm{rad}
=
\frac{180}{\pi}\ \mathrm{deg}
=
206264.806\ \mathrm{arcsec}
\]

---

## 4. Earth Parameters

The nominal Earth rotation rate is

\[
\omega_\oplus
=
7.2921151467\times10^{-5}\ \mathrm{rad\,s^{-1}}.
\]

The initial obliquity is

\[
\varepsilon_0 = 23.5^\circ.
\]

The inertia-difference ratios are

\[
\gamma_1=\frac{C-B}{A},
\qquad
\gamma_2=\frac{A-C}{B},
\qquad
\gamma_3=\frac{B-A}{C}.
\]

The model uses

\[
\gamma_1=0.003295669,
\qquad
\gamma_2=-0.003295669,
\qquad
\gamma_3=0.
\]

Because \(\gamma_3=0\), the model assumes

\[
A=B,
\]

and therefore treats Earth as an axisymmetric rigid body.

---

## 5. Moon Parameters

The lunar gravitational parameter is approximated by

\[
\mu_M
=
\frac{398.6\times10^{12}}{81.3}
\ \mathrm{m^3\,s^{-2}}.
\]

The simplified lunar orbit uses

\[
r_M=3.8\times10^8\ \mathrm{m},
\]

\[
n_M=2.661707223\times10^{-6}\ \mathrm{rad\,s^{-1}},
\]

and an orbital inclination

\[
i_M=28^\circ.
\]

The Moon is initially modeled as moving on a circular orbit.