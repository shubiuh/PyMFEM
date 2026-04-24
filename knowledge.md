## Maxwell's equations

To derive the curl-curl equation with a source term—typically an impressed current density $\mathbf{J}_s$—we begin with Maxwell’s equations in the frequency domain ($e^{j\omega t}$ convention).

### 1. The Governing Maxwell Equations
We start with Faraday’s Law and Ampere’s Law for a linear, isotropic medium:

$$\nabla \times \mathbf{E} = -j\omega\mathbf{B} \tag{1}$$
$$\nabla \times \mathbf{H} = \mathbf{J} + j\omega\mathbf{D} \tag{2}$$

### 2. Incorporating Constitutive Relations and Sources
We define the relationships for the flux densities and the total current density. In geophysical applications and FEM modeling, we explicitly separate the **impressed source current** ($\mathbf{J}_s$) from the **induced conduction current** ($\sigma\mathbf{E}$):

* $\mathbf{B} = \mu_0\mu_r\mathbf{H}$
* $\mathbf{D} = \epsilon_0\epsilon_r\mathbf{E}$
* $\mathbf{J} = \sigma\mathbf{E} + \mathbf{J}_s$

Substituting these into Equation (2):
$$\nabla \times \mathbf{H} = (\sigma + j\omega\epsilon_0\epsilon_r)\mathbf{E} + \mathbf{J}_s \tag{3}$$

### 3. Combining the Equations
From Equation (1), we solve for the magnetic field $\mathbf{H}$:
$$\mathbf{H} = -\frac{1}{j\omega\mu_0\mu_r} \nabla \times \mathbf{E}$$

Now, substitute this expression for $\mathbf{H}$ into Equation (3):
$$\nabla \times \left( \frac{1}{-j\omega\mu_0\mu_r} \nabla \times \mathbf{E} \right) = (\sigma + j\omega\epsilon_0\epsilon_r)\mathbf{E} + \mathbf{J}_s$$

### 4. Final Algebraic Rearrangement
To match the standard FEM form (like the one in your COMSOL screenshot), we multiply the entire equation by $-j\omega\mu_0$:

$$\nabla \times \mu_r^{-1}(\nabla \times \mathbf{E}) = -j\omega\mu_0(\sigma + j\omega\epsilon_0\epsilon_r)\mathbf{E} - j\omega\mu_0\mathbf{J}_s$$

Distributing the terms on the right side:
$$\nabla \times \mu_r^{-1}(\nabla \times \mathbf{E}) = (-j\omega\mu_0\sigma + \omega^2\mu_0\epsilon_0\epsilon_r)\mathbf{E} - j\omega\mu_0\mathbf{J}_s$$

Recall that the free-space wavenumber squared is $k_0^2 = \omega^2\mu_0\epsilon_0$. We can factor this out to align with the complex permittivity form:
$$\nabla \times \mu_r^{-1}(\nabla \times \mathbf{E}) - \omega^2\mu_0\epsilon_0 \left( \epsilon_r - \frac{j\sigma}{\omega\epsilon_0} \right) \mathbf{E} = -j\omega\mu_0\mathbf{J}_s$$

### The Final Curl-Curl Equation
The resulting equation, which accounts for both material properties and the source term, is:

$$\nabla \times \mu_r^{-1}(\nabla \times \mathbf{E}) - k_0^2 \left( \epsilon_r - \frac{j\sigma}{\omega\epsilon_0} \right) \mathbf{E} = -j\omega\mu_0\mathbf{J}_s$$

---

### Implementation Note for FEM Solvers
When implementing this in a 3D solver (like your `FEM3D_UDAR` project):
* The **Left-Hand Side (LHS)** remains the same as the source-free Helmholtz equation you shared earlier.
* The **Right-Hand Side (RHS)** represents the source vector. In an edge-element formulation, you would integrate the $\mathbf{J}_s$ term against the vector basis functions $\mathbf{N}_i$:
    $$\mathbf{b}_i = -j\omega\mu_0 \int_{\Omega} \mathbf{J}_s \cdot \mathbf{N}_i \, d\Omega$$
* For a **magnetic dipoles** (common in LWD/geosteering), the source is often treated as an equivalent current loop or via the magnetic source term $\nabla \times \mu_r^{-1}\mathbf{M}_s$.

The choice between these two conventions is essentially a choice of coordinate systems in the complex plane. While both are mathematically valid, they lead to a "mirror-image" set of signs in your derivations and code.

## Core Mathematical Difference between $e^{j\omega t}$ and $e^{-i\omega t}$ conventions in the frequnecy domain
The fundamental difference lies in how the time-derivative operator $\frac{\partial}{\partial t}$ is represented:

| Feature | Engineering/Applied ($e^{j\omega t}$) | Physics/Theory ($e^{-i\omega t}$) |
| :--- | :--- | :--- |
| **Time Derivative ($\partial_t$)** | $+j\omega$ | $-i\omega$ |
| **Phasor Rotation** | Counter-clockwise | Clockwise |
| **Faraday's Law** | $\nabla \times \mathbf{E} = -j\omega\mathbf{B}$ | $\nabla \times \mathbf{E} = i\omega\mathbf{B}$ |
| **Propagation Factor** | $e^{-jkz}$ (forward wave) | $e^{ikz}$ (forward wave) |
| **Lossy Permittivity** | $\epsilon = \epsilon' - j\frac{\sigma}{\omega}$ | $\epsilon = \epsilon' + i\frac{\sigma}{\omega}$ |
| **Common Software** | COMSOL, Ansys, Keysight | Mathematica, many QM solvers |

---

### Impact on the Curl-Curl Equation

If you switch from the $e^{j\omega t}$ convention used in your previous derivation to $e^{-i\omega t}$, several signs flip.

#### In the $e^{j\omega t}$ Convention (Your Image):
The complex permittivity is expressed with a **minus** sign to ensure that energy is dissipated (decaying wave) rather than amplified:
$$\epsilon_r^* = \epsilon_r - \frac{j\sigma}{\omega\epsilon_0}$$

#### In the $e^{-i\omega t}$ Convention:
To maintain the same physical reality (waves must decay in a conductor), the complex permittivity must use a **plus** sign:
$$\epsilon_r^* = \epsilon_r + \frac{i\sigma}{\omega\epsilon_0}$$

The resulting Helmholtz equation would look like this:
$$\nabla \times \mu_r^{-1}(\nabla \times \mathbf{E}) - \omega^2\mu_0\epsilon_0 \left( \epsilon_r + \frac{i\sigma}{\omega\epsilon_0} \right) \mathbf{E} = i\omega\mu_0\mathbf{J}_s$$

---

### Why Geophysics Prefers $e^{j\omega t}$
In computational geophysics and electrical engineering, $e^{j\omega t}$ is the dominant standard. 

1.  **Impedance Consistency:** It aligns with the circuit theory definition of impedance $Z = R + jX$. If you used the physics convention, an inductor would have a negative reactance ($-i\omega L$), which is counterintuitive for most developers.
2.  **Software Interoperability:** Tools like COMSOL (which you are using) and standard C++ libraries for electromagnetics almost universally default to $+j\omega$. 

### The "Conjugate" Rule
If you have a solution $E(j)$ derived in the engineering convention and you need the equivalent in the physics convention, you can simply take the **complex conjugate**:
$$E(i) = \left[ E(j) \right]^*$$
*(Assuming you also swap $j$ for $-i$ everywhere in your material definitions).*

> **A Note for your Code:** When moving between MATLAB (which handles `j` and `i` interchangeably) and C++ (where you might use `std::complex`), it is critical to stick to one convention for the entire solver. Mixing these up is the most common cause of "reversed" phase results in 3D EM modeling.