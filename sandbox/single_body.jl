
using Revise
using ConjugateHeatTransfer
using CairoMakie
using LinearAlgebra
using QuadGK

# Parameters
U = 1;      # far-field velocity
N = 16;     # points per slit
T1 = 1;     # temperature of body
quad(f, a, b) = quadgk(f, a, b; atol=1e-10, rtol=1e-10)[1];
quad(f, a, b, c) = quadgk(f, a, b, c; atol=1e-10, rtol=1e-10)[1];

# Mapping
W(z) = conj(U)*z + U/z;

# Panels
p1 = panel(0, 0, 2, N);
panels = [p1];

# Solve for temperature potentials
M = SystemMatrix(panels; quadrature=quad);
println("Condition number: ", cond(M))
T = T1 * ones(N);
f = M \ T;
f1 = BuildInterpolant(p1.space, f[1:N]);
f = [f1];

# Build regular physical grid for plotting
x = Vector(range(-4, 4, length=200));
y = Vector(range(-4, 4, length=200));
z = @. x' + im*y;
Wz = W.(z);
φ = real(Wz);
ψ = imag(Wz);

# Evaluate temperature on regular grid
T = EvaluateT(φ, ψ, panels, f; quadrature=quad);
T[abs.(z) .< 1] .= T1;

# Plot
fig = Figure(size=(800, 800));
ax = Axis(fig[1, 1], aspect=DataAspect());
co = contourf!(ax, x, y, T', levels=20, extendlow=:auto, extendhigh=:auto);
arc!((0,0), 1, 0, 2pi, color=:black, linewidth=1);
Colorbar(fig[1,2], co);
fig

