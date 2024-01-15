
using Revise
using ConjugateHeatTransfer
using CairoMakie
using LinearAlgebra
using QuadGK

# Parameters
U = 1;      # far-field velocity
N = 16;     # points per slit
T1 = 1;     # temperature of body
quad(args...) = quadgk_count(args...; atol=1e-10, rtol=1e-10);
solve_quad(args...) = quad(args...)[1];
plot_quad(args...) = collect(quad(args...));

# Mapping
W(z) = conj(U)*z + U/z;

# Panels
p1 = panel(0, 0, 2, N);
panels = [p1];

# Solve for temperature potentials
M = SystemMatrix(panels; quadrature=solve_quad);
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
data = EvaluateSystem(φ, ψ, panels, f; quadrature=plot_quad);
T = (x->getindex(x,1)).(data);
error = (x->getindex(x,2)).(data);
counts = (x->getindex(x,3)).(data);
T[abs.(z) .< 1] .= T1;

# Plot
fig = Figure(size=(2000, 600));

ax = Axis(fig[1,1], aspect=DataAspect());
co = contourf!(ax, x, y, T', levels=20, extendlow=:auto, extendhigh=:auto);
arc!((0,0), 1, 0, 2pi, color=:black, linewidth=1);
Colorbar(fig[1,2], co);

ax = Axis(fig[1,3], aspect=DataAspect());
co = contourf!(ax, x, y, error', levels=20);
arc!((0,0), 1, 0, 2pi, color=:black, linewidth=1);
Colorbar(fig[1,4], co);

ax = Axis(fig[1,5], aspect=DataAspect());
co = contourf!(ax, x, y, counts', levels=20);
arc!((0,0), 1, 0, 2pi, color=:black, linewidth=1);
Colorbar(fig[1,6], co);

save("single_body.png", fig);

