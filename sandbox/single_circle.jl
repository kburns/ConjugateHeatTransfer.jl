
using Revise
using ConjugateHeatTransfer
using CairoMakie
using LinearAlgebra
using QuadGK

# Parameters
U = 1;      # far-field velocity
N = 16;     # points per slit
solve_quad(args...) = quadgk(args...; atol=1e-10, rtol=1e-10)[1];
plot_quad(args...) = collect(quadgk_count(args...; atol=1e-3, rtol=1e-3));

# Bodies
z1(θ) = exp(im*θ);
T1(θ) = 2 + cos(4*θ) + sin(4*θ);
bodies = [DirichletBody(0, z1, T1)];

# Flow map
U = 1;
W(z) = conj(U)*z + U/z;
t0 = time();
reparametrize!(bodies, W);
println("  Reparametrization done (", round(time()-t0, digits=3), " s)");

# Panels
panels = [DirichletPanel(body) for body in bodies];
t0 = time();
solve_densities!(panels, N; quadrature=solve_quad);
println("  Densities solved (", round(time()-t0, digits=3), " s)");

# Build regular physical grid for plotting
x = Vector(range(-4, 4, length=200));
y = Vector(range(-4, 4, length=200));
z = @. x' + im*y;
Wz = W.(z);
φ = real(Wz);
ψ = imag(Wz);

# Evaluate temperature on regular grid
t0 = time();
data = EvaluateSystem(φ, ψ, panels; quadrature=plot_quad);
println("  System evaluated (", round(time()-t0, digits=3), " s)");
T = (x->getindex(x,1)).(data);
err = (x->getindex(x,2)).(data);
counts = (x->getindex(x,3)).(data);
T[abs.(z) .< 1] .= T1.(angle.(z))[abs.(z) .< 1];

# Plot
fig = Figure(size=(2000, 600));

ax = Axis(fig[1,1], aspect=DataAspect());
co = contourf!(ax, x, y, T', levels=20, extendlow=:auto, extendhigh=:auto);
arc!((0,0), 1, 0, 2pi, color=:black, linewidth=1);
Colorbar(fig[1,2], co);

ax = Axis(fig[1,3], aspect=DataAspect());
co = contourf!(ax, x, y, err', levels=20);
arc!((0,0), 1, 0, 2pi, color=:black, linewidth=1);
Colorbar(fig[1,4], co);

ax = Axis(fig[1,5], aspect=DataAspect());
co = contourf!(ax, x, y, counts', levels=20);
arc!((0,0), 1, 0, 2pi, color=:black, linewidth=1);
Colorbar(fig[1,6], co);

save("sandbox/single_body.png", fig);

