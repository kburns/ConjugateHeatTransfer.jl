
using Revise
using ConjugateHeatTransfer
using CairoMakie
using LinearAlgebra
using QuadGK

# Parameters
U = -1;     # far-field velocity
L = 5;      # distance between bodies
N = 32;     # points per slit
solve_quad(args...) = quadgk(args...; atol=1e-10, rtol=1e-10)[1];
plot_quad(args...) = collect(quadgk_count(args...; atol=1e-3, rtol=1e-3));

# Bodies
z1(θ) = exp(im*θ);
T1(θ) = 2 + cos(4*θ) + sin(4*θ);
z2(θ) = L + exp(im*θ);
T2(θ) = 3;
bodies = [DirichletBody(z1, T1), DirichletBody(z2, T2)];

# Flow map
a = (-L + sqrt(L^2-4)) / 2;
A = 1 - abs(a)^2;
ρ = (-2 + L^2 - L*sqrt(L^2-4)) / 2;
function Crowdy_K(z, a, r, trunc=100)
    out = -(z/a) / (1 - z/a)
    for n = 1:trunc
        out -= r^(2*n) * (z/a) / (1 - r^(2*n) * z/a)
        out += r^(2*n) * (a/z) / (1 - r^(2*n) * a/z)
    end
    return out
end
ξ(z) = a + A / (z + a);
W(ξ) = U * (A/a) * (Crowdy_K(ξ,1/a,ρ) - Crowdy_K(ξ,a,ρ));
t0 = time();
reparametrize!(bodies, z->W(ξ(z)));
println("  Reparametrization done (", round(time()-t0, digits=3), " s)");

# Panels
panels = [DirichletPanel(body, N) for body in bodies];
t0 = time();
solve_densities!(panels, quadrature=solve_quad);
println("  Densities solved (", round(time()-t0, digits=3), " s)");

# Build regular physical grid for plotting
x = Vector(range(-L, 2*L, length=200));
y = Vector(range(-1.5*L, 1.5*L, length=200));
z = @. x' + im*y;
Wz = W.(ξ.(z));
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
T[abs.(z.-L) .< 1] .= T2.(angle.(z.-L))[abs.(z.-L) .< 1];

# Plot
fig = Figure(size=(2000, 600));

ax = Axis(fig[1,1], aspect=DataAspect());
co = contourf!(ax, x, y, T', levels=20, extendlow=:auto, extendhigh=:auto);
arc!((0,0), 1, 0, 2pi, color=:black, linewidth=1);
arc!((L,0), 1, 0, 2pi, color=:black, linewidth=1);
Colorbar(fig[1,2], co);

ax = Axis(fig[1,3], aspect=DataAspect());
co = contourf!(ax, x, y, err', levels=20);
arc!((0,0), 1, 0, 2pi, color=:black, linewidth=1);
arc!((L,0), 1, 0, 2pi, color=:black, linewidth=1);
Colorbar(fig[1,4], co);

ax = Axis(fig[1,5], aspect=DataAspect());
co = contourf!(ax, x, y, counts', levels=20);
arc!((0,0), 1, 0, 2pi, color=:black, linewidth=1);
arc!((L,0), 1, 0, 2pi, color=:black, linewidth=1);
Colorbar(fig[1,6], co);

save("double_body.png", fig);

