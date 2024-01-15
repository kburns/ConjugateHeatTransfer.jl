
using Revise
using ConjugateHeatTransfer
using CairoMakie
using LinearAlgebra
using QuadGK

# Parameters
U = -1;     # far-field velocity
L = 5;      # distance between bodies
N = 32;     # points per slit
T1 = 1;     # temperature of body 1
T2 = -1;    # temperature of body 2
quad(args...) = quadgk_count(args...; atol=1e-10, rtol=1e-10);
solve_quad(args...) = quad(args...)[1];
plot_quad(args...) = collect(quad(args...));

# Mapping
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

# Panels
a1 = W(ξ(1))
b1 = W(ξ(-1))
p1 = panel((a1+b1)/2, 0, abs(b1-a1)/2, N);
a2 = W(ξ(L+1))
b2 = W(ξ(L-1))
p2 = panel((a2+b2)/2, 0, abs(b2-a2)/2, N);
panels = [p1, p2];

# Solve for temperature potentials
M = SystemMatrix(panels; quadrature=solve_quad);
println("Condition number: ", cond(M))
T = [T1*ones(N); T2*ones(N)]
f = M \ T;
f1 = BuildInterpolant(p1.space, f[1:N]);
f2 = BuildInterpolant(p2.space, f[N+1:end]);
f = [f1, f2];

# Build regular physical grid for plotting
x = Vector(range(-L, 2*L, length=200));
y = Vector(range(-1.5*L, 1.5*L, length=200));
z = @. x' + im*y;
Wz = W.(ξ.(z));
φ = real(Wz);
ψ = imag(Wz);

# Evaluate temperature on regular grid
data = EvaluateSystem(φ, ψ, panels, f; quadrature=plot_quad);
T = (x->getindex(x,1)).(data);
error = (x->getindex(x,2)).(data);
counts = (x->getindex(x,3)).(data);
T[abs.(z) .< 1] .= T1;
T[abs.(z.-L) .< 1] .= T2;

# Plot
# Plot
fig = Figure(size=(2000, 600));

ax = Axis(fig[1,1], aspect=DataAspect());
co = contourf!(ax, x, y, T', levels=20, extendlow=:auto, extendhigh=:auto);
arc!((0,0), 1, 0, 2pi, color=:black, linewidth=1);
arc!((L,0), 1, 0, 2pi, color=:black, linewidth=1);
Colorbar(fig[1,2], co);

ax = Axis(fig[1,3], aspect=DataAspect());
co = contourf!(ax, x, y, error', levels=20);
arc!((0,0), 1, 0, 2pi, color=:black, linewidth=1);
arc!((L,0), 1, 0, 2pi, color=:black, linewidth=1);
Colorbar(fig[1,4], co);

ax = Axis(fig[1,5], aspect=DataAspect());
co = contourf!(ax, x, y, counts', levels=20);
arc!((0,0), 1, 0, 2pi, color=:black, linewidth=1);
arc!((L,0), 1, 0, 2pi, color=:black, linewidth=1);
Colorbar(fig[1,6], co);

save("double_body.png", fig);

