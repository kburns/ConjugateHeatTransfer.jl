

using Revise
using ConjugateHeatTransfer
using CairoMakie
using LinearAlgebra
using QuadGK
using PolygonOps
using StaticArrays


# Parameters
U = 1 + 1im; # free stream velocity
N_bie = 32; # number of points per slit for BIE
solve_quad(args...) = quadgk(args...; atol=1e-10, rtol=1e-10)[1];
plot_quad(args...) = quadgk(args...; atol=1e-5, rtol=1e-5)[1];

# Bodies
bodyshape(θ) = exp(im*θ) * (1 + 0.2*sin(3*θ));
bodytemp(θ) = 1 + 0.5*sin(3*θ);
zc1 = -2;
zc2 = 2im;
zc3 = -2im;
zc4 = 2;
centers = [zc1, zc2, zc3, zc4];
bodies = [DirichletBody(zc, θ->zc+bodyshape(θ), θ->bodytemp(θ)) for zc in centers];

# Solve potential flow
t0 = time();
W = adaptive_laurent_potential_flow(bodies, U; atol=1e-6)
println("Solved potential flow (", round(time()-t0, digits=3), " s)");
t0 = time();
reparametrize!(bodies, W; atol=1e-6);
println("Joukowsky reparametrization done (", round(time()-t0, digits=3), " s)");

# Panels
panels = [DirichletPanel(body) for body in bodies];
t0 = time();
solve_densities!(panels, N_bie; atol=1e-6, quadrature=solve_quad);
println("Densities solved (", round(time()-t0, digits=3), " s)");

# Evaluate on a grid
gridsize = 200
x = collect(LinRange(-5, 5, gridsize));
y = collect(LinRange(-5, 5, gridsize));
z = x' .+ im*y;
# Evaluate flow
t0 = time();
w = reshape(W(vec(z)), size(z));
println("Flow evaluated (", round(time()-t0, digits=3), " s)");
# Mask interiors before evaluating temperature
φ = real(w);
ψ = imag(w);
function check_inside(body, z)
    poly = body.zθ.(LinRange(0, 2π, 100))
    poly[end] = poly[1]
    poly = [[real(z), imag(z)] for z in poly]
    return inpolygon([real(z), imag(z)], poly; in=true, on=false, out=false)
end
for body in bodies
    check_body(z) = check_inside(body, z)
    interior = check_body.(z)
    φ[interior] .= 0
    ψ[interior] .= 0
end
# Evaluate temperature
t0 = time();
T = EvaluateSystem(φ, ψ, panels; quadrature=plot_quad);
println("Temperature evaluated (", round(time()-t0, digits=3), " s)");
# Mask interiors
for body in bodies
    check_body(z) = check_inside(body, z)
    interior = check_body.(z)
    φ[interior] .= NaN
    ψ[interior] .= body.ψ
    T[interior] .= NaN
end

# Plot
fig = Figure(size=(1400, 600));

ax = Axis(fig[1,1], aspect=DataAspect(), title="Streamfunction");
co = contourf!(ax, x, y, ψ', levels=20, extendlow=:auto, extendhigh=:auto);
for body in bodies
    boundary = body.zθ.(LinRange(0, 2π, 100))
    lines!(ax, real(boundary), imag(boundary), color=:black, linewidth=6)
end
Colorbar(fig[1,2], co);

ax = Axis(fig[1,3], aspect=DataAspect(), title="Temperature");
co = contourf!(ax, x, y, T', levels=20, extendlow=:auto, extendhigh=:auto);
for body in bodies
    boundary = body.zθ.(LinRange(0, 2π, 100))
    lines!(ax, real(boundary), imag(boundary), color=:black, linewidth=6)
end
Colorbar(fig[1,4], co);

save("sandbox/trefoils.png", fig);

