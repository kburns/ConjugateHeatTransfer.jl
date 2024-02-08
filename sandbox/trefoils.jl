

using Revise
using ConjugateHeatTransfer
using CairoMakie
using LinearAlgebra
using QuadGK
using PolygonOps
using StaticArrays


# Parameters
bodyshape(θ) = exp(im*θ) * (1 + 0.2*sin(3*θ));
N_samples = 200; # number of sample points on each body
N_laurent = 70; # number of terms in Laurent series around each body
U = 1 + 1im; # free stream velocity

# Bodies
zc1 = -2;
zc2 = 2im;
zc3 = -2im;
zc4 = 2;
centers = [zc1, zc2, zc3, zc4];
bodies = [DirichletBody(zc, θ->zc+bodyshape(θ), θ->0) for zc in centers];

# Solve potential flow
W = laurent_potential_flow(bodies, U, N_samples, N_laurent)

# Evaluate on grid
gridsize = 400
x = collect(LinRange(-5, 5, gridsize));
y = collect(LinRange(-5, 5, gridsize));
z = x .+ im*y';
w = reshape(W(vec(z)), size(z));

# Mask inside
function check_inside(body, z)
    poly = body.zθ.(LinRange(0, 2π, 100))
    poly[end] = poly[1]
    poly = [[real(z), imag(z)] for z in poly]
    return inpolygon([real(z), imag(z)], poly; in=true, on=false, out=false)
end
for body in bodies
    check_body(z) = check_inside(body, z)
    w[check_body.(z)] .= NaN+NaN*im
end

# Plot
fig = Figure(size=(1000, 1000));
ax = Axis(fig[1,1], aspect=DataAspect());
co = contourf!(ax, x, y, imag(w), levels=50, linewidth=2);
#co2 = contour!(ax, x, y, imag(Wmateval'), levels=20, color=:turbo,linewidth=2);
#lines!(xv[1:NB], yv[1:NB], color = :tomato)
#lines!(xv[NB+1:2*NB], yv[NB+1:2*NB], color = :tomato)
#plot!(xv,yv,marker=(:circle,5))
#arc!((0,0), 1, 0, 2pi, color=:black, linewidth=1);
Colorbar(fig[1,2], co);
save("sandbox/trefoils.png", fig);

