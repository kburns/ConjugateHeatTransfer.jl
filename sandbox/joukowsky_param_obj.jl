

using Revise
using ConjugateHeatTransfer
using CairoMakie
using LinearAlgebra
using QuadGK
using PolygonOps
using StaticArrays
using ColorSchemes


# Parameters
U = 1; # free stream velocity

# Bodies
bodyshape(θ) = exp(im*θ) * (1 + 0.2*cos(2*θ-0.5) + 0.05*sin(5*θ));
zc1 = 0
centers = [zc1];
bodies = [DirichletBody(zc, θ->zc+bodyshape(θ)) for zc in centers];

# Solve potential flow
t0 = time();
W = adaptive_laurent_potential_flow(bodies, U; atol=1e-6)
println("Solved potential flow (", round(time()-t0, digits=3), " s)");

w = W.(bodies[1].zθ.(LinRange(0, 2π, 1000)))
φmin = minimum(real(w))
φmax = maximum(real(w))
ψmean = sum(imag(w)) / length(w)

# Evaluate on a grid
gridsize = 200;
x = collect(LinRange(-3, 3, gridsize));
y = collect(LinRange(-3, 3, gridsize));
z = x' .+ im*y;
# Evaluate flow
t0 = time();
w = reshape(W(vec(z)), size(z));
println("Flow evaluated (", round(time()-t0, digits=3), " s)");
φ = real(w);
ψ = imag(w);

# Plot
colors = ColorSchemes.seaborn_colorblind6
φ_levels = collect(LinRange(-3, 3, 19)) * (φmax - φmin) / 4 .+ (φmin + φmax) / 2
ψ_levels = collect(LinRange(-3, 3, 19)) .+ ψmean
fig = Figure(size=(600, 600));
ax = Axis(fig[1,1], aspect=DataAspect());
co = contour!(ax, x, y, φ', levels=φ_levels, color=colors[3], linewidth=3);
co = contour!(ax, x, y, ψ', levels=ψ_levels, color=colors[1], linewidth=3);
for body in bodies
    boundary = body.zθ.(LinRange(0, 2π, 100))
    poly!(ax, Point2.(real(boundary), imag(boundary)), color=(:grey))
    lines!(ax, real(boundary), imag(boundary), color=:black, linewidth=4)
end
hidedecorations!(ax)
hidespines!(ax)
save("sandbox/joukowsky_obj.pdf", fig);

