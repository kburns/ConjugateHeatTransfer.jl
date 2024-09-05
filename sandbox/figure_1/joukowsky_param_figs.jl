

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
zc = 0
bodyshapes = [θ -> exp(im*θ),
              θ -> exp(im*θ) * (1 + 0.3*cos(2*θ-1) + 0.1*sin(5*θ))];

for (nb, bodyshape) in enumerate(bodyshapes)

    # Define bodies
    bodies = [DirichletBody(zc, θ->zc+bodyshape(θ))];

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
    x = collect(LinRange(-3.1, 3.1, gridsize));
    y = collect(LinRange(-3.1, 3.1, gridsize));
    z = x' .+ im*y;
    # Evaluate flow
    t0 = time();
    w = reshape(W(vec(z)), size(z));
    println("Flow evaluated (", round(time()-t0, digits=3), " s)");
    w[abs.(z) .< 0.2] .= 0
    φ = real(w);
    ψ = imag(w);

    # Plot
    colors = ColorSchemes.seaborn_colorblind6
    φ_levels = collect(LinRange(-3, 3, 19)) * (φmax - φmin) / 4 .+ (φmin + φmax) / 2
    ψ_levels = collect(LinRange(-3, 3, 19)) .+ ψmean
    fig = Figure(size=(600, 600));
    ax = Axis(fig[1,1], aspect=DataAspect());
    contour!(ax, x, y, φ', levels=φ_levels, color=(colors[3],0.75), linewidth=3);
    contour!(ax, x, y, ψ', levels=ψ_levels, color=(colors[1],0.75), linewidth=3);
    for body in bodies
        boundary = body.zθ.(LinRange(0, 2π, 100))
        poly!(ax, Point2.(real(boundary), imag(boundary)), color=:white)
        poly!(ax, Point2.(real(boundary), imag(boundary)), color=(:black,0.25))
        lines!(ax, real(boundary), imag(boundary), color=:black, linewidth=6)
    end
    hidedecorations!(ax)
    hidespines!(ax)
    save("sandbox/figure_1/joukowsky_z_$nb.pdf", fig);
end


# Plot W slit
gridsize = 200;
x = collect(LinRange(-3.1, 3.1, gridsize));
y = collect(LinRange(-3.1, 3.1, gridsize));
z = x' .+ im*y;
colors = ColorSchemes.seaborn_colorblind6
φ_levels = collect(LinRange(-3, 3, 19))
ψ_levels = collect(LinRange(-3, 3, 19))
fig = Figure(size=(600, 600));
ax = Axis(fig[1,1], aspect=DataAspect());
contour!(ax, x, y, real(z)', levels=φ_levels, color=(colors[3],0.75), linewidth=3);
contour!(ax, x, y, imag(z)', levels=ψ_levels, color=(colors[1],0.75), linewidth=3);
for body in bodies
    boundary = body.zθ.(LinRange(0, 2π, 100))
    lines!(ax, [-2, 2], [0, 0], color=:black, linewidth=6)
end
hidedecorations!(ax)
hidespines!(ax)
save("sandbox/figure_1/joukowsky_w.pdf", fig);

