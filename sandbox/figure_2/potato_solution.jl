

using Revise
using ConjugateHeatTransfer
using CairoMakie
using LinearAlgebra
using QuadGK
using PolygonOps
using StaticArrays


# Parameters
U = 1; # free stream velocity
N_bie = 32; # number of points per slit for BIE
solve_quad(args...) = quadgk(args...; atol=1e-10, rtol=1e-10)[1];
plot_quad(args...) = quadgk(args...; atol=1e-5, rtol=1e-5)[1];

# bodyshapes
zc = 0;
bodyshape(θ) = exp(im*θ) * (1 + 0.3*cos(2*θ-1) + 0.1*sin(5*θ));
bodytemp(θ) = 1 + 0.5*sin(6*θ);
bodies = [DirichletBody(zc, bodyshape, bodytemp)]

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
x = collect(LinRange(-3.1, 3.1, gridsize));
y = collect(LinRange(-3.1, 3.1, gridsize));
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


# Evaluate in w domain
gridsize = 200
φw = collect(LinRange(-5.1, 5.1, gridsize));
ψw = collect(LinRange(-5.1, 5.1, gridsize));
ww = φw' .+ im*ψw;
Tw = EvaluateSystem(real(ww), imag(ww), panels; quadrature=plot_quad);


# Plot
fig = Figure(size=(700, 300));
set_theme!(theme_latexfonts());
θp = LinRange(0, 2π, 1000);

# ax = Axis(fig[1,1], aspect=DataAspect(), title=L"Streamfunction $\psi(z)$");
# co = contourf!(ax, x, y, ψ', levels=20);
# for body in bodies
#     boundary = body.zθ.(θp)
#     lines!(ax, real(boundary), imag(boundary), color=:black, linewidth=4)
# end
# Colorbar(fig[1,2], co);
T_levels = LinRange(0, 1.5, 20);

ax = Axis(fig[1,1], aspect=DataAspect(), title=L"Temperature $T(w)$");
co = contourf!(ax, φw, ψw, Tw', levels=T_levels, nan_color=:black);
# contour!(ax, φw, ψw, real(ww), levels=ψ_levels, color=(:white,0.75), linewidth=0.5);
colorrange = (minimum(co._computed_levels.val), maximum(co._computed_levels.val));
for body in bodies
    lines!(ax, [body.φmin, body.φmax], [0, 0], color=:black, linewidth=2)
    # φ0 = (body.φmin + body.φmax) / 2
    # s = (body.φmax - body.φmin) / 2
    # boundary = φ0 .+ s * cos.(θp) .+ 0.05im*((θp .< π) - (θp .> π))
    # temp = body.Tη.(θp)
    # #poly!(ax, [(real(p), imag(p)) for p in boundary], color=:gray)
    # lines!(ax, real(boundary), imag(boundary), color=:black, linewidth=2, colorrange=colorrange)
    # lines!(ax, real(boundary), imag(boundary), color=temp, linewidth=1, colorrange=colorrange, smooth=true)
    # # Replot with shift to get rid of color artifacts
    # boundary = φ0 .+ s * cos.(θp .+ π/1000) .+ 0.1im*((θp .< π) - (θp .> π))
    # temp = body.Tη.(θp.+π/1000)
    # lines!(ax, real(boundary), imag(boundary), color=temp, linewidth=1, colorrange=colorrange, smooth=true)
end
xlims!(ax, -5.1, 5.1)
ylims!(ax, -5.1, 5.1)
# Colorbar(fig[1,2], co);

ax = Axis(fig[1,2], aspect=DataAspect(), title=L"Temperature $T(z)$");
poly!(ax, [(-3, -3), (3, -3), (3, 3), (-3, 3)], color=:gray, rasterize=true)
co = contourf!(ax, x, y, T', levels=T_levels, nan_color=:black);
colorrange = (minimum(co._computed_levels.val), maximum(co._computed_levels.val));

ψ_levels = collect(LinRange(-3, 3, 13)) .+ bodies[1].ψ
contour!(ax, x, y, ψ', levels=ψ_levels, color=(:white,0.75), linewidth=0.5);

for body in bodies
    boundary = body.zθ.(θp)
    temp = body.Tθ.(θp)
    #poly!(ax, [(real(p), imag(p)) for p in boundary], color=:gray)
    lines!(ax, real(boundary), imag(boundary), color=:black, linewidth=4, colorrange=colorrange)
    lines!(ax, real(boundary), imag(boundary), color=temp, linewidth=2, colorrange=colorrange, smooth=true)
    # Replot with shift to get rid of color artifacts
    boundary = body.zθ.(θp.+π/1000)
    temp = body.Tθ.(θp.+π/1000)
    lines!(ax, real(boundary), imag(boundary), color=temp, linewidth=2, colorrange=colorrange, smooth=true)
end
xlims!(ax, -3.1, 3.1)
ylims!(ax, -3.1, 3.1)

Colorbar(fig[1,3], co);

save("sandbox/figure_2/potato.pdf", fig);

