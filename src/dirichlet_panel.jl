

using BlockArrays


mutable struct DirichletPanel
    body::DirichletBody
    φ0::Float64
    ψ0::Float64
    s::Float64
    space::Chebyshev
    T_top::Function
    T_bot::Function
    SL_density::Function
    DL_density::Function
    flux::Float64
end


function DirichletPanel(body)
    φ0 = (body.φmin + body.φmax) / 2
    ψ0 = body.ψ
    s = (body.φmax - body.φmin) / 2
    space = Chebyshev(-s..s)
    T_top(z) = body.Tη(acos(z/s))
    T_bot(z) = body.Tη(-acos(z/s))
    return DirichletPanel(body, φ0, ψ0, s, space, T_top, T_bot, error, error, NaN)
end


function solve_DL_densities!(panels::Vector{DirichletPanel}; Pe=1, atol=0, N=Inf)
    for p in panels
        g(z) = -(p.T_top(z) - p.T_bot(z)) / sqrt(p.s^2 - z^2)
        p.DL_density = build_interpolant(g, p.space; atol=atol, N=N, label="g")
        #gg(θ) = (p.body.Tη(θ) - p.body.Tη(-θ))
        #build_interpolant(gg, Fourier(0..2π); atol=atol, N=N, label="gθ")
    end
end


function solve_SL_densities!(panels::Vector{DirichletPanel}, N::Int; Pe=1, quadrature=default_quadrature)
    Ns = N
    Nt = 2*N
    # Build RHS from mean panel temperatures
    nodes = points(Chebyshev(), Nt)
    RHS = [(p.T_top.(p.s*nodes)+p.T_bot.(p.s*nodes))/2 for p in panels]
    # Subtract off DL contributions
    for i = 1:length(panels)
        for j = 1:length(panels)
            if i == j
                continue
            end
            RHS[i] -= DoubleLayer.(panels[i].φ0 .+ panels[i].s*nodes .- panels[j].φ0, panels[i].ψ0 - panels[j].ψ0, panels[j].s, panels[j].DL_density; Pe=Pe, quadrature=quadrature)
        end
    end
    # Solve for SL densities
    M = SystemMatrix(panels, Ns, Nt; Pe=Pe, quadrature=quadrature)
    println("size(M) = ", size(M))
    println("cond(M) = ", cond(M))
    f = M \ vcat(RHS...)
    Ns = [N for p in panels]
    f = split(f, Ns)
    for i = 1:length(panels)
        panels[i].SL_density = interpolate_values(panels[i].space, f[i])
    end
    # Compute flux
    compute_flux!(panels)
end


function compute_flux!(panels::Vector{DirichletPanel})
    for p in panels
        f(s) = p.SL_density(s) / sqrt(p.s^2 - s^2)
        f_interp = build_interpolant(f, JacobiWeight(-0.5, -0.5, p.space))
        p.flux = sum(f_interp)
    end
end


function solve_densities!(panels::Vector{DirichletPanel}, N::Int; Pe=1, atol=0, quadrature=default_quadrature)
    solve_DL_densities!(panels; Pe=Pe, atol=atol)
    solve_SL_densities!(panels, N; Pe=Pe, quadrature=quadrature)
end


function split(a, n)
    out = []
    n0 = 1
    for ni in n
        n1 = n0 + ni - 1
        push!(out, a[n0:n1])
        n0 = n1 + 1
    end
    return out
end


function CardinalFunction(space, N, n)
    values = zeros(N)
    values[n] = 1
    return interpolate_values(space, values)
end


function InteractionMatrix(source_panel, target_panel, Ns, Nt; Pe=1, quadrature=default_quadrature)
    nodes = points(Chebyshev(), Nt)
    M = zeros(Nt, Ns)
    for ns = 1:Ns
        fs = CardinalFunction(source_panel.space, Ns, ns)
        for nt = 1:Nt
            dφ = target_panel.φ0 - source_panel.φ0 + target_panel.s*nodes[nt]
            dψ = target_panel.ψ0 - source_panel.ψ0
            M[nt,ns] = SplitSingleLayer(dφ, dψ, source_panel.s, fs; Pe=Pe, quadrature=quadrature)
        end
    end
    return M
end


function SystemMatrix(panels, Ns, Nt; Pe=1, quadrature=default_quadrature)
    M = BlockArray{Float64}(undef_blocks, [Nt for i in eachindex(panels)], [Ns for i in eachindex(panels)])
    for ns in eachindex(panels)
        for nt in eachindex(panels)
            M[Block(nt,ns)] = InteractionMatrix(panels[ns], panels[nt], Ns, Nt; Pe=Pe, quadrature=quadrature)
        end
    end
    return M
end


function EvaluatePanel(φ, ψ, panel; Pe=1, quadrature=default_quadrature)
    dφ = φ .- panel.φ0
    dψ = ψ .- panel.ψ0
    f = panel.SL_density
    g = panel.DL_density
    if f == 0
        Sf = 0
    else
        Sf = SplitSingleLayer.(dφ, dψ, panel.s, f; Pe=Pe, quadrature=quadrature)
    end
    if g == 0
        Dg = 0
    else
        Dg = DoubleLayer.(dφ, dψ, panel.s, g; Pe=Pe, quadrature=quadrature)
    end
    return Sf .+ Dg
end


function EvaluateSystem(φ, ψ, panels; Pe=1, quadrature=default_quadrature)
    return sum(EvaluatePanel(φ, ψ, panels[i]; Pe=Pe, quadrature=quadrature) for i = eachindex(panels))
end

