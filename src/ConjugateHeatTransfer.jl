module ConjugateHeatTransfer

using ExportAll
using ApproxFun
using SpecialFunctions
using BlockArrays

function AdaptiveQuadrature(f, a, b)
    F = Fun(f, a..b)
    return sum(F)
end

function BuildInterpolant(space, values)
    return Fun(space, ApproxFun.transform(space, values))
end

function ClenshawCurtisQuadrature(f, a, b, N)
    space = Chebyshev(a..b)
    values = f.(points(space, N))
    F = BuildInterpolant(space, values)
    return sum(F)
end

"""
T(dφ,dψ,s,f) = int_{-s,s} exp((dφ-z)/2) * K0(sqrt((dφ-z)^2 + dψ^2)/2) * f(z) dz
"""
function SingleLayer(dφ, dψ, s, f; quadrature=AdaptiveQuadrature)
    F(z) = f(z) * exp((dφ-z)/2) / sqrt(s^2 - z^2)
    integrand(z) = besselk(0, sqrt((dφ-z)^2 + dψ^2)/2) * F(z)
    integral = quadrature(integrand, -s, s)
    return integral
end

function SplitSingleLayer(dφ, dψ, s, f; quadrature=AdaptiveQuadrature)
    F(z) = f(z) * exp((dφ-z)/2) / sqrt(s^2 - z^2)
    integrand(z) = besselk(0, sqrt((dφ-z)^2 + dψ^2)/2) * F(z)
    if abs(dφ) < s
        # Split integral
        integral1 = quadrature(integrand, -s, dφ)
        integral2 = quadrature(integrand, dφ, s)
        integral = integral1 + integral2
    else
        # Single integral
        integral = quadrature(integrand, -s, s)
    end
    return integral
end

struct panel
    φ0::Float64
    ψ0::Float64
    s::Float64
    N::Int64
    space::Chebyshev
    nodes::Vector{Float64}
end

function panel(φ0, ψ0, s, N)
    space = Chebyshev(-s..s)
    nodes = points(space, N)
    return panel(φ0, ψ0, s, N, space, nodes)
end

function CardinalFunction(space, N, n)
    values = zeros(N)
    values[n] = 1
    return BuildInterpolant(space, values)
end

function InteractionMatrix(source_panel, target_panel; quadrature=AdaptiveQuadrature)
    Ns = source_panel.N
    Nt = target_panel.N
    M = zeros(Nt, Ns)
    for ns = 1:Ns
        fs = CardinalFunction(source_panel.space, Ns, ns)
        for nt = 1:Nt
            dφ = target_panel.φ0 - source_panel.φ0 + target_panel.nodes[nt]
            dψ = target_panel.ψ0 - source_panel.ψ0
            M[nt,ns] = SplitSingleLayer(dφ, dψ, source_panel.s, fs; quadrature=quadrature)
        end
    end
    return M
end

function SystemMatrix(panels; quadrature=AdaptiveQuadrature)
    N = length(panels)
    sizes = [panels[i].N for i = 1:N]
    M = BlockArray{Float64}(undef_blocks, sizes, sizes)
    for ns = 1:N
        for nt = 1:N
            M[Block(nt,ns)] = InteractionMatrix(panels[ns], panels[nt]; quadrature=quadrature)
        end
    end
    return M
end

function EvaluateT(φ, ψ, panels, f; quadrature=AdaptiveQuadrature)
    T = 0
    for i = eachindex(panels)
        dφ = φ .- panels[i].φ0
        dψ = ψ .- panels[i].ψ0
        T = T .+ SplitSingleLayer.(dφ, dψ, panels[i].s, f[i]; quadrature=quadrature)
    end
    return T
end

@exportAll()

end # module ConjugateHeatTransfer