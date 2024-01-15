module ConjugateHeatTransfer

using ExportAll
using ApproxFun
using SpecialFunctions
using BlockArrays
using QuadGK

#################
## Quadratures ##
#################

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

default_quadrature(f, a, b) = quadgk(f, a, b; atol=1e-10, rtol=1e-10)[1];
default_quadrature(f, a, b, c) = quadgk(f, a, b, c; atol=1e-10, rtol=1e-10)[1];

######################
## Layer potentials ##
######################

"""
T(dφ,dψ,s,f) = int_{-s,s} exp((dφ-z)/2) * K0(sqrt((dφ-z)^2 + dψ^2)/2) * f(z) / sqrt(s^2 - z^2) dz
"""
kernel(dφ, dψ) = exp(dφ/2) * besselk(0, sqrt(dφ^2 + dψ^2)/2)

function SingleLayer(dφ, dψ, s, f; quadrature=default_quadrature)
    # int_{-s,s} K(dφ-z,dψ) f(z) / sqrt(s^2 - z^2) dz
    # int_{0,π} K(dφ-s*cos(q),dψ) f(s*cos(q)) dq
    integrand(q) = kernel(dφ-s*cos(q),dψ) * f(s*cos(q))
    integral = quadrature(integrand, 0, π)
    return integral
end

function SplitSingleLayer(dφ, dψ, s, f; dψ_split=0, quadrature=default_quadrature)
    # int_{-s,s} K(dφ-z,dψ) f(z) / sqrt(s^2 - z^2) dz
    # int_{0,π} K(dφ-s*cos(q),dψ) f(s*cos(q)) dq
    integrand(q) = kernel(dφ-s*cos(q),dψ) * f(s*cos(q))
    if (abs(dφ) >= s) || (abs(dψ) > dψ_split)
        # Single integral
        integral = quadrature(integrand, 0, π)
    else
        # Split integral
        q0 = acos(dφ/s)
        integral = quadrature(integrand, 0, q0, π)
    end
    return integral
end

############
## Panels ##
############

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

function InteractionMatrix(source_panel, target_panel; quadrature=default_quadrature)
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

function SystemMatrix(panels; quadrature=default_quadrature)
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

function EvaluatePanel(φ, ψ, panel, f; quadrature=default_quadrature)
    dφ = φ .- panel.φ0
    dψ = ψ .- panel.ψ0
    return SplitSingleLayer.(dφ, dψ, panel.s, f; quadrature=quadrature)
end

function EvaluateSystem(φ, ψ, panels, f; quadrature=default_quadrature)
    return sum(EvaluatePanel(φ, ψ, panels[i], f[i]; quadrature=quadrature) for i = eachindex(panels))
end

@exportAll()

end # module ConjugateHeatTransfer