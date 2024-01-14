module ConjugateHeatTransfer

using ExportAll
using ApproxFun
using SpecialFunctions
using BlockArrays
using DoubleExponentialFormulas

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

default_quadrature(f, a, b) = quadde(f, a, b; atol=1e-10, rtol=1e-10)[1];

######################
## Layer potentials ##
######################

"""
T(dφ,dψ,s,f) = int_{-s,s} exp((dφ-z)/2) * K0(sqrt((dφ-z)^2 + dψ^2)/2) * f(z) / sqrt(s^2 - z^2) dz
"""
kernel(dφ, dψ) = exp(dφ/2) * besselk(0, sqrt(dφ^2 + dψ^2)/2)

function SingleLayer(dφ, dψ, s, f; quadrature=default_quadrature)
    # int_{-s,s} K(dφ-z,dψ) f(z) / sqrt(s^2 - z^2) dz
    integrand(z) = kernel(dφ-z,dψ) * f(z) / sqrt(s^2 - z^2)
    integral = quadrature(integrand, -s, s)
    return integral
end

function SplitSingleLayer(dφ, dψ, s, f; quadrature=default_quadrature)
    if (abs(dφ) >= s)
        # Single integral
        # Align kernel singularity at Z = z-dφ = 0
        # int_{-s-dφ,s-dφ} K(-Z,dψ) f(Z+dφ) / sqrt(s^2 - (Z+dφ)^2) dZ
        integrand(Z) = kernel(-Z,dψ) * f(Z+dφ) / sqrt(abs(s^2 - (Z+dφ)^2))
        integral = quadrature(integrand, -s-dφ, s-dφ)
    else
        # Split integral
        # Rescale so left endpoint singularity is at q = Z/(s+dφ) = -1
        # int_{-1,0} K(-q*(s+dφ),dψ) f(q*(s+dφ)+dφ) / sqrt(s^2 - (q*(s+dφ)+dφ)^2) (s+dφ) dq
        # Rescale so right endpoint singularity is at q = Z/(s-dφ) = 1
        # int_{0, 1} K(-q*(s-dφ),dψ) f(q*(s-dφ)+dφ) (s-dφ) / sqrt(s^2 - (q*(s-dφ)+dφ)^2) dq
        integrand1(q) = kernel(-q*(s+dφ),dψ) * f(q*(s+dφ)+dφ) * (s+dφ) / sqrt(s^2 - (s*q+(1+q)*dφ)^2)
        integrand2(q) = kernel(-q*(s-dφ),dψ) * f(q*(s-dφ)+dφ) * (s-dφ) / sqrt(s^2 - (s*q+(1-q)*dφ)^2)
        integral1 = quadrature(integrand1, -1, 0)
        integral2 = quadrature(integrand2, 0, 1)
        integral = integral1 + integral2
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

function EvaluateT(φ, ψ, panels, f; quadrature=default_quadrature)
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