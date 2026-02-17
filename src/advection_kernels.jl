

using SpecialFunctions


"""
D(φ,ψ) = sqrt(φ^2 + ψ^2)
G(φ,ψ) = (1/2π) exp(Pe*φ/2) K0(Pe*D(φ,ψ)/2)
H(φ,ψ) = -(Pe/4π) exp(Pe*φ/2) K1(Pe*D(φ,ψ)/2) / D(φ,ψ)

dG = (Pe/2) G dφ + H (φ dφ + ψ dψ)
   = ((Pe/2) G + H φ) dφ + H ψ dψ
dG/dφ = (Pe/2) G + H φ
dG/dψ = H ψ

G0(φ,ψ) = -(1/2π) (log(Pe*D(φ,ψ)/4) + γ)
H0(φ,ψ) = -(1/4π) / D(φ,ψ)^2

dG0 = -(1/4π) / D^2 * (φ dφ + ψ dψ)
dG0/dψ = -(1/4π) / D^2 * ψ = H0 ψ

S[f](φ,ψ) = int G(φ-φ',ψ) f(φ') dφ'
D[g](φ,ψ) = int H(φ-φ',ψ) ψ g(φ') dφ'
D[g](φ,0+) = -(1/2) g(φ)
D[g](φ,0-) = +(1/2) g(φ)

T(φ,ψ) = S[f](φ,ψ) + D[g](φ,ψ)
T(φ,0+) = S[f](φ,0) - (1/2) g(φ)
T(φ,0-) = S[f](φ,0) + (1/2) g(φ)
g(φ) = T(φ,0-) - T(φ,0+)
S[f](φ,0) = (T(φ,0+) + T(φ,0-)) / 2

dT/dψ(φ,ψ) = D[f](φ,ψ) + DD[g](φ,ψ)
dT/dψ(φ,0+) = -(1/2) f(φ) + ...
dT/dψ(φ,0-) = +(1/2) f(φ) + ...
Flux = (1/Pe) int f(φ) dφ
"""

using HypergeometricFunctions

# D(φ,ψ) = sqrt(φ^2 + ψ^2)
# G(φ,ψ,Pe) = (1/2π) * exp(Pe*φ/2) * besselk(0, Pe*D(φ,ψ)/2)
# H(φ,ψ,Pe) = - (Pe/4π) * exp(Pe*φ/2) * besselk(1, Pe*D(φ,ψ)/2) / D(φ,ψ)

function G(φ,ψ,Pe)
    D = sqrt(φ^2 + ψ^2)
    if Pe*D < 1
        return (1/2π) * exp(Pe*φ/2) * besselk(0, Pe*D/2)
    else
        return (1/2π) * exp(Pe*(φ-D)/2) * HypergeometricFunctions.U(1/2, 1, Pe*D) * sqrt(pi)
    end
end

function H(φ,ψ,Pe)
    D = sqrt(φ^2 + ψ^2)
    if Pe*D < 1
        return - (Pe/4π) * exp(Pe*φ/2) * besselk(1, Pe*D/2) / D
    else
        return - (Pe/4π) * exp(Pe*(φ-D)/2) * HypergeometricFunctions.U(3/2, 3, Pe*D) * sqrt(pi) * Pe
    end
end

function SingleLayer(φ, ψ, s, f; Pe=1, quadrature=default_quadrature)
    # int_{-s,s} G(φ-φ',ψ) f(φ') / sqrt(s^2 - φ'^2) dφ'
    # Change to trigonometric integral to remove endpoints singularities
    # int_{0,π} G(φ-s*cos(η),ψ) f(s*cos(η)) dη
    if isnan(φ) || isnan(ψ)
        return NaN
    end
    integrand(η) = G(φ-s*cos(η),ψ,Pe) * f(s*cos(η))
    integral = quadrature(integrand, 0, π)
    return integral
end


function SplitSingleLayer(φ, ψ, s, f; Pe=1, ψ_split=1e-10, quadrature=default_quadrature)
    # int_{-s,s} G(φ-φ',ψ) f(φ') / sqrt(s^2 - φ'^2) dφ'
    # Change to trigonometric integral to remove endpoints singularities
    # int_{0,π} G(φ-s*cos(η),ψ) f(s*cos(η)) dη
    if isnan(φ) || isnan(ψ)
        return NaN
    end
    integrand(q) = G(φ-s*cos(q),ψ,Pe) * f(s*cos(q))
    if (abs(φ) >= s) || (abs(ψ) > ψ_split)
        # Single integral
        integral = quadrature(integrand, 0, π)
    else
        # Split integral
        η0 = acos(φ/s)
        integral = quadrature(integrand, 0, η0, π)
    end
    return integral
end


function DoubleLayer(φ, ψ, s, g; Pe=1, ψ_split=1e-10, quadrature=default_quadrature)
    # int_{-s,s} H(φ-φ',ψ) ψ g(φ') sqrt(s^2 - φ'^2) dφ'
    # int_{0,π} H(φ-s*cos(η),ψ) ψ g(s*cos(η)) s^2 sin^2(η) dη
    if isnan(φ) || isnan(ψ)
        return NaN
    end
    integrand(η) = H(φ-s*cos(η),ψ,Pe) * ψ * g(s*cos(η)) * s^2 * sin(η)^2
    if (abs(ψ) > ψ_split)
        if (abs(φ) >= s)
            # Single integral
            integral = quadrature(integrand, 0, π)
        else
            # Split integral
            η0 = acos(φ/s)
            integral = quadrature(integrand, 0, η0, π)
        end
    else
        if (abs(φ) >= s)
            return 0
        else
            println(φ, ψ, s)
            error("shouldn't happen")
        end
    end
    return integral
end

