

using SpecialFunctions


"""
D(φ,ψ) = sqrt(φ^2 + ψ^2)
G(φ,ψ) = exp(φ/2) K0(D(φ,ψ)/2)
H(φ,ψ) = - (1/2) exp(φ/2) K1(D(φ,ψ)/2) / D(φ,ψ)

dG = (1/2) G dφ + H (φ dφ + ψ dψ)
   = ((1/2) G + H φ) dφ + H ψ dψ
dG/dφ = (1/2) G + H φ
dG/dψ = H ψ

S[f](φ,ψ) = int G(φ-φ',ψ) f(φ') dφ'
D[g](φ,ψ) = int H(φ-φ',ψ) ψ g(φ') dφ'
D[g](φ,0+) = - π g(φ)
D[g](φ,0-) = + π g(φ)

T(φ,ψ) = S[f](φ,ψ) + D[g](φ,ψ)
T(φ,0+) = S[f](φ,0) - π g(φ)
T(φ,0-) = S[f](φ,0) + π g(φ)
g(φ) = - (T(φ,0+) - T(φ,0-)) / (2π)
S[f](φ,0) = (T(φ,0+) + T(φ,0-)) / 2
"""


D(φ,ψ) = sqrt(φ^2 + ψ^2)
G(φ,ψ) = exp(φ/2) * besselk(0, D(φ,ψ)/2)
H(φ,ψ) = - (1/2) * exp(φ/2) * besselk(1, D(φ,ψ)/2) / D(φ,ψ)


function SingleLayer(φ, ψ, s, f; quadrature=default_quadrature)
    # int_{-s,s} G(φ-φ',ψ) f(φ') / sqrt(s^2 - φ'^2) dφ'
    # int_{0,π} G(φ-s*cos(η),ψ) f(s*cos(η)) dη
    integrand(η) = G(φ-s*cos(η),ψ) * f(s*cos(η))
    integral = quadrature(integrand, 0, π)
    return integral
end


function SplitSingleLayer(φ, ψ, s, f; ψ_split=0, quadrature=default_quadrature)
    # int_{-s,s} G(φ-φ',ψ) f(φ') / sqrt(s^2 - φ'^2) dφ'
    # int_{0,π} G(φ-s*cos(η),ψ) f(s*cos(η)) dη
    integrand(q) = G(φ-s*cos(q),ψ) * f(s*cos(q))
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


function DoubleLayer(φ, ψ, s, g; ψ_split=1e-10, quadrature=default_quadrature)
    # int_{-s,s} H(φ-φ',ψ) ψ g(φ') sqrt(s^2 - φ'^2) dφ'
    # int_{0,π} H(φ-s*cos(η),ψ) ψ g(s*cos(η)) s^2 sin^2(η) dη
    integrand(η) = H(φ-s*cos(η),ψ) * ψ * g(s*cos(η)) * s^2 * sin(η)^2
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
        error("shouldn't happen")
        # integral = - sign(dψ) * π * g(dφ)
    end
    return integral
end

