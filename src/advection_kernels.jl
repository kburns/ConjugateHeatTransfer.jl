

using SpecialFunctions


"""
D(φ,ψ) = sqrt(φ^2 + ψ^2)
G(φ,ψ) = exp(φ/2) K0(D(φ,ψ)/2)
H(φ,ψ) = - (1/2) exp(φ/2) K1(D(φ,ψ)/2) / D(φ,ψ)

dG = (1/2) G dφ + H (φ dφ + ψ dψ)
   = ((1/2) G + H φ) dφ + H ψ dψ
dG/dφ = (1/2) G + H φ
dG/dψ = H ψ

S[f](φ,ψ) = int G(φ-z,ψ) f(z) dz
D[f](φ,ψ) = int H(φ-z,ψ) ψ f(z) dz
D[f](φ,0+) = - π f(φ)
D[f](φ,0-) = π f(φ)

T(φ,ψ) = S[f](φ,ψ) + D[g](φ,ψ)
T(φ,0+) = S[f](φ,0) - π g(φ)
T(φ,0-) = S[f](φ,0) + π g(φ)
g(φ) = (T(φ,0-) - T(φ,0+)) / (2π)
S[f](φ,0) = (T(φ,0+) + T(φ,0-)) / 2
"""


D(φ,ψ) = sqrt(φ^2 + ψ^2)
G(φ,ψ) = exp(φ/2) * besselk(0, D(φ,ψ)/2)
H(φ,ψ) = - (1/2) * exp(φ/2) * besselk(1, D(φ,ψ)/2) / D(φ,ψ)


function SingleLayer(dφ, dψ, s, f; quadrature=default_quadrature)
    # int_{-s,s} G(dφ-z,dψ) f(z) / sqrt(s^2 - z^2) dz
    # int_{0,π} G(dφ-s*cos(q),dψ) f(s*cos(q)) dq
    integrand(q) = G(dφ-s*cos(q),dψ) * f(s*cos(q))
    integral = quadrature(integrand, 0, π)
    return integral
end


function SplitSingleLayer(dφ, dψ, s, f; dψ_split=0, quadrature=default_quadrature)
    # int_{-s,s} G(dφ-z,dψ) f(z) / sqrt(s^2 - z^2) dz
    # int_{0,π} G(dφ-s*cos(q),dψ) f(s*cos(q)) dq
    integrand(q) = G(dφ-s*cos(q),dψ) * f(s*cos(q))
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


function DoubleLayer(dφ, dψ, s, g; dψ_split=1e-10, quadrature=default_quadrature)
    # int_{-s,s} H(dφ-z,dψ) dψ g(z) sqrt(s^2 - z^2) dz
    # int_{0,π} H(dφ-s*cos(q),dψ) dψ g(s*cos(q)) s^2 sin^2(q) dq
    integrand(q) = H(dφ-s*cos(q),dψ) * dψ * g(s*cos(q)) * s^2 * sin(q)^2
    if (abs(dψ) > dψ_split)
        if (abs(dφ) >= s)
            # Single integral
            integral = quadrature(integrand, 0, π)
        else
            # Split integral
            q0 = acos(dφ/s)
            integral = quadrature(integrand, 0, q0, π)
        end
    else
        integral = - sin(dψ) * π * g(dφ)
    end
    return integral
end

