

using ApproxFun
using Roots


mutable struct DirichletBody
    zc::Complex
    zθ::Function
    Tθ::Function
    θη::Function
    zη::Function
    Tη::Function
    φmin::Real
    φmax::Real
    ψ::Real
end


function DirichletBody(zc, zθ, Tθ)
    return DirichletBody(zc, zθ, Tθ, error, error, error, NaN, NaN, NaN)
end


function reparametrize!(body::DirichletBody, W; N=Inf)
    φmin, φmax, ψ, θ_adj = joukowsky_adjustment(W, body.zθ; N=N)
    body.θη = η -> η + θ_adj(η)
    body.zη = η -> body.zθ(body.θη(η))
    body.Tη = η -> body.Tθ(body.θη(η))
    body.φmin = φmin
    body.φmax = φmax
    body.ψ = ψ
end


function reparametrize!(bodies::Vector{DirichletBody}, W; N=Inf)
    for body in bodies
        reparametrize!(body, W; N=N)
    end
end


function joukowsky_adjustment(W, z; N=Inf)
    S = Fourier(0..2π)
    # Approximate W(z(θ)) using ApproxFun
    w = build_interpolant(θ->W(z(θ)), S; N=N)
    # Find min and max φ and θ
    φmin, θmin = findmin(real(w))
    φmax, θmax = findmax(real(w))
    ψ = sum(imag(w)) / (2π)
    if θmin < θmax
        θmin += 2π
    end
    # Invert θ(η)-η pointwise via rootfinding
    function θ_adj(η)
        φ = (φmin + φmax) / 2 + (φmax - φmin) / 2 * cos(η)
        F(θ) = real(w(θ)) - φ
        if (η ≈ 0) || (η ≈ 2π)
            θ = θmax
        elseif η ≈ π
            θ = θmin
        elseif η < π
            # Pad brackets by 10eps() to avoid rootfinding issues
            θ = find_zero(F, (θmax+10eps(), θmin-10eps()), Bisection())
        else
            # Pad brackets by 10eps() to avoid rootfinding issues
            θ = find_zero(F, (θmin+10eps(), θmax+2π-10eps()), Bisection())
        end
        return θ - η
    end
    # Approximate θ_adj using ApproxFun
    θ_adj = build_interpolant(θ_adj, S; N=N)
    return φmin, φmax, ψ, θ_adj
end

