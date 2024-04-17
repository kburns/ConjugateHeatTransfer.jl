

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


function DirichletBody(zc, zθ)
    Tθ = θ -> 1
    return DirichletBody(zc, zθ, Tθ, error, error, error, NaN, NaN, NaN)
end


"""Reparametrize body to be locally Joukowsky using the flow map."""
function reparametrize!(body::DirichletBody, W; atol=0, N=Inf)
    φmin, φmax, ψ, θη = joukowsky_parametrization(W, body.zθ; atol=atol, N=N)
    body.θη = θη
    body.zη = build_interpolant(η -> body.zθ(body.θη(η)), Fourier(0..2π); atol=atol, N=N, label="zη")
    body.Tη = build_interpolant(η -> body.Tθ(body.θη(η)), Fourier(0..2π); atol=atol, N=N, label="Tη")
    body.φmin = φmin
    body.φmax = φmax
    body.ψ = ψ
end


"""Reparametrize body to be locally Joukowsky using the flow map."""
function reparametrize!(bodies::Vector{DirichletBody}, W; atol=0, N=Inf)
    for body in bodies
        reparametrize!(body, W; atol=atol, N=N)
    end
end


"""Find Joukowsky parametrization of a curve given a flow map."""
function joukowsky_parametrization(W, z; atol=0, N=Inf)
    # Approximate W(z(θ)) using ApproxFun to find extrema
    w = build_interpolant(θ->W(z(θ)), Fourier(0..2π); atol=atol, N=N, label="w")
    wr = real(w)
    φmin, θmin = findmin(wr)
    φmax, θmax = findmax(wr)
    ψ = sum(imag(w)) / (2π)
    # Find dθ(η) such that θ(η) = η + dθ(η)
    dθ = find_dθ_roots(wr, θmin, θmax, φmin, φmax; atol=atol, N=N)
    #dθ = find_dθ_fourier_least_squares(wr, θmin, θmax, φmin, φmax; atol=atol)
    θη(η) = η + dθ(η)
    # Return limits and Joukowsky parametrization
    return φmin, φmax, ψ, θη
end


function find_dθ_roots(wr, θmin, θmax, φmin, φmax; atol=1e-10, N=Inf)
    if θmin < θmax
        θmin += 2π
    end
    # Invert θ(η)-η pointwise via rootfinding
    function dθ(η)
        φ = (φmin + φmax) / 2 + (φmax - φmin) / 2 * cos(η)
        F(θ) = wr(θ) - φ
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
    # Approximate dθ using ApproxFun for performance
    return build_interpolant(dθ, Fourier(0..2π); atol=atol, N=N, label="dθ")
end


function find_dθ_fourier_least_squares(wr, θmin, θmax, φmin, φmax; atol=1e-10)
    function η(θ)
        if θ ≈ θmax
            return 0
        elseif θ ≈ θmin
            return π
        end
        cos_η = (wr(θ) - (φmin + φmax) / 2) / ((φmax - φmin) / 2)
        if θmax < θmin
            if θ < θmax
                return -acos(cos_η)
            elseif θ < θmin
                return acos(cos_η)
            else
                return 2π - acos(cos_η)
            end
        else
            if θ < θmin
                return -2π + acos(cos_η)
            elseif θ < θmax
                return - acos(cos_η)
            else
                return acos(cos_η)
            end
        end
    end
    function generate_samples(M)
        θv = fourier_grid(M)
        ηv = η.(θv)
        return (ηv, θv-ηv)
    end
    return adaptive_fourier_least_squares(generate_samples; atol=atol)
end
