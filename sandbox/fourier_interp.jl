
w = -build_interpolant(θ->W(b1.zθ(θ)), Fourier(0..2π); label="w")

φmin, θmin = findmin(real(w))
φmax, θmax = findmax(real(w))
ψ = sum(imag(w)) / (2π)

θv = LinRange(0, 2π, 100)
cos_η = (real(w.(θv)) .- (φmin + φmax) / 2) / ((φmax - φmin) / 2)

function η(θ)
    if θ ≈ θmax
        return 0
    elseif θ ≈ θmin
        return π
    end
    cos_η = (real(w(θ)) - (φmin + φmax) / 2) / ((φmax - φmin) / 2)
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

dη(θ) = η(θ) - θ


function DFT_matrix(x, n)
    D = ones(length(x), 1+2*n)
    D[:,1:2:end] = cos.(x .* (0:n)')
    D[:,2:2:end] = sin.(x .* (1:n)')
    return D
end


function fourier_least_squares(x, f; N_step=16, N_max=1024)
    N = N_step
    while N <= N_max
        D = DFT_matrix(x, N)
        fhat = (D' * D) \ (D' * f)
        if norm(D * fhat - f, Inf) < 1e-10
            return fhat
        end
        N += N_step
    end
    fhat = (D' * D) \ (D' * f)
    return fhat
end

