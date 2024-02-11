

using ApproxFun


function build_interpolant(func, space; atol=0, N=Inf, label=nothing)
    if N == Inf
        # Test a few random points since ApproxFun seems to choke when f≈0
        a = leftendpoint(domain(space))
        b = rightendpoint(domain(space))
        test_points = a .+ (b-a)*rand(5)
        if isapprox(func.(test_points), zeros(5), atol=1e-10)
            f = Fun(x->func(x)+1, space) - 1
        else
            f = Fun(func, space)
        end
        if atol > 0
            f = ApproxFun.chop(f, atol)
        end
        if !isnothing(label)
            println("  ", label, " coeffs: ", ncoefficients(f))
        end
        return f
    else
        values = func.(points(space, N))
        return Fun(space, ApproxFun.transform(space, values))
    end
end


function interpolate_values(space, values)
    return Fun(space, ApproxFun.transform(space, values))
end


function fourier_grid(N)
    return collect(LinRange(0, 2π, N+1)[1:end-1])
end


function DFT_matrix(x::Vector{Float64}, K::Int)
    D = ones(length(x), 1+2*K)
    D[:,1:2:end] = cos.(x .* (0:K)')
    D[:,2:2:end] = sin.(x .* (1:K)')
    return D
end


function fourier_least_squares(x::Vector{Float64}, y, K::Int)
    D = DFT_matrix(x, K)
    yhat = (D' * D) \ (D' * y)
    f = Fun(Fourier(0..2π), yhat)
    return f
end


function adaptive_fourier_least_squares(generate_samples; atol=1e-10, K_steps=16, K_max=1024)
    K = K_steps
    while K <= K_max
        # Least square fit with 2x constraints
        M = 4*K
        x, y = generate_samples(M)
        f = fourier_least_squares(x, y, K)
        # Check error with 8x resampling
        x, y = generate_samples(8*M)
        error = norm(y - f.(x), Inf)
        println("    K = ", K, ", error = ", error)
        if error < atol
            println("  K = ", K)
            return f
        end
        K += K_steps
    end
    error("Failed to converge with K_max = ", K_max)
end

