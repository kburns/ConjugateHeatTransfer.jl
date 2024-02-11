

using LinearAlgebra
using Statistics


"""Solver for potential flow around bodies using Laurent series."""
function adaptive_laurent_potential_flow(bodies::Vector{DirichletBody}, U::Number; atol=1e-10, N_laurent_steps=32, N_laurent_max=1024)
    N_laurent = N_laurent_steps
    while N_laurent <= N_laurent_max
        N_sample = 4 * N_laurent
        W = laurent_potential_flow(bodies, U, N_sample, N_laurent)
        error = potential_flow_boundary_error(W, bodies, 8*N_sample)
        if error < atol
            println("  N_laurent = ", N_laurent, ", error = ", round(error, sigdigits=3))
            return W
        end
        N_laurent += N_laurent_steps
    end
    error("Failed to converge")
end


"""Solve the potential flow around bodies using Laurent series."""
function laurent_potential_flow(bodies::Vector{DirichletBody}, U::Number, N_sample::Int, N_laurent::Int)
    sample_points = fourier_grid(N_sample)
    body_samples = [b.zθ.(sample_points) for b in bodies]
    body_centers = Vector{ComplexF64}([b.zc for b in bodies])
    fit_values = -imag(conj(U)*body_samples)
    d, H = fit_laurent_arnoldi(body_samples, body_centers, fit_values, N_laurent)
    W(z) = conj(U)*z + evaluate_laurent_arnoldi(d, H, body_centers, z)
    return W
end


"""Compute the error in the potential flow map on the bodies."""
function potential_flow_boundary_error(W::Function, bodies::Vector{DirichletBody}, N_sample::Int)
    # Sample bodies
    sample_points = fourier_grid(N_sample)
    body_samples = [b.zθ.(sample_points) for b in bodies]
    # Evaluate ψ on bodies
    imag_values = [imag.(W(bs)) for bs in body_samples]
    # Determine error as maximum deviation from the mean on each body
    errors = [norm(ψ .- mean(ψ), Inf) for ψ in imag_values]
    return maximum(errors)
end


"""Fit orthogonalized Laurent series using least squares."""
function fit_laurent_arnoldi(x_vec::Vector{Vector{ComplexF64}}, zc_vec::Vector{ComplexF64}, f_vec::Vector{Vector{Float64}}, NL::Int)
    NB = length(x_vec)
    x = reduce(vcat, x_vec)
    M = length(x)
    Q_full = zeros(Float64, M, NB + 2*NB*NL)
    # Add constant term to absorb ψ for each body
    i0 = 1
    for nb = 1:NB
        i1 = i0 + length(x_vec[nb])
        Q_full[i0:i1-1, nb] .=  1
        i0 = i1
    end
    # Orthogonalize the Laurent series for each body
    H_vec = Vector{Matrix{ComplexF64}}()
    for nb = 1:NB
        zc = zc_vec[nb]
        Q = zeros(ComplexF64, M, NL)
        H = zeros(ComplexF64, NL, NL-1)
        Q[:,1] .= 1 ./ (x .- zc)
        for k = 1:NL-1
            q = Q[:,k] ./ (x .- zc)
            for j = 1:k
                H[j,k] = Q[:,j]' * q / M
                q = q - H[j,k] * Q[:,j]
            end
            H[k+1,k] = norm(q) / sqrt(M)
            Q[:,k+1] = q / H[k+1,k]
        end
        push!(H_vec, H)
        # Add to the system matrix
        j0 = NB + (nb-1)*2*NL
        Q_full[:, j0+1:j0+NL] = real(Q)
        Q_full[:, j0+NL+1:j0+2*NL] = imag(Q)
    end
    # Solve the least squares problem
    f = reduce(vcat, f_vec)
    #println("  Condition number: ", cond(Q_full))
    d = Q_full \ f
    #println("  Fitting error: ", norm(Q_full*d - f, Inf))
    # Complexify the coefficients
    d_complex = zeros(ComplexF64, NB*NL)
    for nb = 1:NB
        i0 = (nb-1)*NL
        d_complex[i0+1:i0+NL] = im*d[NB+2*i0+1:NB+2*i0+NL] .+ d[NB+2*i0+NL+1:NB+2*i0+2*NL]
    end
    return d_complex, H_vec
end


"""Evaluate orthogonalized Laurent series."""
function evaluate_laurent_arnoldi(d::Vector{ComplexF64}, H_vec::Vector{Matrix{ComplexF64}}, zc_vec::Vector{ComplexF64}, z::Vector{ComplexF64})
    NB = length(H_vec)
    NL = size(H_vec[1], 1)
    W = zeros(ComplexF64, length(z), NL)
    y = zeros(ComplexF64, length(z))
    for nb = 1:NB
        zc = zc_vec[nb]
        H = H_vec[nb]
        W[:,1] = 1 ./ (z .- zc)
        y .= y + W[:,1] * d[(nb-1)*NL+1]
        for k = 1:NL-1
            w = W[:,k] ./ (z .- zc)
            for j = 1:k
                w = w - H[j,k] * W[:,j]
            end
            W[:,k+1] = w / H[k+1,k]
            y .= y + W[:,k+1] * d[(nb-1)*NL+k+1]
        end
        #y = y + conj(W') * d[(nb-1)*NL+1:nb*NL]
    end
    return y
end


"""Evaluate orthogonalized Laurent series."""
function evaluate_laurent_arnoldi(d::Vector{ComplexF64}, H_vec::Vector{Matrix{ComplexF64}}, zc_vec::Vector{ComplexF64}, z::ComplexF64)
    NB = length(H_vec)
    NL = size(H_vec[1], 1)
    W = zeros(ComplexF64, NL)
    y = 0
    for nb = 1:NB
        zc = zc_vec[nb]
        H = H_vec[nb]
        W[1] = 1 ./ (z .- zc)
        y = y + W[1] * d[(nb-1)*NL+1]
        for k = 1:NL-1
            w = W[k] ./ (z .- zc)
            for j = 1:k
                w = w - H[j,k] * W[j]
            end
            W[k+1] = w / H[k+1,k]
            y = y + W[k+1] * d[(nb-1)*NL+k+1]
        end
        #y = y + conj(W') * d[(nb-1)*NL+1:nb*NL]
    end
    return y
end

