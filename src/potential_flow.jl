

using LinearAlgebra
using Statistics
using RationalFunctionApproximation


"""Compute the error in the potential flow map on a single body."""
function potential_flow_boundary_errors(W::Series, bodies::Vector{DirichletBody}, N_sample::Int)
    # Sample W on bodies
    θ_grid = fourier_grid(N_sample)
    body_samples = [b.zθ.(θ_grid) for b in bodies]
    # Determine error as maximum deviation from the mean ψ on each body
    errors = zeros(length(bodies))
    for i = 1:length(bodies)
        ψ = imag.(W(body_samples[i]))
        errors[i] = norm(ψ .- mean(ψ), Inf)
    end
    return errors
end


"""Adaptively solve for potential flow around bodies using Laurent series."""
function adaptive_laurent_potential_flow(bodies::Vector{DirichletBody}; U=1, atol=1e-10, N_laurent_steps=32, N_laurent_max=1024, sample_ratio=6)
    N_laurent = N_laurent_steps
    while N_laurent <= N_laurent_max
        W = laurent_potential_flow(bodies, N_laurent; U=U, sample_ratio=sample_ratio)
        errors = potential_flow_boundary_errors(W, bodies, 4*sample_ratio*N_laurent)
        if maximum(errors) < atol
            println("  N_laurent = ", N_laurent, ", error = ", round(maximum(errors), sigdigits=3))
            return W
        end
        N_laurent += N_laurent_steps
    end
    error("Failed to converge")
end


"""Solve the potential flow around bodies using fixed-degree Laurent series."""
function laurent_potential_flow(bodies::Vector{DirichletBody}, N_laurent::Int; U=1, sample_ratio=6)
    # Samples body boundaries (TODO: allow variable samples)
    N_sample = sample_ratio * N_laurent
    θ_grid = fourier_grid(N_sample)
    body_degrees = [N_laurent for b in bodies]
    body_samples = [b.zθ.(θ_grid) for b in bodies]
    # Setup least squares problem
    nodes = reduce(vcat, body_samples)
    Q_vec = Vector{Matrix{ComplexF64}}()
    # Add constant term to absorb ψ for each body
    Qb = zeros(ComplexF64, length(nodes), length(bodies))
    i0 = 1
    for nb = 1:length(bodies)
        i1 = i0 + length(body_samples[nb])
        Qb[i0:i1-1, nb] .= 1
        i0 = i1
    end
    push!(Q_vec, Qb)
    # Add orthogonalized Laurent series for each body
    H_vec = Vector{Matrix{ComplexF64}}()
    for nb = 1:length(bodies)
        Q, H = OrthogonalizedLaurentMatrix(bodies[nb].zc, body_degrees[nb], nodes)
        push!(Q_vec, Q)
        push!(H_vec, H)
    end
    # Solve the least squares problem
    Q = reduce(hcat, Q_vec)
    Q = [imag(Q) real(Q)]
    y = -imag(conj(U)*nodes)
    x = Q \ y
    println("  Q shape: ", size(Q))
    println("  Fitting error: ", norm(Q*x - y, Inf))
    nx = size(Q,2) ÷ 2
    xc = x[1:nx] .+ im*x[nx+1:end]
    # Build function
    series_vec = Vector{Series}([PolynomialSeries([0, conj(U)])]) # Add conj(U)*z
    i0 = length(bodies) + 1
    for nb = 1:length(bodies)
        i1 = i0 + body_degrees[nb]
        series = OrthogonalizedLaurentSeries(bodies[nb].zc, xc[i0:i1-1], H_vec[nb])
        push!(series_vec, series)
        i0 = i1
    end
    W = SeriesCollection(series_vec)
    return W
end


"""Solve the potential flow around bodies using AAA+Laurent."""
function aaa_potential_flow(bodies::Vector{DirichletBody}, N_laurent::Int; U=1, sample_ratio=6)
    # Find AAA-LS poles
    for body in bodies
        find_aaa_poles_adaptive!(body)
    end
    # Samples body boundaries
    body_samples = Vector{Vector{ComplexF64}}()
    for body in bodies
        N_sample = sample_ratio * (N_laurent + length(body.aaa_poles))
        push!(body_samples, body.zθ.(fourier_grid(N_sample)))
    end
    # Setup least squares problem
    nodes = reduce(vcat, body_samples)
    Q_vec = Vector{Matrix{ComplexF64}}()
    # Add constant term to absorb ψ for each body
    Qb = zeros(ComplexF64, length(nodes), length(bodies))
    i0 = 1
    for nb = 1:length(bodies)
        i1 = i0 + length(body_samples[nb])
        Qb[i0:i1-1, nb] .= 1
        i0 = i1
    end
    push!(Q_vec, Qb)
    # Add orthogonalized Laurent series for each body
    H_vec = Vector{Matrix{ComplexF64}}()
    for nb = 1:length(bodies)
        Q, H = OrthogonalizedLaurentMatrix(bodies[nb].zc, N_laurent, nodes)
        push!(Q_vec, Q)
        push!(H_vec, H)
    end
    # Add AAA poles
    for nb = 1:length(bodies)
        Q = SimplePoleMatrix(bodies[nb].aaa_poles, nodes)
        push!(Q_vec, Q)
    end
    # Solve the least squares problem
    Q = reduce(hcat, Q_vec)
    Q = [imag(Q) real(Q)]
    y = -imag(conj(U)*nodes)
    x = Q \ y
    println("  Q shape: ", size(Q))
    println("  Fitting error: ", norm(Q*x - y, Inf))
    nx = size(Q,2) ÷ 2
    xc = x[1:nx] .+ im*x[nx+1:end]
    # Build function
    series_vec = Vector{Series}([PolynomialSeries([0, conj(U)])]) # Add conj(U)*z
    i0 = length(bodies) + 1
    for nb = 1:length(bodies)
        i1 = i0 + N_laurent
        series = OrthogonalizedLaurentSeries(bodies[nb].zc, xc[i0:i1-1], H_vec[nb])
        push!(series_vec, series)
        i0 = i1
    end
    for nb = 1:length(bodies)
        i1 = i0 + length(bodies[nb].aaa_poles)
        series = SimplePoleSeries(bodies[nb].aaa_poles, xc[i0:i1-1])
        push!(series_vec, series)
        i0 = i1
    end
    W = SeriesCollection(series_vec)
    # TODO: print boundary errors
    return W
end


"""Simple poles matrix."""
function simple_poles_matrix(poles::Vector{ComplexF64}, targets::Vector{ComplexF64})
    Q = zeros(ComplexF64, length(targets), length(poles))
    for j = 1:length(poles)
        Q[:,j] = 1 ./ (targets .- poles[j])
    end
    return Q
end


"""NEW LAURENT AT AAA"""
function fit_aaa_laurent(x_vec::Vector{Vector{ComplexF64}}, zc_vec::Vector{ComplexF64}, f_vec::Vector{Vector{Float64}}, NL::Int)
    NB = length(x_vec)
    x = reduce(vcat, x_vec)
    M = length(x)
    Q_vec = Vector{Matrix{ComplexF64}}()
    # Add constant term to absorb ψ for each body
    Qc = zeros(Float64, M, NB)
    i0 = 1
    for nb = 1:NB
        i1 = i0 + length(x_vec[nb])
        Qc[i0:i1-1, nb] .=  1
        i0 = i1
    end
    # Add AAA poles
    aaa_poles_vec = Vector{Vector{ComplexF64}}()
    for nb = 1:NB
        Q, aaa_poles = AAA_poles_matrix(x_vec[nb], x)
        #push!(Q_vec, Q)
        push!(aaa_poles_vec, aaa_poles)
    end
    aaa_poles = reduce(vcat, aaa_poles_vec)
    NAAA = length(aaa_poles)
    # Add orthogonalized Laurent series
    H_vec = Vector{Matrix{ComplexF64}}()
    for nb = 1:NB
        aaa_poles_b = aaa_poles_vec[nb]
        for i = 1:length(aaa_poles_b)
            Q, H = laurent_arnoldi_matrix(x, aaa_poles_b[i], NL)
            push!(Q_vec, Q)
            push!(H_vec, H)
        end
    end
    # Solve the least squares problem
    Q = reduce(hcat, Q_vec)
    Q = [Qc real(Q) imag(Q)]
    f = reduce(vcat, f_vec)
    d = Q \ f
    println("  Q shape: ", size(Q))
    println("  Fitting error: ", norm(Q*d - f, Inf))
    # Build interpolant
    d_complex = im*d[NB+1:NB+NAAA+NB*NL] + d[NB+NAAA+NB*NL+1:end]
    d_aaa = d_complex[1:NAAA]
    d_laurent = d_complex[NAAA+1:end]
    W(z) = evaluate_aaa(d_aaa, aaa_poles, z) + evaluate_laurent_arnoldi(d_laurent, H_vec, zc_vec, z)
    return W
end


"""Fit AAA+Laurent using least squares."""
function fit_aaa_laurent_OLD(x_vec::Vector{Vector{ComplexF64}}, zc_vec::Vector{ComplexF64}, f_vec::Vector{Vector{Float64}}, NL::Int)
    NB = length(x_vec)
    x = reduce(vcat, x_vec)
    M = length(x)
    Q_vec = Vector{Matrix{ComplexF64}}()
    # Add constant term to absorb ψ for each body
    Qc = zeros(Float64, M, NB)
    i0 = 1
    for nb = 1:NB
        i1 = i0 + length(x_vec[nb])
        Qc[i0:i1-1, nb] .=  1
        i0 = i1
    end
    # Add AAA poles
    aaa_poles_vec = Vector{Vector{ComplexF64}}()
    for nb = 1:NB
        push!(Q_vec, simple_poles_matrix(bodies[nb].aaa_poles, x))
        push!(aaa_poles_vec, bodies[nb].aaa_poles)
    end
    aaa_poles = reduce(vcat, aaa_poles_vec)
    NAAA = length(aaa_poles)
    # Add orthogonalized Laurent series
    H_vec = Vector{Matrix{ComplexF64}}()
    for nb = 1:NB
        Q, H = laurent_arnoldi_matrix(x, zc_vec[nb], NL)
        push!(Q_vec, Q)
        push!(H_vec, H)
    end
    # Solve the least squares problem
    Q = reduce(hcat, Q_vec)
    Q = [Qc real(Q) imag(Q)]
    f = reduce(vcat, f_vec)
    d = Q \ f
    println("  Q shape: ", size(Q))
    println("  Fitting error: ", norm(Q*d - f, Inf))
    # Build interpolant
    d_complex = im*d[NB+1:NB+NAAA+NB*NL] + d[NB+NAAA+NB*NL+1:end]
    d_aaa = d_complex[1:NAAA]
    d_laurent = d_complex[NAAA+1:end]
    W(z) = evaluate_aaa(d_aaa, aaa_poles, z) + evaluate_laurent_arnoldi(d_laurent, H_vec, zc_vec, z)
    return W
end


"""Fit orthogonalized Laurent series using least squares."""
function fit_laurent(x_vec::Vector{Vector{ComplexF64}}, zc_vec::Vector{ComplexF64}, f_vec::Vector{Vector{Float64}}, NL::Int)
    NB = length(x_vec)
    x = reduce(vcat, x_vec)
    M = length(x)
    Q_vec = Vector{Matrix{ComplexF64}}()
    # Add constant term to absorb ψ for each body
    Qc = zeros(Float64, M, NB)
    i0 = 1
    for nb = 1:NB
        i1 = i0 + length(x_vec[nb])
        Qc[i0:i1-1, nb] .=  1
        i0 = i1
    end
    # Orthogonalize the Laurent series for each body
    H_vec = Vector{Matrix{ComplexF64}}()
    for nb = 1:NB
        Q, H = laurent_arnoldi_matrix(x, zc_vec[nb], NL)
        push!(H_vec, H)
        # Add to the system matrix
        push!(Q_vec, Q)
    end
    # Solve the least squares problem
    Q = reduce(hcat, Q_vec)
    Q = [Qc real(Q) imag(Q)]
    f = reduce(vcat, f_vec)
    #println("  Condition number: ", cond(Q_full))
    d = Q \ f
    println("  Q shape: ", size(Q))
    println("  Fitting error: ", norm(Q*d - f, Inf))
    # Build interpolant
    d_laurent = im*d[NB+1:NB+NB*NL] .+ d[NB+NB*NL+1:end]
    W(z) = evaluate_laurent_arnoldi(d_laurent, H_vec, zc_vec, z)
    return W
end


"""Evaluate AAA series."""
function evaluate_aaa(d::Vector{ComplexF64}, aaa_poles::Vector{ComplexF64}, z::Vector{ComplexF64})
    y = zeros(ComplexF64, length(z))
    for i = 1:length(d)
        y .= y .+ d[i] ./ (z .- aaa_poles[i])
    end
    return y
end


"""Evaluate AAA series."""
function evaluate_aaa(d::Vector{ComplexF64}, aaa_poles::Vector{ComplexF64}, z::ComplexF64)
    y = 0
    for i = 1:length(d)
        y = y + d[i] / (z - aaa_poles[i])
    end
    return y
end

