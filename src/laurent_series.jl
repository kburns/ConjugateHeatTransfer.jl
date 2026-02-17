import Base.length


##########################
## Series abstract type ##
##########################

abstract type Series end

"""Series length."""
function length(series::Series)
    return length(series.coefficients)
end

"""Callable interface for series."""
function (series::Series)(z)
    return evaluate(series, z)
end

"""Vectorized evaluation without output array."""
function evaluate(series::Series, z::Vector{ComplexF64})
    out = zeros(ComplexF64, length(z))
    return evaluate!(series, z, out)
end


#######################
## Series Collection ##
#######################

"""Collection of series."""
mutable struct SeriesCollection <: Series
    series::Vector{Series}
end

"""Series collection length."""
function length(collection::SeriesCollection)
    return sum(length(series) for series in collection.series)
end

"""Scalar evaluation."""
function evaluate(collection::SeriesCollection, z::ComplexF64)
    return sum(evaluate(series, z) for series in collection.series)
end

"""Vectorized evaluation adding to output array."""
function evaluate!(collection::SeriesCollection, z::Vector{ComplexF64}, out::Vector{ComplexF64})
    # Serial evaluation, summing in-place
    for series in collection.series
        evaluate!(series, z, out)
    end
    return out
end


#######################
## Polynomial Series ##
#######################

"""
Polynomial series:
    f(z) = sum_{i=0}^{N} c_i z^i
"""
mutable struct PolynomialSeries <: Series
    coefficients::Vector{ComplexF64}
end

"""Scalar evaluation."""
function evaluate(series::PolynomialSeries, z::ComplexF64)
    return sum(series.coefficients[i] * z^(i-1) for i in 1:length(series))
end

"""Vectorized evaluation adding to output array."""
function evaluate!(series::PolynomialSeries, z::Vector{ComplexF64}, out::Vector{ComplexF64})
    for i = 1:length(series)
        @inbounds out .+= series.coefficients[i] * z.^(i-1)
    end
    return out
end


####################
## Laurent Series ##
####################

"""
Principal Laurent series:
    f(z) = sum_{i=1}^{N} c_i (z - z_c)^{-i}
"""
mutable struct PrincipalLaurentSeries <: Series
    center::ComplexF64
    coefficients::Vector{ComplexF64}
end

"""Scalar evaluation."""
function evaluate(series::PrincipalLaurentSeries, z::ComplexF64)
    return sum(series.coefficients[i] * (z - series.center)^(-i) for i in 1:length(series))
end

"""Vectorized evaluation adding to output array."""
function evaluate!(series::PrincipalLaurentSeries, z::Vector{ComplexF64}, out::Vector{ComplexF64})
    for i = 1:length(series)
        @inbounds out .+= series.coefficients[i] * (z .- series.center).^(-i)
    end
    return out
end


###################################
## Orthogonalized Laurent Series ##
###################################

"""
Principal Laurent series orthogonalized with respect to some set of nodes.
"""
mutable struct OrthogonalizedLaurentSeries <: Series
    center::ComplexF64
    coefficients::Vector{ComplexF64}
    H::Matrix{ComplexF64}
end

"""Scalar evaluation."""
function evaluate(series::OrthogonalizedLaurentSeries, z::ComplexF64)
    J = length(series.coefficients)
    Q = ones(ComplexF64, J+1)
    out = zero(ComplexF64)
    @inbounds for j = 2:(J+1)
        q = Q[j-1] / (z - series.center)
        @inbounds for k = 1:(j-1)
            q -= series.H[k,j] * Q[k]
        end
        Q[j] = q / series.H[j,j]
        out += Q[j] * series.coefficients[j-1]
    end
    return out
end

"""Vectorized evaluation adding to output array."""
function evaluate!(series::OrthogonalizedLaurentSeries, z::Vector{ComplexF64}, out::Vector{ComplexF64})
    I = length(z)
    J = length(series.coefficients)
    Q = ones(ComplexF64, I, J+1)
    q = zeros(ComplexF64, I)
    for j = 2:(J+1)
        @inbounds @views q .= Q[:,j-1] ./ (z .- series.center)
        for k = 1:(j-1)
            @inbounds @views q .-= series.H[k,j] .* Q[:,k]
        end
        @inbounds Q[:,j] .= q ./ series.H[j,j]
        @inbounds @views out .+= Q[:,j] .* series.coefficients[j-1]
    end
    return out
end

"""Orthogonalized Laurent matrix."""
function OrthogonalizedLaurentMatrix(center::ComplexF64, degree::Int, nodes::Vector{ComplexF64})
    I = length(nodes)
    J = degree
    Q = ones(ComplexF64, I, J+1)
    H = zeros(ComplexF64, J+1, J+1)
    q = zeros(ComplexF64, I)
    # Orthogonalize the Laurent series, including constant term
    # Normalize columns s.t. norm(q,2) = sqrt(I)
    @inbounds for j = 2:(J+1)
        @inbounds @views q .= Q[:,j-1] ./ (nodes .- center)
        @inbounds for k = 1:(j-1)
            @inbounds @views H[k,j] = (Q[:,k]' * q) / I
            @inbounds @views q .-= H[k,j] .* Q[:,k]
        end
        H[j,j] = norm(q) / sqrt(I)
        @inbounds Q[:,j] .= q ./ H[j,j]
    end
    # Return the series matrix, dropping constant column
    return @view(Q[:,2:end]), H
end


########################
## Simple Pole Series ##
########################

"""
Simple pole series:
    f(z) = sum_{i=1}^{N} c_i / (z - z_{c,i})
"""
mutable struct SimplePoleSeries <: Series
    centers::Vector{ComplexF64}
    coefficients::Vector{ComplexF64}
end

"""Scalar evaluation."""
function evaluate(series::SimplePoleSeries, z::ComplexF64)
    out = zero(ComplexF64)
    @inbounds for i = 1:length(series)
        out += series.coefficients[i] / (z - series.centers[i])
    end
    return out
end

"""Vectorized evaluation adding to output array."""
function evaluate!(series::SimplePoleSeries, z::Vector{ComplexF64}, out::Vector{ComplexF64})
    @inbounds for i = 1:length(series)
        @inbounds out .+= series.coefficients[i] ./ (z .- series.centers[i])
    end
    return out
end

"""Simple pole matrix."""
function SimplePoleMatrix(centers::Vector{ComplexF64}, nodes::Vector{ComplexF64})
    I = length(nodes)
    J = length(centers)
    M = zeros(ComplexF64, I, J)
    @inbounds for j = 1:J
        @inbounds for i = 1:I
            M[i,j] = 1 / (nodes[i] - centers[j])
        end
    end
    return M
end
