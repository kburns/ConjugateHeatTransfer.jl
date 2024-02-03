

using ApproxFun


function build_interpolant(func, space; N=Inf)
    if N == Inf
        # Test a few random points since ApproxFun seems to choke when f≈0
        a, b = domain(space).a, domain(space).b
        test_points = a .+ (b-a)*rand(5)
        if isapprox(func.(test_points), zeros(5), atol=1e-10)
            return Fun(x->func(x)+1, space) - 1
        else
            return Fun(func, space)
        end
    else
        values = func.(points(space, N))
        return Fun(space, ApproxFun.transform(space, values))
    end
end


function interpolate_values(space, values)
    return Fun(space, ApproxFun.transform(space, values))
end

