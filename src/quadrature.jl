

using ApproxFun
using QuadGK


function adaptive_quadrature(f, a, b)
    F = Fun(f, a..b)
    return sum(F)
end


function adaptive_quadrature(f, a, b, c)
    F1 = Fun(f, a..b)
    F2 = Fun(f, b..c)
    return sum(F1) + sum(F2)
end


function clenshaw_curtis_quadrature(f, a, b, N)
    space = Chebyshev(a..b)
    F = BuildInterpolant(f, space; N=N)
    return sum(F)
end


default_quadrature(f, a, b) = quadgk(f, a, b; atol=1e-10, rtol=1e-10)[1];
default_quadrature(f, a, b, c) = quadgk(f, a, b, c; atol=1e-10, rtol=1e-10)[1];


function nested_quadgk(f, a, b)
    g(x) = quadgk(y->f(x,y), a, x, b)[1]
    return quadgk(g, a, b)[1]
end