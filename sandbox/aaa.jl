
using ComplexRegions
using ComplexPlots
using RationalFunctionApproximation
using CairoMakie
using DomainColoring


const shg = current_figure
f = conj

bodyshape(θ) = exp(im*θ) * (1 + 0.2*sin(3*θ))
b1(s) = bodyshape(2π*s) - 1.13
b2(s) = bodyshape(2π*s) + 1.13

g = LinRange(0, 1, 100)[1:end-1]
c1 = b1.(g)
c2 = b2.(g)
c = [c1; c2]
scatter(c, markersize=5)

a = aaa(c, conj.(c))
scatter!(poles(a), markersize=5, color=:black)
limits!(-3, 3, -2, 2)
shg()
save("aaa.pdf", shg())

a1 = aaa(c1, conj.(c1))
scatter!(poles(a1), markersize=4, color=:red)
limits!(-3, 3, -2, 2)
shg()

d12 = abs.(c1 .- conj.(c2'))
d1 = minimum(d12, dims=2)[:,1]
d2 = minimum(d12, dims=1)[1,:]
d = [d1; d2]

T1 = cos.(10*2π*g)
scatter(c1, markersize=5, color=T1)
a1 = aaa(c1, T1 .* conj.(c1))
scatter!(poles(a1), markersize=4, color=:red)
limits!(-3, 3, -2, 2)
shg()
limits!(-0.5, 0.5, 0, 1)
shg()

C1 = Polygon(c1)
C2 = Polygon(c2)
c = ExteriorRegion([C1, C2])
plot(c)
r = approximate(f, c)


C1 = ClosedCurve(b1)
C2 = ClosedCurve(b2)
r = approximate(f, C1)
r = approximate(f, exterior(C1))
C = ExteriorRegion([C1, C2])
r = approximate(f, C)

domaincolor(r, [-1.5, 1.5, -1.5, 1.5], abs=true)
lines!(unit_circle, color=:white, linewidth=5)
scatter!(poles(r), markersize=18, color=:black, marker=:xcross)
limits!(-1.5, 1.5, -1.5, 1.5)
shg()

