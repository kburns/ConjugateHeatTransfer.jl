module ConjugateHeatTransfer

using ExportAll
include("interpolation.jl")
include("quadrature.jl")
include("dirichlet_body.jl")
include("advection_kernels.jl")
include("dirichlet_panel.jl")
@exportAll()

end # module ConjugateHeatTransfer