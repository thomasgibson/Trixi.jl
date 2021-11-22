using Trixi, OrdinaryDiffEq

dg = DGMulti(polydeg=3, element_type = Tri(), approximation_type = Polynomial(),
             surface_integral = SurfaceIntegralWeakForm(FluxLaxFriedrichs()),
             volume_integral = VolumeIntegralFluxDifferencing(flux_chandrashekar))

equations = CompressibleEulerEquations2D(1.4)


"""
    initial_condition_isentropic_vortex(x, t, equations::CompressibleEulerEquations2D)

Initial condition adapted from Section 2 (equation 2) of:
- Ken Mattsson, Magnus Sv\"{a}rd, Mark Carpenter, and Jan Nordstr\"{o}m (2006).
  High-order accurate computations for unsteady aerodynamics.
  [DOI](https://doi.org/10.1016/j.compfluid.2006.02.004).
"""
@inline function initial_condition_isentropic_vortex(x, t, equations::CompressibleEulerEquations2D)
    M = 0.5
    gamma = equations.gamma
    epsilon = 1
    x0 = 5

    fxyt = 1 - (((x[1] - x0) - t)^2 + x[2]^2)
    expterm = exp(fxyt/2)

    rho = (
        1 - ((epsilon^2 * (gamma - 1) * M^2)/(8*pi^2)) * exp(fxyt)
    ) ^ (1 / (gamma - 1))
    u = 1 - (epsilon*x[2]/(2*pi))*expterm
    v = ((epsilon*(x[1] - x0) - t)/(2*pi)) * expterm

    p = (rho^gamma)/(gamma * M^2)

    return prim2cons(SVector(rho, u, v, p), equations)
end


Nc = 10
cells_per_dimension = (2*Nc, Nc)
vertex_coordinates, EToV = StartUpDG.uniform_mesh(dg.basis.elementType,
                                                  cells_per_dimension...)

# remap the domain by modifying `vertex_coordinates`
vx, vy = vertex_coordinates
vx = map(x-> 20 * 0.5*(1 + x), vx)   # map [-1, 1] to [0, 20]
vy = map(x-> 5 * x, vy)              # map [-1, 1] to [-5, 5]
vertex_coordinates = (vx, vy)

mesh = VertexMappedMesh(vertex_coordinates, EToV, dg; is_periodic=(true, true))

initial_condition = initial_condition_isentropic_vortex

semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition, dg)

tspan = (0.0, 5.0)
ode = semidiscretize(semi, tspan)

summary_callback = SummaryCallback()
alive_callback = AliveCallback(alive_interval=10)
analysis_interval = 100
analysis_callback = AnalysisCallback(semi,
                                     interval=analysis_interval,
                                     uEltype=real(dg))
callbacks = CallbackSet(summary_callback,
                        analysis_callback,
                        alive_callback)

###############################################################################
# run the simulation

sol = solve(ode,
            CarpenterKennedy2N54(williamson_condition=false),
            dt = estimate_dt(mesh, dg)/10,
            save_everystep=false,
            callback=callbacks);

summary_callback() # print the timer summary
