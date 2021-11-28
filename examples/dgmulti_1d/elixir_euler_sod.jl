using Trixi, OrdinaryDiffEq, Plots


vol_flux = FluxRotated(flux_chandrashekar)
surface_flux = FluxLaxFriedrichs()

dg = DGMulti(polydeg = 4, element_type = Line(), approximation_type = Polynomial(),
             surface_integral = SurfaceIntegralWeakForm(surface_flux),
             volume_integral = VolumeIntegralFluxDifferencing(vol_flux))

equations = CompressibleEulerEquations1D(1.4)


@inline function initial_condition_sod(x, t, equations::CompressibleEulerEquations1D)
    gamma = equations.gamma
    gmn1 = 1.0 / (gamma - 1.0)
    x0 = 0.0
    x = x[1]
    sigma = 1e-13
    weight = 0.5 * (1.0 - tanh(1.0/sigma * (x - x0)))

    rho_ll = 1.0
    rho_rr = 0.125
    rho = rho_rr + (rho_ll - rho_rr)*weight

    rho_v1 = 0.0 * x

    p_ll = 1.0
    p_rr = 0.1
    rho_e_ll = gmn1 * p_ll
    rho_e_rr = gmn1 * p_rr
    rho_e = rho_e_rr + (rho_e_ll - rho_e_rr)*weight

    return SVector(rho, rho_v1, rho_e)
end


initial_condition = initial_condition_sod
boundary_conditions = BoundaryConditionDirichlet(initial_condition)

Nc = 32
cells_per_dimension = (Nc,)
vertex_coordinates, EToV = StartUpDG.uniform_mesh(dg.basis.elementType,
                                                  cells_per_dimension...)
mesh = VertexMappedMesh(vertex_coordinates, EToV, dg)

semi = SemidiscretizationHyperbolic(mesh,
                                    equations,
                                    initial_condition,
                                    dg;
                                    boundary_conditions=boundary_conditions)

tspan = (0.0, 0.2)
ode = semidiscretize(semi, tspan)

summary_callback = SummaryCallback()
alive_callback = AliveCallback(alive_interval=10)
analysis_interval = 100
analysis_callback = AnalysisCallback(semi,
                                     interval=analysis_interval,
                                     uEltype=real(dg))
visualization = VisualizationCallback(interval=10)

callbacks = CallbackSet(summary_callback,
                        analysis_callback,
                        alive_callback,
                        visualization)

###############################################################################
# run the simulation

sol = solve(ode,
            CarpenterKennedy2N54(williamson_condition=false),
            dt = estimate_dt(mesh, dg)/10,
            save_everystep=false,
            callback=callbacks);

summary_callback() # print the timer summary
