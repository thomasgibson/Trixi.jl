# TODO: Taal refactor, rename to
# - euler_blast_wave_shockcapturing_amr.jl
# or something similar?

using OrdinaryDiffEq
using Trixi

###############################################################################
# semidiscretization of the compressible Euler equations

equations = CompressibleEulerEquations2D(1.4)

initial_conditions = initial_conditions_blast_wave

surface_flux = flux_lax_friedrichs
volume_flux  = flux_chandrashekar
indicator_hg = IndicatorHennemannGassner(alpha_max=0.5,
                                         alpha_min=0.001,
                                         alpha_smooth=true,
                                         variable=density_pressure)
volume_integral = VolumeIntegralShockCapturingHG(indicator_hg;
                                                 volume_flux_dg=volume_flux,
                                                 volume_flux_fv=surface_flux)
solver = DGSEM(3, surface_flux, volume_integral)

coordinates_min = (-2, -2)
coordinates_max = ( 2,  2)
mesh = TreeMesh(coordinates_min, coordinates_max,
                initial_refinement_level=6,
                n_cells_max=10_000)


semi = SemidiscretizationHyperbolic(mesh, equations, initial_conditions, solver)


###############################################################################
# ODE solvers, callbacks etc.

tspan = (0.0, 12.5)
ode = semidiscretize(semi, tspan)

summary_callback = SummaryCallback()

analysis_interval = 100
alive_callback = AliveCallback(analysis_interval=analysis_interval)
analysis_callback = AnalysisCallback(semi, analysis_interval=analysis_interval)

stepsize_callback = StepsizeCallback(cfl=1.0)

save_solution = SaveSolutionCallback(solution_interval=10,
                                     save_initial_solution=true,
                                     save_final_solution=true,
                                     solution_variables=:primitive)

callbacks = CallbackSet(summary_callback, stepsize_callback, analysis_callback, save_solution, alive_callback)


###############################################################################
# run the simulation

sol = solve(ode, CarpenterKennedy2N54(williamson_condition=false), dt=stepsize_callback(ode),
            save_everystep=false, callback=callbacks);
summary_callback() # print the timer summary
