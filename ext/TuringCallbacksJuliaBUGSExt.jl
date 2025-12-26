module TuringCallbacksJuliaBUGSExt

if isdefined(Base, :get_extension)
    import JuliaBUGS
    import JuliaBUGS.Model: BUGSModel
    import TuringCallbacks
    import AbstractMCMC
    import AdvancedHMC
else
    import ..JuliaBUGS
    import ..JuliaBUGS.Model: BUGSModel
    import ..TuringCallbacks
    import ..AbstractMCMC
    import ..AdvancedHMC
end

"""
    params_and_values(model::AbstractMCMC.LogDensityModel{<:BUGSModel}, transition::AdvancedHMC.Transition; kwargs...)

Extract parameter names and values from a JuliaBUGS model transition.

Maps the flattened parameter vector from the HMC transition back to named parameters,
handling both scalar and vector parameters correctly.
"""
function TuringCallbacks.params_and_values(
    model::AbstractMCMC.LogDensityModel{<:BUGSModel},
    transition::AdvancedHMC.Transition;
    kwargs...
)
    bugs_model = model.logdensity
    gd = bugs_model.graph_evaluation_data
    param_names = gd.sorted_parameters
    param_values = transition.z.θ
    
    # Build pairs of (name, value) by mapping the flattened vector back to parameters
    pairs = Tuple{String, Float64}[]
    pos = 1
    
    for vn in param_names
        len = if bugs_model.transformed
            bugs_model.transformed_var_lengths[vn]
        else
            bugs_model.untransformed_var_lengths[vn]
        end
        
        if len == 1
            # Scalar parameter
            push!(pairs, (string(vn), param_values[pos]))
            pos += 1
        else
            # Vector/array parameter - log each element individually
            for i in 1:len
                push!(pairs, (string(vn) * "[$i]", param_values[pos]))
                pos += 1
            end
        end
    end
    
    return pairs
end

"""
    extras(model::AbstractMCMC.LogDensityModel{<:BUGSModel}, transition::AdvancedHMC.Transition; kwargs...)

Extract extra statistics from a JuliaBUGS model HMC transition.

Includes log probability and HMC-specific statistics like acceptance rate, 
step size, tree depth, etc.
"""
function TuringCallbacks.extras(
    model::AbstractMCMC.LogDensityModel{<:BUGSModel},
    transition::AdvancedHMC.Transition;
    kwargs...
)
    # Extract HMC statistics from transition
    stats = AdvancedHMC.stat(transition)
    names = collect(keys(stats))
    vals = collect(values(stats))
    
    # Add log probability at the front
    pushfirst!(names, :lp)
    pushfirst!(vals, transition.z.ℓπ.value)
    
    return zip(string.(names), vals)
end

"""
    hyperparams(model::AbstractMCMC.LogDensityModel{<:BUGSModel}, sampler::AdvancedHMC.NUTS; kwargs...)

Extract hyperparameters from a NUTS sampler used with JuliaBUGS models.
"""
function TuringCallbacks.hyperparams(
    model::AbstractMCMC.LogDensityModel{<:BUGSModel},
    sampler::AdvancedHMC.NUTS;
    kwargs...
)
    return [
        "target_acceptance" => sampler.δ,
        "max_depth" => sampler.max_depth,
        "Δ_max" => sampler.Δ_max
    ]
end

"""
    hyperparam_metrics(model::AbstractMCMC.LogDensityModel{<:BUGSModel}, sampler::AdvancedHMC.NUTS)

Return metric names to track for NUTS hyperparameters with JuliaBUGS models.
"""
function TuringCallbacks.hyperparam_metrics(
    model::AbstractMCMC.LogDensityModel{<:BUGSModel},
    sampler::AdvancedHMC.NUTS
)
    return [
        "extras/acceptance_rate/stat/Mean",
        "extras/max_hamiltonian_energy_error/stat/Mean",
        "extras/lp/stat/Mean",
        "extras/n_steps/stat/Mean",
        "extras/tree_depth/stat/Mean"
    ]
end

end
