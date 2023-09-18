using Dates

"""
    $(TYPEDEF)

Wraps a `TensorBoardLogger.TBLogger` to construct a callback to be passed to
`Turing.sample`.

# Usage

    TensorBoardCallback(; kwargs...)
    TensorBoardCallback(directory::string[, stats]; kwargs...)
    TensorBoardCallback(lg::TBLogger[, stats]; kwargs...)

Constructs an instance of a `TensorBoardCallback`, creating a `TBLogger` if `directory` is 
provided instead of `lg`.

## Arguments
- `stats = nothing`: `OnlineStat` or lookup for variable name to statistic estimator.
  If `stats isa OnlineStat`, we will create a `DefaultDict` which copies `stats` for unseen
  variable names.
  If `isnothing`, then a `DefaultDict` with a default constructor returning a
  `OnlineStats.Series` estimator with `Mean()`, `Variance()`, and `KHist(num_bins)`
  will be used.

## Keyword arguments
- `num_bins::Int = 100`: Number of bins to use in the histograms.
- `variable_filter = nothing`: Filter determining whether or not we should log stats for a 
  particular variable. 
  If `isnothing` a default-filter constructed from `exclude` and 
  `include` will be used.
- `exclude = String[]`: If non-empty, these variables will not be logged.
- `include = String[]`: If non-empty, only these variables will be logged.
- `include_extras::Bool = true`: Include extra statistics from transitions.
- `extras_include = String[]`: If non-empty, only these extra statistics will be logged.
- `extras_exclude = String[]`: If non-empty, these extra statistics will not be logged.
- `include_hyperparams::Bool = true`: Include hyperparameters.
- `hyperparam_include = String[]`: If non-empty, only these hyperparameters will be logged.
- `hyperparam_exclude = String[]`: If non-empty, these hyperparameters will not be logged.
- `directory::String = nothing`: if specified, will together with `comment` be used to
   define the logging directory.
- `comment::String = nothing`: if specified, will together with `directory` be used to
   define the logging directory.

# Fields
$(TYPEDFIELDS)
"""
struct TensorBoardCallback{F1,F2,F3, L}
    "Underlying logger."
    logger::TBLogger
    "Filter determining whether to include stats for a particular variable."
    variable_filter::F1
    "Include extra statistics from transitions."
    include_extras::Bool
    "Filter determining whether to include a particular extra statistic."
    extras_filter::F2
    "Include hyperparameters."
    include_hyperparams::Bool
    "Filter determining whether to include a particular hyperparameter."
    hyperparam_filter::F3
    "Lookup for variable name to statistic estimate."
    stats::L
end

function TensorBoardCallback(directory::String, args...; kwargs...)
    TensorBoardCallback(args...; directory = directory, kwargs...)
end

function TensorBoardCallback(args...; comment = "", directory = nothing, kwargs...)
    log_dir = if isnothing(directory)
        "runs/$(Dates.format(now(), dateformat"Y-m-d_H-M-S"))-$(gethostname())$(comment)"
    else
        directory
    end
    
    # Set up the logger
    lg = TBLogger(log_dir, min_level=Logging.Info; step_increment=0)

    return TensorBoardCallback(lg, args...; kwargs...)
end

function TensorBoardCallback(
    lg::TBLogger,
    stats = nothing;
    num_bins::Int = 100,
    exclude = nothing,
    include = nothing,
    filter = nothing,
    include_extras::Bool = true,
    extras_include = nothing,
    extras_exclude = nothing,
    extras_filter = nothing,
    include_hyperparams::Bool = false,
    hyperparams_include = nothing,
    hyperparams_exclude = nothing,
    hyperparams_filter = nothing,
    kwargs...
)
    # Create the filters.
    variable_filter_f = if !isnothing(filter)
        filter
    else
        Filter(include=include, exclude=exclude)
    end
    extras_filter_f = if !isnothing(extras_filter)
        extras_filter
    else
        Filter(include=extras_include, exclude=extras_exclude)
    end
    hyperparams_filter_f = if !isnothing(hyperparams_filter)
        hyperparams_filter
    else
        Filter(include=hyperparams_include, exclude=hyperparams_exclude)
    end
    
    # Lookups: create default ones if not given
    stats_lookup = if stats isa OnlineStat
        # Warn the user if they've provided a non-empty `OnlineStat`
        nobs(stats) > 0 && @warn("using statistic with observations as a base: $(stats)")
        let o = stats
            DefaultDict{String, typeof(o)}(() -> deepcopy(o))
        end
    elseif !isnothing(stats)
        # If it's not an `OnlineStat` nor `nothing`, assume user knows what they're doing
        stats
    else
        # This is default
        let o = OnlineStats.Series(Mean(), Variance(), KHist(num_bins))
            DefaultDict{String, typeof(o)}(() -> deepcopy(o))
        end
    end

    return TensorBoardCallback(
        lg,
        variable_filter_f,
        include_extras,
        extras_filter_f,
        include_hyperparams,
        hyperparams_filter_f,
        stats_lookup
    )
end

const TuringTransition = Union{
    Turing.Inference.Transition,
    Turing.Inference.SMCTransition,
    Turing.Inference.PGTransition,
}

"""
    params_from_transition(model, transition)

Return an iterator over `(name, value)` present in `transition` where
`name` is a string and `value` is a scalar.
"""
function params_from_transition(model::Turing.DynamicPPL.Model, transition::TuringTransition)
    vns, vals = Turing.Inference._params_to_array(model, [transition])
    return zip(Iterators.map(string, vns), vals)
end

"""
    extras_from_transition(model, transition)

Return an iterator over `(name, value)` present in `transition` where
`name` is a string and `value` is a scalar.
"""
function extras_from_transition(model::Turing.DynamicPPL.Model, transition::TuringTransition)
    names, vals = Turing.Inference.get_transition_extras([transition])
    return zip(string.(names), vec(vals))
end

function hyperparameters(model, sampler::Turing.Sampler{<:Turing.Inference.NUTS})
    return (
        string(f) => getfield(sampler.alg, f)
        for f in fieldnames(typeof(sampler.alg))
    )
end

function hyperparameter_metrics(model, sampler::Turing.Sampler{<:Turing.Inference.NUTS})
    return [
        "extras/acceptance_rate/stat/Mean",
        "extras/max_hamiltonian_energy_error/stat/Mean",
        "extras/lp/stat/Mean",
        "extras/n_steps/stat/Mean",
        "extras/tree_depth/stat/Mean"
    ]
end

function hyperparameter_metrics(model, sampler)
    @warn "No hyperparameter metrics specified for $(typeof(model)) and $(typeof(sampler)). Implement `hyperparameter_metrics(model, sampler)` to specify them."
    return String[]
end

function (cb::TensorBoardCallback)(
    rng, model, sampler, transition, state, iteration;
    kwargs...
)
    stats = cb.stats
    lg = cb.logger
    variable_filter = cb.variable_filter
    extras_filter = cb.extras_filter
    hyperparams_filter = cb.hyperparam_filter

    if iteration == 1 && cb.include_hyperparams
        # If it's the first iteration, we write the hyperparameters.
        TensorBoardLogger.write_hparams!(
            lg,
            Dict(
                (name, val) for (name, val) in hyperparameters(model, sampler)
                    if hyperparams_filter(name)
            ),
            hyperparameter_metrics(model, sampler)
        )
    end
    
    with_logger(lg) do
        for (name, val) in params_from_transition(model, transition)
            variable_filter(name) || continue
            stat = stats[name]

            # Log the raw value
            @info name val

            # Update statistic estimators
            fit!(stat, val)

            # Need some iterations before we start showing the stats
            @info name stat
        end

        # Transition statstics
        if cb.include_extras
            for (name, val) in extras_from_transition(model, transition)
                extras_filter(name) || continue
                @info ("extras/" * name) val

                # TODO: Make this customizable.
                if val isa Real
                    stat = stats["extras/" * name]
                    fit!(stat, float(val))
                    @info ("extras/" * name) stat
                end
            end
        end
        @info "" log_step_increment=1
    end
end
