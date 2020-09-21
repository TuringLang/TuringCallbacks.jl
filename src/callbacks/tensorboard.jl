#############################
### `TensorBoardCallback` ###
#############################
"""
    $(TYPEDEF)

Wraps a `TensorBoardLogger.TBLogger` to construct a callback to be passed to
`Turing.sample`.

# Usage

    TensorBoardCallback(lg::TBLogger, num_samples::Int, stats = nothing; kwargs...)
    TensorBoardCallback(directory::String, num_samples::Int, stats = nothing; kwargs...)

Constructs an instance of a `TensorBoardCallback`, creating a `TBLogger` if `directory` is 
provided instead of `lg`.

## Arguments
- `num_samples::Int`: Total number of MCMC steps that will be taken.
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

# Fields
$(TYPEDFIELDS)
"""
struct TensorBoardCallback{F, L}
    "Underlying logger."
    logger::TBLogger
    "Total number of MCMC steps that will be taken."
    num_samples::Int
    "Filter determining whether or not we should log stats for a particular variable."
    variable_filter::F
    "Include extra statistics from transitions."
    include_extras::Bool
    "Lookup for variable name to statistic estimate."
    stats::L
end

function TensorBoardCallback(directory::String, args...; kwargs...)
    # Set up the logger
    lg = TBLogger(directory, min_level=Logging.Info; step_increment=0)

    return TensorBoardCallback(lg, args...; kwargs...)
end

function TensorBoardCallback(
    lg::TBLogger,
    num_samples::Int,
    stats = nothing;
    num_bins::Int = 100,
    exclude = String[],
    include = String[],
    include_extras::Bool = true,
    variable_filter = nothing
)
    # Create the filter
    filter = if !isnothing(variable_filter)
        variable_filter
    else
        varname -> (
            (isempty(exclude) || varname ∉ exclude) &&
            (isempty(include) || varnmae ∈ include)
        )
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
        lg, num_samples, filter, include_extras, stats_lookup
    )
end

function (cb::TensorBoardCallback)(rng, model, sampler, transition, iteration)
    stats = cb.stats
    lg = cb.logger
    filter = cb.variable_filter
    
    with_logger(lg) do
        for (varname, (vals, ks)) in pairs(transition.θ)
            # Skip those variables which are to be excluded
            if !filter(string(varname))
                continue
            end
            
            for (k, val) in zip(ks, vals)
                if !filter(k)
                    continue
                end
                stat = stats[k]
                
                # Log the raw value
                @info k val

                # Update statistic estimators
                fit!(stat, val)

                # Need some iterations before we start showing the stats
                @info k stat
            end
        end

        # Transition statstics
        if cb.include_extras
            names, vals = Turing.Inference.get_transition_extras([transition])
            for (name, val) in zip(string.(names), vec(vals))
                @info ("extras/" * name) val
            end
        end
        @info "" log_step_increment=1
    end
end
