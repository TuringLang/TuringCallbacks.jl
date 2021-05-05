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
- `directory::String = nothing`: if specified, will together with `comment` be used to
   define the logging directory.
- `comment::String = nothing`: if specified, will together with `directory` be used to
   define the logging directory.

# Fields
$(TYPEDFIELDS)
"""
struct TensorBoardCallback{F, L}
    "Underlying logger."
    logger::TBLogger
    "Filter determining whether or not we should log stats for a particular variable."
    variable_filter::F
    "Include extra statistics from transitions."
    include_extras::Bool
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
    exclude = String[],
    include = String[],
    include_extras::Bool = true,
    variable_filter = nothing,
    kwargs...
)
    # Create the filter
    filter = if !isnothing(variable_filter)
        variable_filter
    else
        varname -> (
            (isempty(exclude) || varname ∉ exclude) &&
            (isempty(include) || varname ∈ include)
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
        lg, filter, include_extras, stats_lookup
    )
end

function (cb::TensorBoardCallback)(rng, model, sampler, transition, iteration, state; kwargs...)
    stats = cb.stats
    lg = cb.logger
    filter = cb.variable_filter
    
    with_logger(lg) do
        for (ksym, val) in zip(Turing.Inference._params_to_array([transition])...)
            k = string(ksym)
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
