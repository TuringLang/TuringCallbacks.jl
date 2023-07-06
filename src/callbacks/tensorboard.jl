using Dates

"""
    $(TYPEDEF)

Wraps a `TensorBoardLogger.AbstractLogger` to construct a callback to
be passed to `AbstractMCMC.step`.

# Usage

    TensorBoardCallback(; kwargs...)
    TensorBoardCallback(directory::string[, stats]; kwargs...)
    TensorBoardCallback(lg::AbstractLogger[, stats]; kwargs...)

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
- `filter = nothing`: Filter determining whether or not we should log stats for a
  particular variable and value; expected signature is `filter(varname, value)`.
  If `isnothing` a default-filter constructed from `exclude` and
  `include` will be used.
- `exclude = nothing`: If non-empty, these variables will not be logged.
- `include = nothing`: If non-empty, only these variables will be logged.
- `include_extras::Bool = true`: Include extra statistics from transitions.
- `directory::String = nothing`: if specified, will together with `comment` be used to
   define the logging directory.
- `comment::String = nothing`: if specified, will together with `directory` be used to
   define the logging directory.

# Fields
$(TYPEDFIELDS)
"""
struct TensorBoardCallback{L,F,VI,VE}
    "Underlying logger."
    logger::AbstractLogger
    "Lookup for variable name to statistic estimate."
    stats::L
    "Filter determining whether or not we should log stats for a particular variable."
    filter::F
    "Variables to include in the logging."
    include::VI
    "Variables to exclude from the logging."
    exclude::VE
    "Include extra statistics from transitions."
    include_extras::Bool
    "Prefix used for logging realizations/parameters"
    param_prefix::String
    "Prefix used for logging extra statistics"
    extras_prefix::String
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
    lg::AbstractLogger,
    stats = nothing;
    num_bins::Int = 100,
    exclude = nothing,
    include = nothing,
    include_extras::Bool = true,
    filter = nothing,
    param_prefix::String = "",
    extras_prefix::String = "extras/",
    kwargs...
)
    # Lookups: create default ones if not given
    stats_lookup = if stats isa OnlineStat
        # Warn the user if they've provided a non-empty `OnlineStat`
        OnlineStats.nobs(stats) > 0 && @warn("using statistic with observations as a base: $(stats)")
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
        lg, stats_lookup, filter, include, exclude, include_extras, param_prefix, extras_prefix
    )
end

"""
    filter_param_and_value(cb::TensorBoardCallback, param_name, value)

Filter parameters and values from a `transition` based on the `filter` of `cb`.
"""
function filter_param_and_value(cb::TensorBoardCallback, param, value)
    if !isnothing(cb.filter)
        return cb.filter(param, value)
    end

    # Otherwise we construct from `include` and `exclude`.
    if !isnothing(cb.include)
        # If only `include` is given, we only return the variables in `include`.
        return param ∈ cb.include
    elseif !isnothing(cb.exclude)
        # If only `exclude` is given, we return all variables except those in `exclude`.
        return !(param ∈ cb.exclude)
    end

    # Otherwise we return `true` by default.
    return true
end
filter_param_and_value(cb::TensorBoardCallback, param_and_value::Tuple) = filter_param_and_value(cb, param_and_value...)

"""
    default_param_names_for_values(x)

Return an iterator of `θ[i]` for each element in `x`.
"""
default_param_names_for_values(x) = ("θ[$i]" for i = 1:length(x))


"""
    params_and_values(transition[, state]; kwargs...)
    params_and_values(model, sampler, transition, state; kwargs...)

Return an iterator over parameter names and values from a `transition`.
"""
params_and_values(transition, state; kwargs...) = params_and_values(transition; kwargs...)
params_and_values(model, sampler, transition, state; kwargs...) = params_and_values(transition, state; kwargs...)

"""
    extras(transition[, state]; kwargs...)
    extras(model, sampler, transition, state; kwargs...)

Return an iterator with elements of the form `(name, value)` for additional statistics in `transition`.

Default implementation returns an empty iterator.
"""
extras(transition; kwargs...) = ()
extras(transition, state; kwargs...) = extras(transition; kwargs...)
extras(model, sampler, transition, state; kwargs...) = extras(transition, state; kwargs...)

function (cb::TensorBoardCallback)(rng, model, sampler, transition, state, iteration; kwargs...)
    stats = cb.stats
    lg = cb.logger
    filterf = Base.Fix1(filter_param_and_value, cb)

    # TODO: Should we use the explicit interface for TensorBoardLogger?
    with_logger(lg) do
        for (k, val) in Iterators.filter(filterf, params_and_values(transition, state; kwargs...))
            stat = stats[k]

            # Log the raw value
            @info "$(cb.param_prefix)$k" val

            # Update statistic estimators
            OnlineStats.fit!(stat, val)

            # Need some iterations before we start showing the stats
            @info "$(cb.param_prefix)$k" stat
        end

        # Transition statstics
        if cb.include_extras
            for (name, val) in extras(transition, state; kwargs...)
                @info "$(cb.extras_prefix)$(name)" val
            end
        end
        # Increment the step for the logger.
        TensorBoardLogger.increment_step!(lg, 1)
    end
end
