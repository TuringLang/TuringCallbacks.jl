using Dates

"""
    $(TYPEDEF)

Wraps a `CoreLogging.AbstractLogger` to construct a callback to be
passed to `AbstractMCMC.step`.

# Usage

    TensorBoardCallback(; kwargs...)
    TensorBoardCallback(directory::string[, stats]; kwargs...)
    TensorBoardCallback(lg::AbstractLogger[, stats]; kwargs...)

Constructs an instance of a `TensorBoardCallback`, creating a `TBLogger` if `directory` is
provided instead of `lg`.

## Arguments
- `lg`: an instance of an `AbstractLogger` which implements `TuringCallbacks.increment_step!`.
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
struct TensorBoardCallback{L,F1,F2,F3}
    "Underlying logger."
    logger::AbstractLogger
    "Lookup for variable name to statistic estimate."
    stats::L
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

maybe_filter(f; kwargs...) = f
maybe_filter(::Nothing; exclude=nothing, include=nothing) = NameFilter(; exclude, include)

function TensorBoardCallback(
    lg::AbstractLogger,
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
    param_prefix::String = "",
    extras_prefix::String = "extras/",
    kwargs...
)
    # Create the filters.
    variable_filter_f = maybe_filter(filter; include=include, exclude=exclude)
    extras_filter_f = maybe_filter(
        extras_filter; include=extras_include, exclude=extras_exclude
    )
    hyperparams_filter_f = maybe_filter(
        hyperparams_filter; include=hyperparams_include, exclude=hyperparams_exclude
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
        lg,
        stats_lookup,
        variable_filter_f,
        include_extras,
        extras_filter_f,
        include_hyperparams,
        hyperparams_filter_f,
        param_prefix,
        extras_prefix
    )
end

"""
    filter_param_and_value(cb::TensorBoardCallback, param_name, value)

Filter parameters and values from a `transition` based on the `filter` of `cb`.
"""
function filter_param_and_value(cb::TensorBoardCallback, param, value)
    return cb.variable_filter(param, value)
end
function filter_param_and_value(cb::TensorBoardCallback, param_and_value::Tuple)
    filter_param_and_value(cb, param_and_value...)
end

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
function params_and_values(model, sampler, transition, state; kwargs...)
    return params_and_values(transition, state; kwargs...)
end

"""
    extras(transition[, state]; kwargs...)
    extras(model, sampler, transition, state; kwargs...)

Return an iterator with elements of the form `(name, value)` for additional statistics in `transition`.

Default implementation returns an empty iterator.
"""
extras(transition; kwargs...) = ()
extras(transition, state; kwargs...) = extras(transition; kwargs...)
extras(model, sampler, transition, state; kwargs...) = extras(transition, state; kwargs...)

"""
    filter_extras_and_value(cb::TensorBoardCallback, name, value)

Filter extras and values from a `transition` based on the `filter` of `cb`.
"""
function filter_extras_and_value(cb::TensorBoardCallback, name, value)
    return cb.extras_filter(name, value)
end
function filter_extras_and_value(cb::TensorBoardCallback, name_and_value::Tuple)
    return filter_extras_and_value(cb, name_and_value...)
end

"""
    hyperparams(model, sampler[, transition, state]; kwargs...)

Return an iterator with elements of the form `(name, value)` for hyperparameters in `model`.
"""
hyperparams(model, sampler; kwargs...) = Pair{String, Any}[]
function hyperparams(model, sampler, transition, state; kwargs...)
    return hyperparams(model, sampler; kwargs...)
end

"""
    filter_hyperparams_and_value(cb::TensorBoardCallback, name, value)

Filter hyperparameters and values from a `transition` based on the `filter` of `cb`.
"""
function filter_hyperparams_and_value(cb::TensorBoardCallback, name, value)
    return cb.hyperparam_filter(name, value)
end
function filter_hyperparams_and_value(cb::TensorBoardCallback, name_and_value::Tuple)
    return filter_hyperparams_and_value(cb, name_and_value...)
end

"""
    hyperparam_metrics(model, sampler[, transition, state]; kwargs...)

Return a `Vector{String}` of metrics for hyperparameters in `model`.
"""
hyperparam_metrics(model, sampler; kwargs...) = String[]
function hyperparam_metrics(model, sampler, transition, state; kwargs...)
    return hyperparam_metrics(model, sampler; kwargs...)
end

increment_step!(lg::TensorBoardLogger.TBLogger, Δ_Step) =
    TensorBoardLogger.increment_step!(lg, Δ_Step)

function (cb::TensorBoardCallback)(rng, model, sampler, transition, state, iteration; kwargs...)
    stats = cb.stats
    lg = cb.logger
    variable_filter = Base.Fix1(filter_param_and_value, cb)
    extras_filter = Base.Fix1(filter_extras_and_value, cb)
    hyperparams_filter = Base.Fix1(filter_hyperparams_and_value, cb)

    if iteration == 1 && cb.include_hyperparams
        # If it's the first iteration, we write the hyperparameters.
        TensorBoardLogger.write_hparams!(
            lg,
            Dict(
                Iterators.filter(
                    hyperparams_filter,
                    hyperparams(model, sampler, transition, state; kwargs...)
                )
            ),
            hyperparam_metrics(model, sampler)
        )
    end


    # TODO: Should we use the explicit interface for TensorBoardLogger?
    with_logger(lg) do
        for (k, val) in Iterators.filter(
            variable_filter,
            params_and_values(transition, state; kwargs...)
        )
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
            for (name, val) in Iterators.filter(
                extras_filter,
                extras(transition, state; kwargs...)
            )
                @info "$(cb.extras_prefix)$(name)" val

                # TODO: Make this customizable.
                if val isa Real
                    stat = stats["$(cb.extras_prefix)$(name)"]
                    fit!(stat, float(val))
                    @info ("$(cb.extras_prefix)$(name)") stat
                end
            end
        end
        # Increment the step for the logger.
        increment_step!(lg, 1)
    end
end
