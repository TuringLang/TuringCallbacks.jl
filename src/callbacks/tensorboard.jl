#############################
### `TensorBoardCallback` ###
#############################
"""
    $(TYPEDEF)

Wraps a `TensorBoardLogger.TBLogger` to construct a callback to be passed to
`Turing.sample`.

# Usage

    TensorBoardCallback(lg::TBLogger, num_samples::Int; kwargs...)
    TensorBoardCallback(directory::String, num_samples::Int; kwargs...)

Constructs an instance of a `TensorBoardCallback`, creating a `TBLogger` if `directory` is 
provided instead of `lg`.

## Arguments
- `num_samples::Int`: Total number of MCMC steps that will be taken.

## Keyword arguments
- `num_bins::Int = 100`: Number of bins to use in the histograms.
- `window::Int = min(num_samples, 1_000)`: Size of the window to compute stats for.
- `window_num_bins::Int = 50`: Number of bins to use in the histogram of the small window.
- `stats = nothing`: Lookup for variable name to statistic estimator. 
  If `isnothing`, then a `DefaultDict` with a default constructor returning a
  `OnlineStats.Series` estimator with `OnlineStats.Mean()`, `OnlineStats.Variance()`, and
  `OnlineStats.KHist(num_bins)` will be used.
- `buffers = nothing`: Lookup for variable name to window buffers. 
  If `isnothing`, then a `OnlineStats.MovingWindow(Float64, window)` will be used.
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
struct TensorBoardCallback{F, T1, T2}
    "Underlying logger."
    logger::TBLogger
    "Total number of MCMC steps that will be taken."
    num_samples::Int
    "Number of bins to use in the histogram."
    num_bins::Int
    "Size of the window to compute stats for."
    window::Int
    "Number of bins to use in the histogram of the small window."
    window_num_bins::Int
    "Filter determining whether or not we should log stats for a particular variable."
    variable_filter::F
    "Include extra statistics from transitions."
    include_extras::Bool
    "Lookup for variable name to statistic estimate."
    stats::T1
    "Lookup for variable name to window buffers."
    buffers::T2
end

function TensorBoardCallback(directory::String, args...; kwargs...)
    # Set up the logger
    lg = TBLogger(directory, min_level=Logging.Info; step_increment=0)

    return TensorBoardCallback(lg, args...; kwargs...)
end

function TensorBoardCallback(
    lg::TBLogger,
    num_samples::Int;
    num_bins::Int = 100,
    window::Int = min(num_samples, 1_000),
    window_num_bins::Int = 50,
    stats = nothing,
    buffers = nothing,
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
    stats = if !isnothing(stats)
        stats
    else
        make_estimator() = OnlineStats.Series(
            OnlineStats.Mean(),     # Online estimator for the mean
            OnlineStats.Variance(), # Online estimator for the variance
            OnlineStats.KHist(num_bins)  # Online estimator of a histogram with `100` bins
        )
        DefaultDict{String, typeof(make_estimator())}(make_estimator)
    end

    buffers = if !isnothing(buffers)
        buffers
    else
        make_buffer() = MovingWindow(Float64, window)
        DefaultDict{String, typeof(make_buffer())}(make_buffer)
    end
    
    return TensorBoardCallback(
        lg, num_samples, num_bins, window, window_num_bins,
        filter, include_extras, stats, buffers
    )
end

function (cb::TensorBoardCallback)(rng, model, sampler, transition, iteration)
    stats = cb.stats
    buffers = cb.buffers
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
                buffer = buffers[k]
                
                # Log the raw value
                @info k val

                # Update buffer and estimator
                fit!(buffer, val)
                fit!(stat, val)

                # Need some iterations before we start showing the stats
                if iteration > 1
                    @info "$k" stat

                    # Because the `Distribution` and `Histogram` functionality in
                    # TB is quite crude, we additionally log "later" values to provide
                    # a slightly more useful view of the later samples in the chain.
                    # TODO: make this, say, 25% of the total number of iterations
                    if iteration > 0.25 * cb.num_samples
                        @info "$k/late" stat
                    end
                end
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
