module TuringCallbacks

export make_callback, TBCallback

using Turing
using TensorBoardLogger, Logging
using OnlineStats      # used to compute different statistics on-the-fly
using StatsBase        # Provides us with the `Histogram` which is supported by `TensorBoardLogger.jl`
using LinearAlgebra

using DocStringExtensions

const TBL = TensorBoardLogger

#########################################
### Overloads for `TensorBoardLogger` ###
#########################################
function TensorBoardLogger.preprocess(name, stat::OnlineStat, data)
    return TensorBoardLogger.preprocess(name, value(stat), data)
end

function TensorBoardLogger.preprocess(name, stat::OnlineStats.Series, data)
    # Iterate through the stats and process those independently
    for s in stat.stats
        s_name = string(nameof(typeof(s)))
        TensorBoardLogger.preprocess(name * "/" * s_name, s, data)
    end
end

function TensorBoardLogger.preprocess(name, hist::KHist, data)
    # Creates a NORMALIZED histogram
    edges = OnlineStats.edges(hist)
    cnts = OnlineStats.counts(hist)
    return TensorBoardLogger.preprocess(
        name, Histogram(edges, cnts ./ sum(cnts), :left, true), data
    )
end

# Unlike the `preprocess` overload, this allows us to specify if we want to normalize
function TensorBoardLogger.log_histogram(
    logger::TBLogger, name::AbstractString, hist::OnlineStats.HistogramStat;
    step=nothing, normalize=false
)
    edges = edges(hist)
    cnts = Float64.(OnlineStats.counts(hist))
    hist = if normalize
        Histogram(edges, cnts ./ sum(cnts), :left, true)
    else
        Histogram(edges, cnts, :left, false)
    end
    summ = TBL.SummaryCollection(TBL.histogram_summary(name, hist))
    TBL.write_event(logger.file, TBL.make_event(logger, summ, step=step))
end

####################
### `TBCallback` ###
####################
"""
    TBCallback(directory::String)
    TBCallback(logger::TBLogger)

Wraps a `TensorBoardLogger.TBLogger` to construct a callback to be passed to
`Turing.sample`.

See also [`make_callback`](@ref).

TODO: add some information about the stats logged.

# Fields
$(TYPEDFIELDS)
"""
struct TBCallback
    logger::TBLogger
end

function TBCallback(directory::String)
    # Set up the logger
    lg = TBLogger(directory, min_level=Logging.Info; step_increment=0)

    return TBCallback(lg)
end

make_estimator(cb::TBCallback, num_bins::Int) = OnlineStats.Series(
    Mean(),     # Online estimator for the mean
    Variance(), # Online estimator for the variance
    KHist(num_bins)  # Online estimator of a histogram with `100` bins
)
make_buffer(cb::TBCallback, window::Int) = MovingWindow(Float64, window)

"""
    make_callback(cb::TBCallback, spl::InferenceAlgorithm, num_samples::Int; kwargs...)

Returns a function which can be used as a callback in `Turing.sample`.

# Keyword arguments
- `num_bins::Int = 100`: number of bins to use in the histogram.
- `window::Int = min(num_samples, 1_000)`: size of the window to compute stats for.
- `window_num_bins::Int = 50`: number of bins to use in the histogram of the small window.
"""
function make_callback(
    cb::TBCallback,
    spl::Turing.InferenceAlgorithm, # used to extract sampler-specific parameters in the future
    num_samples::Int;
    num_bins::Int = 100,
    window::Int = min(num_samples, 1_000),
    window_num_bins::Int = 50
)
    lg = cb.logger

    # Lookups
    estimators = Dict{String, typeof(make_estimator(cb, num_bins))}()
    buffers = Dict{String, typeof(make_buffer(cb, window))}()

    return function callback(rng, model, sampler, transition, iteration)
        with_logger(lg) do
            for (vals, ks) in values(transition.θ)
                for (k, val) in zip(ks, vals)
                    if !haskey(estimators, k)
                        estimators[k] = make_estimator(cb, num_bins)
                    end
                    stat = estimators[k]

                    if !haskey(buffers, k)
                        buffers[k] = make_buffer(cb, window)
                    end
                    buffer = buffers[k]
                    
                    # Log the raw value
                    @info k val

                    # Update buffer and estimator
                    fit!(buffer, val)
                    fit!(stat, val)
                    mean, variance, hist = stat.stats

                    # Need some iterations before we start showing the stats
                    if iteration > 10
                        hist_window = fit!(KHist(window_num_bins), value(buffer))

                        @info "$k" stat
                        @info "$k/stat/" hist_window

                        # Because the `Distribution` and `Histogram` functionality in
                        # TB is quite crude, we additionally log "later" values to provide
                        # a slightly more useful view of the later samples in the chain.
                        # TODO: make this, say, 25% of the total number of iterations
                        if iteration > 0.25 * num_samples
                            @info "$k/late" stat
                            @info "$k/late/state/" hist_window
                        end
                    end
                end
            end

            # Transition statstics
            names, vals = Turing.Inference.get_transition_extras([transition])
            for (name, val) in zip(string.(names), vec(vals))
                @info ("extras/" * name) val
            end
            @info "" log_step_increment=1
        end
    end
end

end
