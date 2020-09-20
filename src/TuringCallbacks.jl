module TuringCallbacks

export make_callback, TensorBoardCallback

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
### `TensorBoardCallback` ###
####################
"""
    TensorBoardCallback(directory::String)
    TensorBoardCallback(logger::TBLogger)

Wraps a `TensorBoardLogger.TBLogger` to construct a callback to be passed to
`Turing.sample`.

See also [`make_callback`](@ref).

TODO: add some information about the stats logged.

# Fields
$(TYPEDFIELDS)
"""
struct TensorBoardCallback{T1, T2}
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
    "If non-empty, statistics for only these variables will be logged."
    exclude::Vector{Symbol}
    "Include extra statistics from transitions."
    include_extras::Bool
    "Lookup for variable name to statistic estimate."
    estimators::T1
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
    exclude = Symbol[],
    include_extras::Bool = true
)    
    # Lookups
    estimators = Dict{String, typeof(make_estimator(TensorBoardCallback, num_bins))}()
    buffers = Dict{String, typeof(make_buffer(TensorBoardCallback, window))}()
    
    return TensorBoardCallback(
        lg, num_samples, num_bins, window, window_num_bins,
        exclude, include_extras, estimators, buffers
    )
end

make_estimator(cb::TensorBoardCallback) = make_estimator(typeof(cb), cb.num_bins)
make_estimator(cb::Type{<:TensorBoardCallback}, num_bins::Int) = OnlineStats.Series(
    Mean(),     # Online estimator for the mean
    Variance(), # Online estimator for the variance
    KHist(num_bins)  # Online estimator of a histogram with `100` bins
)
make_buffer(cb::TensorBoardCallback) = make_buffer(typeof(cb), cb.window)
make_buffer(cb::Type{<:TensorBoardCallback}, window::Int) = MovingWindow(Float64, window)

function (cb::TensorBoardCallback)(rng, model, sampler, transition, iteration)
    estimators = cb.estimators
    buffers = cb.buffers
    lg = cb.logger
    
    with_logger(lg) do
        for (varname, (vals, ks)) in pairs(transition.θ)
            # Skip those variables which are to be excluded
            if varname ∈ cb.exclude
                continue
            end
            
            for (k, val) in zip(ks, vals)
                if !haskey(estimators, k)
                    estimators[k] = make_estimator(cb)
                end
                stat = estimators[k]

                if !haskey(buffers, k)
                    buffers[k] = make_buffer(cb)
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
                    hist_window = fit!(KHist(cb.window_num_bins), value(buffer))

                    @info "$k" stat
                    @info "$k/stat" hist_window

                    # Because the `Distribution` and `Histogram` functionality in
                    # TB is quite crude, we additionally log "later" values to provide
                    # a slightly more useful view of the later samples in the chain.
                    # TODO: make this, say, 25% of the total number of iterations
                    if iteration > 0.25 * cb.num_samples
                        @info "$k/late" stat
                        @info "$k/late/stat" hist_window
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

end
