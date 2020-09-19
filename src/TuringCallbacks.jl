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

"""
    centers_to_edges(centers)

Returns a vector of length `length(centers) + 1`, whose elements represents the edges of the
bins, rather than the `centers`.

This is useful for converting something like a `OnlineStats.KHist` to a 
`StatsBase.Histogram` since the former uses the *centers* of the bins while the latter
requires the *edges* of the bins.

# Examples
```jldoctest
julia> using TuringCallbacks: OnlineStats, StatsBase, centers_to_edges

julia> xs = 1.:20;

julia> khist = OnlineStats.fit!(OnlineStats.KHist(10), xs)
KHist: n=20 | value=(centers = [1.0, 3.5, 6.5, 8.5, 10.5, 12.5, 14.5, 17.0, 19.0, 20.0], counts = [1, 4, 2, 2, 2, 2, 2, 3, 1, 1])

julia> centers, cnts = OnlineStats.value(khist)
(centers = [1.0, 3.5, 6.5, 8.5, 10.5, 12.5, 14.5, 17.0, 19.0, 20.0], counts = [1, 4, 2, 2, 2, 2, 2, 3, 1, 1])

julia> StatsBase.Histogram(centers_to_edges(centers), cnts, :left, false)
StatsBase.Histogram{Int64,1,Tuple{Array{Float64,1}}}
edges:
  [-0.25, 2.25, 5.0, 7.5, 9.5, 11.5, 13.5, 15.75, 18.0, 19.5, 20.5]
weights: [1, 4, 2, 2, 2, 2, 2, 3, 1, 1]
closed: left
isdensity: false
```
"""
function centers_to_edges(centers)
    # Find the midpoint between the nearby centers.
    intermediate = map(2:length(centers)) do i
        # Pick the left mid-point
        (centers[i] + centers[i - 1]) / 2
    end
    # Left-most point
    Δ_l = (centers[2] - centers[1]) / 2
    leftmost = centers[1] - Δ_l

    # Right-most point
    Δ_r = (centers[end] - centers[end - 1]) / 2
    rightmost = centers[end] + Δ_r

    return vcat([leftmost], intermediate, [rightmost])
end

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
    hist_raw = value(hist)
    edges = centers_to_edges(hist_raw.centers)
    cnts = hist_raw.counts ./ sum(hist_raw.counts)
    return TensorBoardLogger.preprocess(name, Histogram(edges, cnts, :left, true), data)
end

# Unlike the `preprocess` overload, this allows us to specify if we want to normalize
function TensorBoardLogger.log_histogram(
    logger::TBLogger, name::AbstractString, hist::OnlineStats.HistogramStat;
    step=nothing, normalize=false
)
    hist_raw = value(hist)
    edges = centers_to_edges(hist_raw.centers)
    cnts = Float64.(hist_raw.counts)
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
                        @info "$k" hist_window

                        # Because the `Distribution` and `Histogram` functionality in
                        # TB is quite crude, we additionally log "later" values to provide
                        # a slightly more useful view of the later samples in the chain.
                        # TODO: make this, say, 25% of the total number of iterations
                        if iteration > 0.25 * num_samples
                            @info "$k/late" stat
                            @info "$k/late" hist_window
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
