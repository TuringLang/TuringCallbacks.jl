"""
    $(TYPEDEF)

Wraps a `AbstractPlotting.Scene` to construct a callback to be passed to
`Turing.sample`.

# Usage

    MakieCallback()

Constructs an instance of a `MakieCallback`, creating an AbstractPlotting `Scene`

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
struct MakieCallback <: TuringCallback
    "Scene containing the plots"
    scene::Scene
    "Data storage for each variable"
    data::Dict{Symbol, MovingWindow}
    "Storage of each axis"
    axis_dict::Dict
    "Storage of the variables"
    vars::Dict{Symbol, Any}
    "Storage of the scene parameters"
    params::Dict{Any, Any}
    "Storage of the current iteration (as an observable)"
    iter::Observable{Int64}
end

function MakieCallback(model::DynamicPPL.Model, plots::Union{Series, AbstractVector} = [:histkde, Mean(), Variance(), AutoCov(20)]; kwargs...)
    variables = DynamicPPL.VarInfo(model).metadata
    return MakieCallback(Dict(Pair.(keys(variables), Ref(plots))),
    Dict(kwargs...))
end

function MakieCallBack(varsdict::Dict; kwargs...)
    return MakieCallback(varsdict, Dict(kwargs...))
end

function MakieCallback(vars::Dict, params::Dict)
    # Create a scene and a layout
    outer_padding = 5
    scene, layout = layoutscene(outer_padding, resolution = (1200, 700))
    display(scene)

    window = get!(params, :window, 1000)

    n_rows = length(keys(vars))
    n_cols = maximum(length.(values(vars))) 
    n_plots = n_rows * n_cols
    iter = Node(0)
    data = Dict{Symbol, MovingWindow}(:iter => MovingWindow(window, Int64))
    obs = Dict{Symbol, Any}()
    axis_dict = Dict()
    for (i, (variable, plots)) in enumerate(vars)
        data[variable] = MovingWindow(window, Float32)
        axis_dict[(variable, :varname)] = layout[i, 1, Left()] = LText(scene, string(variable), textsize = 30)
        axis_dict[(variable, :varname)].padding = (0, 50, 0, 0)
        onlineplot!(scene, layout, axis_dict, plots, iter, data, variable, i)
    end
    on(iter) do i
        if i > 10 # To deal with autolimits a certain number of samples are needed
            for (variable, plots) in vars
                for p in plots
                    autolimits!(axis_dict[(variable, p)])
                end
            end
        end
    end
    MakieLayout.trim!(layout)
    MakieCallback(scene, data, axis_dict, vars, params, iter)
end

function addIO!(cb::MakieCallback, io)
    cb.params[:io] = io
end

function (cb::MakieCallback)(rng, model, sampler, transition, iteration)
    fit!(cb.data[:iter], iteration)
    for (vals, ks) in values(transition.θ)
        for (k, val) in zip(ks, vals)
            if haskey(cb.data, Symbol(k))
                fit!(cb.data[Symbol(k)], Float32(val))
            end
        end
    end
    cb.iter[] += 1
    if haskey(cb.params, :io) 
        recordframe!(cb.params[:io])
    end
end
