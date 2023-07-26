###############################
### Saves samples on the go ###
###############################

"""
    SaveCSV

A callback saves samples to .csv file during sampling
"""
function SaveCSV(
    rng::AbstractRNG,
    model::Model,
    sampler::Sampler,
    transition,
    state,
    iteration::Int64;
    kwargs...,
)
    SaveCSV(model, sampler, transition, state.vi, iteration; kwargs...)
end

function SaveCSV(
    rng::AbstractRNG,
    model::Model,
    sampler::Sampler,
    transition,
    vi::AbstractVarInfo,
    iteration::Int64;
    kwargs...,
)
    vii = deepcopy(vi)
    invlink!!(vii, model)
    θ = vii[sampler]
    # it would be good to have the param names as in the chain
    chain_name = get(kwargs, :chain_name, "chain")
    write(string(chain_name, ".csv"), Dict("params" => [θ]); append = true, delim = ";")
end
