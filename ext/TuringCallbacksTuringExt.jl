module TuringCallbacksTuringExt

if isdefined(Base, :get_extension)
    using Turing: Turing, DynamicPPL
    using TuringCallbacks: TuringCallbacks
else
    # Requires compatible.
    using ..Turing: Turing, DynamicPPL
    using ..TuringCallbacks: TuringCallbacks
end

const TuringTransition = Union{
    Turing.Inference.Transition,
    Turing.Inference.SMCTransition,
    Turing.Inference.PGTransition
}

function TuringCallbacks.params_and_values(
    model::DynamicPPL.Model,
    transition::TuringTransition;
    kwargs...
)
    vns, vals = Turing.Inference._params_to_array(model, [transition])
    return zip(Iterators.map(string, vns), vals)
end

function TuringCallbacks.extras(
    model::DynamicPPL.Model, transition::TuringTransition;
    kwargs...
)
    names, vals = Turing.Inference.get_transition_extras([transition])
    return zip(string.(names), vec(vals))
end

default_hyperparams(sampler::DynamicPPL.Sampler) = default_hyperparams(sampler.alg)
default_hyperparams(alg::Turing.Inference.InferenceAlgorithm) = (
    string(f) => getfield(alg, f) for f in fieldnames(typeof(alg)) if f != :adtype
)

const AlgsWithDefaultHyperparams = Union{
    Turing.Inference.HMC,
    Turing.Inference.HMCDA,
    Turing.Inference.NUTS,
    Turing.Inference.SGHMC,
    
}

function TuringCallbacks.hyperparams(
    model::DynamicPPL.Model,
    sampler::DynamicPPL.Sampler{<:AlgsWithDefaultHyperparams};
    kwargs...
)
    return default_hyperparams(sampler)
end

function TuringCallbacks.hyperparam_metrics(
    model,
    sampler::DynamicPPL.Sampler{<:Turing.Inference.NUTS}
)
    return [
        "extras/acceptance_rate/stat/Mean",
        "extras/max_hamiltonian_energy_error/stat/Mean",
        "extras/lp/stat/Mean",
        "extras/n_steps/stat/Mean",
        "extras/tree_depth/stat/Mean"
    ]
end

end
