module TuringCallbacksTuringExt

if isdefined(Base, :get_extension)
    using Turing: Turing
    using TuringCallbacks: TuringCallbacks
else
    # Requires compatible.
    using ..Turing: Turing
    using ..TuringCallbacks: TuringCallbacks
end

const TuringTransition = Union{Turing.Inference.Transition,Turing.Inference.HMCTransition}

function TuringCallbacks.params_and_values(transition::TuringTransition; kwargs...)
    return Iterators.map(
        zip(Turing.Inference._params_to_array([transition])...),
    ) do (ksym, val)
        return string(ksym), val
    end
end

function TuringCallbacks.extras(transition::TuringTransition; kwargs...)
    return Iterators.map(
        zip(Turing.Inference.get_transition_extras([transition])...),
    ) do (ksym, val)
        return string(ksym), val
    end
end

end
