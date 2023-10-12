"""
    MultiCallback

A callback that combines multiple callbacks into one.

Implements [`push!!`](@ref) to add callbacks to the list.
"""
struct MultiCallback{Cs}
    callbacks::Cs
end

MultiCallback() = MultiCallback(())
MultiCallback(callbacks...) = MultiCallback(callbacks)

(c::MultiCallback)(args...; kwargs...) = foreach(c -> c(args...; kwargs...), c.callbacks)

"""
    push!!(cb::MultiCallback, callback)

Add a callback to the list of callbacks, mutating if possible.
"""
push!!(c::MultiCallback{<:Tuple}, callback) = MultiCallback((c.callbacks..., callback))
push!!(c::MultiCallback{<:AbstractArray}, callback) = (push!(c.callbacks, callback); return c)
