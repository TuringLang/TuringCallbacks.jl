struct Filter{A,B}
    include::A
    exclude::B
end
Filter(; include=nothing, exclude=nothing) = Filter(include, exclude)

function (f::Filter)(x)
    include, exclude = f.include, f.exclude
    return (exclude === nothing || x ∉ exclude) && (include === nothing || x ∈ include)
end
