Base.@kwdef struct NameFilter{A,B}
    include::A=nothing
    exclude::B=nothing
end

(f::NameFilter)(name, value) = f(name)
function (f::NameFilter)(name)
    include, exclude = f.include, f.exclude
    (exclude === nothing || name ∉ exclude) && (include === nothing || name ∈ include)
end
