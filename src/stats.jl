###################
### OnlineStats ###
###################
"""
$(TYPEDEF)

# Usage

    Skip(b::Int, stat::OnlineStat)

Skips the first `b` observations before passing them on to `stat`.
"""
mutable struct Skip{T, O<:OnlineStat{T}} <: OnlineStat{T}
    b::Int
    current_index::Int
    stat::O
end

Skip(b::Int, stat) = Skip(b, 0, stat)

OnlineStats.nobs(o::Skip) = OnlineStats.nobs(o.stat)
OnlineStats.value(o::Skip) = OnlineStats.value(o.stat)
function OnlineStats._fit!(o::Skip, x::Real)
    if o.current_index > o.b
        OnlineStats._fit!(o.stat, x)
    end
    o.current_index += length(x)

    return o
end

Base.show(io::IO, o::Skip) = print(
    io,
    "Skip ($(o.b)): current_index=$(o.current_index) | stat=$(o.stat)`"
)

"""
$(TYPEDEF)

# Usage

    Thin(b::Int, stat::OnlineStat)

Thins `stat` with an interval `b`, i.e. only passes every b-th observation to `stat`.
"""
mutable struct Thin{T, O<:OnlineStat{T}} <: OnlineStat{T}
    b::Int
    current_index::Int
    stat::O
end

Thin(b::Int, stat) = Thin(b, 0, stat)

OnlineStats.nobs(o::Thin) = OnlineStats.nobs(o.stat)
OnlineStats.value(o::Thin) = OnlineStats.value(o.stat)
function OnlineStats._fit!(o::Thin, x::Real)
    if (o.current_index % o.b) == 0
        OnlineStats._fit!(o.stat, x)
    end
    o.current_index += length(x)

    return o
end

Base.show(io::IO, o::Thin) = print(
    io,
    "Thin ($(o.b)): current_index=$(o.current_index) | stat=$(o.stat)`"
)

"""
$(TYPEDEF)

# Usage

    WindowStat(b::Int, stat::O) where {O <: OnlineStat}

"Wraps" `stat` in a `MovingWindow` of length `b`.

`value(o::WindowStat)` will then return an `OnlineStat` of the same type as 
`stat`, which is *only* fitted on the batched data contained in the `MovingWindow`.

"""
struct WindowStat{T, O} <: OnlineStat{T}
    window::MovingWindow{T}
    stat::O
end

WindowStat(b::Int, T::Type, o) = WindowStat{T, typeof(o)}(MovingWindow(b, T), o)
WindowStat(b::Int, o::OnlineStat{T}) where {T} = WindowStat{T, typeof(o)}(
    MovingWindow(b, T), o
)

# Proxy methods to the window
OnlineStats.nobs(o::WindowStat) = OnlineStats.nobs(o.window)
OnlineStats._fit!(o::WindowStat, x) = OnlineStats._fit!(o.window, x)

function OnlineStats.value(o::WindowStat{<:Any, <:OnlineStat})
    stat_new = deepcopy(o.stat)
    fit!(stat_new, OnlineStats.value(o.window))
    return stat_new
end

function OnlineStats.value(o::WindowStat{<:Any, <:Function})
    stat_new = o.stat()
    fit!(stat_new, OnlineStats.value(o.window))
    return stat_new
end
