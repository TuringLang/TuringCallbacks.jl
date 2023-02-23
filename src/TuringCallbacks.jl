module TuringCallbacks

using Reexport

using LinearAlgebra
using Logging
using DocStringExtensions

@reexport using OnlineStats # used to compute different statistics on-the-fly

using TensorBoardLogger
const TBL = TensorBoardLogger

using DataStructures: DefaultDict

if !isdefined(Base, :get_extension)
    using Requires
end

export TensorBoardCallback, DefaultDict, WindowStat, Thin, Skip

include("stats.jl")
include("tensorboardlogger.jl")
include("callbacks/tensorboard.jl")

if !isdefined(Base, :get_extension)
    function __init__()
        @require Turing="fce5fe82-541a-59a6-adf8-730c64b5f9a0" include("../ext/TuringCallbacksTuringExt.jl")
    end
end

end
