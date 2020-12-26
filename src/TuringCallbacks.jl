module TuringCallbacks

using Reexport
using Requires

using LinearAlgebra
using Logging
using DocStringExtensions
import Turing
@reexport using OnlineStats # used to compute different statistics on-the-fly

using TensorBoardLogger
const TBL = TensorBoardLogger

import DataStructures: DefaultDict

export TensorBoardCallback, DefaultDict, WindowStat, Thin, Skip
export MakieCallBack

function __init__()
    @require AbstractPlotting="537997a7-5e4e-5d89-9595-2241ea00577e" include("makiecallback.jl")
end

abstract type TuringCallback end

include("stats.jl")
include("tensorboardlogger.jl")
include(joinpath("callbacks", "tensorboard.jl"))

end
