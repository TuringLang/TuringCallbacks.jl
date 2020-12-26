module TuringCallbacks

using Reexport

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

abstract type TuringCallback end

include("stats.jl")
include("tensorboardlogger.jl")
include("online_stats_plots.jl")
include(joinpath("callbacks", "tensorboard.jl"))
include(joinpath("callbacks", "abstractplotting.jl"))

end
