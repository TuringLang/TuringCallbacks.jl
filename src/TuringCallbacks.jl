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

include("stats.jl")
include("tensorboardlogger.jl")
include("callbacks/tensorboard.jl")

end
