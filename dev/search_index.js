var documenterSearchIndex = {"docs":
[{"location":"","page":"Home","title":"Home","text":"CurrentModule = TuringCallbacks\nDocTestSetup  = quote\n    using TuringCallbacks\nend","category":"page"},{"location":"","page":"Home","title":"Home","text":"using TuringCallbacks","category":"page"},{"location":"#TuringCallbacks","page":"Home","title":"TuringCallbacks","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"","category":"page"},{"location":"#Getting-started","page":"Home","title":"Getting started","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"As the package is not yet officially released, the package has to be added from the GitHub repository:","category":"page"},{"location":"","page":"Home","title":"Home","text":"julia> ]\npkg> add TuringCallbacks.jl","category":"page"},{"location":"#Visualizing-sampling-on-the-fly","page":"Home","title":"Visualizing sampling on-the-fly","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"TensorBoardCallback is a wrapper around TensorBoardLogger.TBLogger which can be used to create a callback compatible with Turing.sample.","category":"page"},{"location":"","page":"Home","title":"Home","text":"To actually visualize the results of the logging, you need to have installed tensorboard in Python. If you do not have tensorboard installed, it should hopefully be sufficient to just run","category":"page"},{"location":"","page":"Home","title":"Home","text":"pip3 install tensorboard","category":"page"},{"location":"","page":"Home","title":"Home","text":"Then you can start up the TensorBoard:","category":"page"},{"location":"","page":"Home","title":"Home","text":"python3 -m tensorboard.main --logdir tensorboard_logs/run","category":"page"},{"location":"","page":"Home","title":"Home","text":"Now we're ready to actually write some Julia code.","category":"page"},{"location":"","page":"Home","title":"Home","text":"The following snippet demonstrates the usage of TensorBoardCallback on a simple model.  This will write a set of statistics at each iteration to an event-file compatible with Tensorboard:","category":"page"},{"location":"","page":"Home","title":"Home","text":"using Turing, TuringCallbacks\n\n@model function demo(x)\n    s ~ InverseGamma(2, 3)\n    m ~ Normal(0, √s)\n    for i in eachindex(x)\n        x[i] ~ Normal(m, √s)\n    end\nend\n\nxs = randn(100) .+ 1;\nmodel = demo(xs);\n\n# Number of MCMC samples/steps\nnum_samples = 10_000\nnum_adapts = 100\n\n# Sampling algorithm to use\nalg = NUTS(num_adapts, 0.65)\n\n# Create the callback\ncallback = TensorBoardCallback(\"tensorboard_logs/run\")\n\n# Sample\nchain = sample(model, alg, num_samples; callback = callback)","category":"page"},{"location":"","page":"Home","title":"Home","text":"While this is sampling, you can head right over to localhost:6006 in your web browser and you should be seeing some plots!","category":"page"},{"location":"","page":"Home","title":"Home","text":"(Image: TensorBoard dashboard)","category":"page"},{"location":"","page":"Home","title":"Home","text":"In particular, note the \"Distributions\" tab in the above picture. Clicking this, you should see something similar to:","category":"page"},{"location":"","page":"Home","title":"Home","text":"(Image: TensorBoard dashboard)","category":"page"},{"location":"","page":"Home","title":"Home","text":"And finally, the \"Histogram\" tab shows a slighly more visually pleasing version of the marginal distributions:","category":"page"},{"location":"","page":"Home","title":"Home","text":"(Image: TensorBoard dashboard)","category":"page"},{"location":"","page":"Home","title":"Home","text":"Note that the names of the stats following a naming $variable_name/... where $variable_name refers to name of the variable in the model.","category":"page"},{"location":"#Choosing-what-and-how-you-log","page":"Home","title":"Choosing what and how you log","text":"","category":"section"},{"location":"#Statistics","page":"Home","title":"Statistics","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"In the above example we didn't provide any statistics explicit and so it used the default statistics, e.g. Mean and Variance. But using other statistics is easy! Here's a much more interesting example:","category":"page"},{"location":"","page":"Home","title":"Home","text":"# Create the stats (estimators are sub-types of `OnlineStats.OnlineStat`)\nstats = Skip(\n    num_adapts, # Consider adaptation steps\n    Series(\n        # Estimators using the entire chain\n        Series(Mean(), Variance(), AutoCov(10), KHist(100)),\n        # Estimators using the entire chain but only every 10-th sample\n        Thin(10, Series(Mean(), Variance(), AutoCov(10), KHist(100))),\n        # Estimators using only the last 1000 samples\n        WindowStat(1000, Series(Mean(), Variance(), AutoCov(10), KHist(100)))\n    )\n)\n# Create the callback\ncallback = TensorBoardCallback(\"tensorboard_logs/run\", stats)\n\n# Sample\nchain = sample(model, alg, num_samples; callback = callback)","category":"page"},{"location":"","page":"Home","title":"Home","text":"Tada! Now you should be seeing waaaay more interesting statistics in your TensorBoard dashboard. See the OnlineStats.jl documentation for more on the different statistics, with the exception of Thin, Skip and WindowStat which are implemented in this package.","category":"page"},{"location":"","page":"Home","title":"Home","text":"Note that these statistic estimators are stateful, and therefore the following is bad:","category":"page"},{"location":"","page":"Home","title":"Home","text":"s = AutoCov(5)\nstat = Series(s, s)\n# => 10 samples but `n=20` since we've called `fit!` twice for each observation\nfit!(stat, randn(10))","category":"page"},{"location":"","page":"Home","title":"Home","text":"while the following is good:","category":"page"},{"location":"","page":"Home","title":"Home","text":"stat = Series(AutoCov(5), AutoCov(5))\n# => 10 samples AND `n=10`; great!\nfit!(stat, randn(10))","category":"page"},{"location":"","page":"Home","title":"Home","text":"Since at the moment the only support statistics are sub-types of OnlineStats.OnlineStat. If you want to log some custom statistic, again, at the moment, you have to make a sub-type and implement OnlineStats.fit! and OnlineStats.value. By default, a OnlineStat is passed to tensorboard by simply calling OnlineStat.value(stat). Therefore, if you also want to customize how a stat is passed to tensorbord, you need to overload TensorBoardLogger.preprocess(name, stat, data) accordingly.","category":"page"},{"location":"#Filter-variables-to-log","page":"Home","title":"Filter variables to log","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"Maybe you want to only log stats for certain variables, e.g. in the above example we might want to exclude m and exclude the sampler statistics:","category":"page"},{"location":"","page":"Home","title":"Home","text":"callback = TensorBoardCallback(\n    \"tensorboard_logs/run\", stats;\n    exclude = [\"m\", ], include_extras = false\n)","category":"page"},{"location":"","page":"Home","title":"Home","text":"Or you can create the filter (a mapping variable_name -> ::Bool yourself:","category":"page"},{"location":"","page":"Home","title":"Home","text":"var_filter(varname) = varname != \"m\"\ncallback = TensorBoardCallback(\n    \"tensorboard_logs/run\", stats;\n    variable_filter = var_filter\n)","category":"page"},{"location":"#Types-and-Functions","page":"Home","title":"Types & Functions","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"Modules = [TuringCallbacks]\nPrivate = false\nOrder = [:type, :function]","category":"page"},{"location":"#TuringCallbacks.Skip","page":"Home","title":"TuringCallbacks.Skip","text":"mutable struct Skip{T, O<:OnlineStat{T}} <: OnlineStat{T}\n\nUsage\n\nSkip(b::Int, stat::OnlineStat)\n\nSkips the first b observations before passing them on to stat.\n\n\n\n\n\n","category":"type"},{"location":"#TuringCallbacks.TensorBoardCallback","page":"Home","title":"TuringCallbacks.TensorBoardCallback","text":"struct TensorBoardCallback{F, L}\n\nWraps a TensorBoardLogger.TBLogger to construct a callback to be passed to Turing.sample.\n\nUsage\n\nTensorBoardCallback(; kwargs...)\nTensorBoardCallback(directory::string[, stats]; kwargs...)\nTensorBoardCallback(lg::TBLogger[, stats]; kwargs...)\n\nConstructs an instance of a TensorBoardCallback, creating a TBLogger if directory is  provided instead of lg.\n\nArguments\n\nstats = nothing: OnlineStat or lookup for variable name to statistic estimator. If stats isa OnlineStat, we will create a DefaultDict which copies stats for unseen variable names. If isnothing, then a DefaultDict with a default constructor returning a OnlineStats.Series estimator with Mean(), Variance(), and KHist(num_bins) will be used.\n\nKeyword arguments\n\nnum_bins::Int = 100: Number of bins to use in the histograms.\nvariable_filter = nothing: Filter determining whether or not we should log stats for a  particular variable.  If isnothing a default-filter constructed from exclude and  include will be used.\nexclude = String[]: If non-empty, these variables will not be logged.\ninclude = String[]: If non-empty, only these variables will be logged.\ninclude_extras::Bool = true: Include extra statistics from transitions.\ndirectory::String = nothing: if specified, will together with comment be used to  define the logging directory.\ncomment::String = nothing: if specified, will together with directory be used to  define the logging directory.\n\nFields\n\nlogger::TensorBoardLogger.TBLogger: Underlying logger.\nvariable_filter::Any: Filter determining whether or not we should log stats for a particular variable.\ninclude_extras::Bool: Include extra statistics from transitions.\nstats::Any: Lookup for variable name to statistic estimate.\n\n\n\n\n\n","category":"type"},{"location":"#TuringCallbacks.Thin","page":"Home","title":"TuringCallbacks.Thin","text":"mutable struct Thin{T, O<:OnlineStat{T}} <: OnlineStat{T}\n\nUsage\n\nThin(b::Int, stat::OnlineStat)\n\nThins stat with an interval b, i.e. only passes every b-th observation to stat.\n\n\n\n\n\n","category":"type"},{"location":"#TuringCallbacks.WindowStat","page":"Home","title":"TuringCallbacks.WindowStat","text":"struct WindowStat{T, O} <: OnlineStat{T}\n\nUsage\n\nWindowStat(b::Int, stat::O) where {O <: OnlineStat}\n\n\"Wraps\" stat in a MovingWindow of length b.\n\nvalue(o::WindowStat) will then return an OnlineStat of the same type as  stat, which is only fitted on the batched data contained in the MovingWindow.\n\n\n\n\n\n","category":"type"},{"location":"#Internals","page":"Home","title":"Internals","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"Modules = [TuringCallbacks]\nPrivate = true\nPublic = false","category":"page"},{"location":"#TuringCallbacks.tb_name-Tuple{Any}","page":"Home","title":"TuringCallbacks.tb_name","text":"tb_name(args...)\n\nReturns a string representing the name for arg or args in TensorBoard.\n\nIf length(args) > 1, args are joined together by \"/\".\n\n\n\n\n\n","category":"method"},{"location":"#Index","page":"Home","title":"Index","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"","category":"page"}]
}
