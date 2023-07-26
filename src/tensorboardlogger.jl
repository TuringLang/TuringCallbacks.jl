#########################################
### Overloads for `TensorBoardLogger` ###
#########################################
# `tb_name` is used by `preprocess` to decide how a given `arg` should look
"""
    tb_name(args...)

Returns a `string` representing the name for `arg` or `args` in TensorBoard.

If `length(args) > 1`, `args` are joined together by `"/"`.
"""
tb_name(arg) = string(arg)
tb_name(stat::OnlineStat) = string(nameof(typeof(stat)))
tb_name(o::Skip) = "Skip($(o.b))"
tb_name(o::Thin) = "Thin($(o.b))"
tb_name(o::WindowStat) = "WindowStat($(o.window.b))"
tb_name(o::AutoCov, b::Int) = "AutoCov(lag=$b)/corr"

# Recursive impl
tb_name(s1::String, s2::String) = s1 * "/" * s2
tb_name(arg1, arg2) = tb_name(arg1) * "/" * tb_name(arg2)
tb_name(arg, args...) = tb_name(arg) * "/" * tb_name(args...)

function TBL.preprocess(name, stat::OnlineStat, data)
    if OnlineStats.nobs(stat) > 0
        TBL.preprocess(tb_name(name, stat), value(stat), data)
    end
end

function TBL.preprocess(name, stat::Skip, data)
    return TBL.preprocess(tb_name(name, stat), stat.stat, data)
end

function TBL.preprocess(name, stat::Thin, data)
    return TBL.preprocess(tb_name(name, stat), stat.stat, data)
end

function TBL.preprocess(name, stat::WindowStat, data)
    return TBL.preprocess(tb_name(name, stat), value(stat), data)
end

function TBL.preprocess(name, stat::AutoCov, data)
    autocor = OnlineStats.autocor(stat)
    for b = 1:(stat.lag.b-1)
        # `autocor[i]` corresponds to the lag of size `i - 1` and `autocor[1] = 1.0`
        bname = tb_name(stat, b)
        TBL.preprocess(tb_name(name, bname), autocor[b+1], data)
    end
end

function TBL.preprocess(name, stat::Series, data)
    # Iterate through the stats and process those independently
    for s in stat.stats
        TBL.preprocess(name, s, data)
    end
end

function TBL.preprocess(name, hist::KHist, data)
    if OnlineStats.nobs(hist) > 0
        # Creates a NORMALIZED histogram
        edges = OnlineStats.edges(hist)
        cnts = OnlineStats.counts(hist)
        TBL.preprocess(name, (edges, cnts ./ sum(cnts)), data)
    end
end

# Unlike the `preprocess` overload, this allows us to specify if we want to normalize
function TBL.log_histogram(
    logger::AbstractLogger,
    name::AbstractString,
    hist::OnlineStats.HistogramStat;
    step = nothing,
    normalize = false,
)
    edges = edges(hist)
    cnts = Float64.(OnlineStats.counts(hist))
    if normalize
        return TBL.log_histogram(logger, name, (edges, cnts ./ sum(cnts)); step = step)
    else
        return TBL.log_histogram(logger, name, (edges, cnts); step = step)
    end
end
