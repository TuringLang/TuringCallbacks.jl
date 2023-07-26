using TuringCallbacks
using Documenter

makedocs(;
    modules = [TuringCallbacks],
    authors = "Tor",
    repo = "https://github.com/TuringLang/TuringCallbacks.jl/blob/{commit}{path}#L{line}",
    sitename = "TuringCallbacks.jl",
    format = Documenter.HTML(;
        prettyurls = get(ENV, "CI", "false") == "true",
        canonical = "https://turinglang.github.io/TuringCallbacks.jl",
        assets = String[],
    ),
    pages = ["Home" => "index.md"],
)

deploydocs(; repo = "github.com/TuringLang/TuringCallbacks.jl")
