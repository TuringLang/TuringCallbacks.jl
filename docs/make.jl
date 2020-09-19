using TuringCallbacks
using Documenter

makedocs(;
    modules=[TuringCallbacks],
    authors="Tor",
    repo="https://github.com/torfjelde/TuringCallbacks.jl/blob/{commit}{path}#L{line}",
    sitename="TuringCallbacks.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://torfjelde.github.io/TuringCallbacks.jl",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/torfjelde/TuringCallbacks.jl",
)
