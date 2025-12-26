using Test
using DynamicPPL
using Turing
using TuringCallbacks
using TensorBoardLogger, ValueHistories

Base.@kwdef struct CountingCallback
    count::Ref{Int}=Ref(0)
end

(c::CountingCallback)(args...; kwargs...) = c.count[] += 1

@model function demo(x)
    s ~ InverseGamma(2, 3)
    m ~ Normal(0, √s)
    for i in eachindex(x)
        x[i] ~ Normal(m, √s)
    end
end

function DynamicPPL.TestUtils.varnames(::DynamicPPL.Model{typeof(demo)})
    return [@varname(s), @varname(m)]
end

const demo_model = demo(randn(100) .+ 1)

@testset "TuringCallbacks.jl" begin
    include("multicallback.jl")
    include("tensorboardcallback.jl")
    
    # Run JuliaBUGS tests in a separate process to avoid Turing conflicts
    @testset "JuliaBUGS Extension (subprocess)" begin
        test_file = joinpath(@__DIR__, "juliabugscallback.jl")
        result = run(`$(Base.julia_cmd()) --project=$(dirname(@__DIR__)) --startup-file=no $test_file`)
        @test success(result)
    end
end
