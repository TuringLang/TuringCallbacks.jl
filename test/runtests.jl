using Turing,TuringCallbacks, ValueHistories
using Test

@testset "TuringCallbacks.jl" begin
    @model function demo(x)
        s ~ InverseGamma(2, 3)
        m ~ Normal(0, √s)
        for i in eachindex(x)
            x[i] ~ Normal(m, √s)
        end
    end

    xs = randn(100) .+ 1;
    model = demo(xs);

    # Number of MCMC samples/steps
    num_samples = 1_000

    # Sampling algorithm to use
    alg = NUTS()

    # Create the callback
    tmpdir = mktempdir()
    callback = TensorBoardCallback(tmpdir)

    # Sample
    chain = sample(model, alg, num_samples; callback = callback)

    # Read the logging info.
    hist = convert(MVHistory, callback.logger)

    # Check the variables.
    vns = [@varname(s), @varname(m)]
    @testset "$vn" for vn in vns
        # Should have the `val` field.
        @test haskey(hist, Symbol(vn, "/val"))
        # Should have the `Mean` and `Variance` stat.
        @test haskey(hist, Symbol(vn, "/stat/Mean"))
        @test haskey(hist, Symbol(vn, "/stat/Variance"))
    end

    # Check the extra statistics.
    @testset "extras" begin
        @test haskey(hist, Symbol("extras/lp/val"))
        @test haskey(hist, Symbol("extras/acceptance_rate/val"))
    end
end
