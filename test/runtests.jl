using Test
using Turing
using TuringCallbacks
using TensorBoardLogger, ValueHistories

Base.@kwdef struct CountingCallback
    count::Ref{Int}=Ref(0)
end

(c::CountingCallback)(args...; kwargs...) = c.count[] += 1

@testset "TuringCallbacks.jl" begin
    # TODO: Improve.
    @model function demo(x)
        s ~ InverseGamma(2, 3)
        m ~ Normal(0, √s)
        for i in eachindex(x)
            x[i] ~ Normal(m, √s)
        end
    end

    xs = randn(100) .+ 1
    model = demo(xs)

    # Number of MCMC samples/steps
    num_samples = 1_000
    num_adapts = 500

    # Sampling algorithm to use
    alg = NUTS(num_adapts, 0.65)

    @testset "MultiCallback" begin
        callback = MultiCallback(CountingCallback(), CountingCallback())
        chain = sample(model, alg, num_samples, callback=callback)

        # Both should have been trigger an equal number of times.
        counts = map(c -> c.count[], callback.callbacks)
        @test counts[1] == counts[2]
        @test counts[1] == num_samples

        # Add a new one and make sure it's not like the others.
        callback = TuringCallbacks.push!!(callback, CountingCallback())
        counts = map(c -> c.count[], callback.callbacks)
        @test counts[1] == counts[2] != counts[3]
    end

    @testset "TensorBoardCallback" begin
        # Create the callback
        callback = TensorBoardCallback(mktempdir())

        # Sample
        chain = sample(model, alg, num_samples; callback=callback)

        # Extract the values.
        hist = convert(MVHistory, callback.logger)

        # Compare the recorded values to the chain.
        m_mean = last(last(hist["m/stat/Mean"]))
        s_mean = last(last(hist["s/stat/Mean"]))

        @test m_mean ≈ mean(chain[:m])
        @test s_mean ≈ mean(chain[:s])
    end
end
