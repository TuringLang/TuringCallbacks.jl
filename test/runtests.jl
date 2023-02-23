using Test
using Turing
using TuringCallbacks
using TensorBoardLogger, ValueHistories

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
