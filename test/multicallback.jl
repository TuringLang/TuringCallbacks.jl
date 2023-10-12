@testset "MultiCallback" begin
    # Number of MCMC samples/steps
    num_samples = 100
    num_adapts = 50

    # Sampling algorithm to use
    alg = NUTS(num_adapts, 0.65)

    callback = MultiCallback(CountingCallback(), CountingCallback())
    chain = sample(demo_model, alg, num_samples, callback=callback)

    # Both should have been trigger an equal number of times.
    counts = map(c -> c.count[], callback.callbacks)
    @test counts[1] == counts[2]
    @test counts[1] == num_samples

    # Add a new one and make sure it's not like the others.
    callback = TuringCallbacks.push!!(callback, CountingCallback())
    counts = map(c -> c.count[], callback.callbacks)
    @test counts[1] == counts[2] != counts[3]
end
