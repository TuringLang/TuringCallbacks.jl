@testset "SaveCallback" begin
    # Number of MCMC samples/steps
    num_samples = 100
    num_adapts = 50

    # Sampling algorithm to use
    alg = NUTS(num_adapts, 0.65)

    sample(demo_model, alg, num_samples; callback = SaveCSV, chain_name="chain_1")
    chain = Matrix(CSV.read("chain_1.csv", DataFrame,  header=false))
    println(chain)
    @test size(chain) == (num_samples, 2)
    rm("chain_1.csv")
end