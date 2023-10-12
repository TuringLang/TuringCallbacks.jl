@testset "SaveCallback" begin
    # Sample
    sample(model, alg, num_samples; callback = SaveCSV, chain_name="chain_1")
    chain = Matrix(CSV.read("chain_1.csv", DataFrame,  header=false))
    @test size(chain) == (num_samples, 2)
    rm("chain_1.csv")
end