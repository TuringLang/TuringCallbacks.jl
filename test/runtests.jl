using Turing,TuringCallbacks, ValueHistories
using Test

@testset "TuringCallbacks.jl" begin
    tmpdir = mktempdir()
    mkpath(tmpdir)

    @model function demo(x)
        s ~ InverseGamma(2, 3)
        m ~ Normal(0, √s)
        for i in eachindex(x)
            x[i] ~ Normal(m, √s)
        end
    end
    vns = [@varname(s), @varname(m)]

    xs = randn(100) .+ 1;
    model = demo(xs);

    # Number of MCMC samples/steps
    num_samples = 100
    num_adapts = 100
    
    # Sampling algorithm to use
    alg = NUTS(num_adapts, 0.65)

    @testset "Default" begin
        # Create the callback
        callback = TensorBoardCallback(
            joinpath(tmpdir, "runs");
        )

        # Sample
        chain = sample(model, alg, num_samples; callback = callback)

        # Read the logging info.
        hist = convert(MVHistory, callback.logger)

        # Check the variables.
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

    @testset "Exclude variable" begin
        # Create the callback
        callback = TensorBoardCallback(
            joinpath(tmpdir, "runs");
            exclude=["s"]
        )

        # Sample
        chain = sample(model, alg, num_samples; callback = callback)

        # Read the logging info.
        hist = convert(MVHistory, callback.logger)

        # Check the variables.
        @testset "$vn" for vn in vns
            if vn == @varname(s)
                @test !haskey(hist, Symbol(vn, "/val"))
                @test !haskey(hist, Symbol(vn, "/stat/Mean"))
                @test !haskey(hist, Symbol(vn, "/stat/Variance"))
            else
                @test haskey(hist, Symbol(vn, "/val"))
                @test haskey(hist, Symbol(vn, "/stat/Mean"))
                @test haskey(hist, Symbol(vn, "/stat/Variance"))
            end
        end

        # Check the extra statistics.
        @testset "extras" begin
            @test haskey(hist, Symbol("extras/lp/val"))
            @test haskey(hist, Symbol("extras/acceptance_rate/val"))
        end
    end

    @testset "Exclude extras" begin
        # Create the callback
        callback = TensorBoardCallback(
            joinpath(tmpdir, "runs");
            include_extras=false
        )

        # Sample
        chain = sample(model, alg, num_samples; callback=callback)

        # Read the logging info.
        hist = convert(MVHistory, callback.logger)

        # Check the variables.
        @testset "$vn" for vn in vns
            @test haskey(hist, Symbol(vn, "/val"))
            @test haskey(hist, Symbol(vn, "/stat/Mean"))
            @test haskey(hist, Symbol(vn, "/stat/Variance"))
        end

        # Check the extra statistics.
        @testset "extras" begin
            @test !haskey(hist, Symbol("extras/lp/val"))
            @test !haskey(hist, Symbol("extras/acceptance_rate/val"))
        end
    end
end
