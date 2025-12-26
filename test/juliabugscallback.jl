using Test
using JuliaBUGS
using AbstractMCMC
using AdvancedHMC
using Random
using TuringCallbacks
using TensorBoardLogger, ValueHistories
using ForwardDiff

@testset "JuliaBUGS Extension" begin
    tmpdir = mktempdir()
    mkpath(tmpdir)

    # Define a simple BUGS model
    model_def = @bugs("""
    model {
        for (i in 1:N) {
            y[i] ~ dnorm(mu, tau)
        }
        mu ~ dnorm(0, 0.001)
        tau ~ dgamma(0.001, 0.001)
        sigma <- 1 / sqrt(tau)
    }
    """, true, false)

    data = (N = 10, y = [1.0, 2.0, 3.0, 2.5, 2.8, 3.2, 2.9, 3.1, 2.6, 2.7])
    
    inits = (mu = 0.0, tau = 1.0)
    
    bugs_model = compile(model_def, data, inits)
    
    logdensity_model = AbstractMCMC.LogDensityModel(bugs_model)

    num_samples = 100

    sampler = NUTS(0.65)

    @testset "Correctness of values" begin
        callback = TensorBoardCallback(joinpath(tmpdir, "runs"))

        rng = Random.MersenneTwister(42)
        chain = AbstractMCMC.sample(rng, logdensity_model, sampler, num_samples; callback=callback)

        hist = convert(MVHistory, callback.logger)

        @test haskey(hist, Symbol("mu/val"))
        @test haskey(hist, Symbol("tau/val"))
        
        @test haskey(hist, Symbol("mu/stat/Mean"))
        @test haskey(hist, Symbol("tau/stat/Mean"))
    end

    @testset "Default logging" begin
        callback = TensorBoardCallback(joinpath(tmpdir, "runs"))

        rng = Random.MersenneTwister(42)
        chain = AbstractMCMC.sample(rng, logdensity_model, sampler, num_samples; callback=callback)

        hist = convert(MVHistory, callback.logger)

        @testset "$param" for param in ["mu", "tau"]
            @test haskey(hist, Symbol(param, "/val"))
            @test haskey(hist, Symbol(param, "/stat/Mean"))
            @test haskey(hist, Symbol(param, "/stat/Variance"))
        end

        @testset "extras" begin
            @test haskey(hist, Symbol("extras/lp/val"))
            @test haskey(hist, Symbol("extras/acceptance_rate/val"))
        end
    end

    @testset "Exclude parameter" begin
        callback = TensorBoardCallback(
            joinpath(tmpdir, "runs");
            exclude=["tau"]
        )

        rng = Random.MersenneTwister(42)
        chain = AbstractMCMC.sample(rng, logdensity_model, sampler, num_samples; callback=callback)

        hist = convert(MVHistory, callback.logger)

        @testset "$param" for param in ["mu", "tau"]
            if param == "tau"
                @test !haskey(hist, Symbol(param, "/val"))
                @test !haskey(hist, Symbol(param, "/stat/Mean"))
                @test !haskey(hist, Symbol(param, "/stat/Variance"))
            else
                @test haskey(hist, Symbol(param, "/val"))
                @test haskey(hist, Symbol(param, "/stat/Mean"))
                @test haskey(hist, Symbol(param, "/stat/Variance"))
            end
        end

        @testset "extras" begin
            @test haskey(hist, Symbol("extras/lp/val"))
            @test haskey(hist, Symbol("extras/acceptance_rate/val"))
        end
    end

    @testset "Exclude extras" begin
        callback = TensorBoardCallback(
            joinpath(tmpdir, "runs");
            include_extras=false
        )

        rng = Random.MersenneTwister(42)
        chain = AbstractMCMC.sample(rng, logdensity_model, sampler, num_samples; callback=callback)

        hist = convert(MVHistory, callback.logger)

        @testset "$param" for param in ["mu", "tau"]
            @test haskey(hist, Symbol(param, "/val"))
            @test haskey(hist, Symbol(param, "/stat/Mean"))
            @test haskey(hist, Symbol(param, "/stat/Variance"))
        end

        @testset "extras" begin
            @test !haskey(hist, Symbol("extras/lp/val"))
            @test !haskey(hist, Symbol("extras/acceptance_rate/val"))
        end
    end

    @testset "With hyperparams" begin
        callback = TensorBoardCallback(
            joinpath(tmpdir, "runs");
            include_hyperparams=true,
        )

        rng = Random.MersenneTwister(42)
        chain = AbstractMCMC.sample(rng, logdensity_model, sampler, num_samples; callback=callback)

        iter = TensorBoardLogger.TBEventFileCollectionIterator(
            callback.logger.logdir, purge=true
        )

        found_one = false
        for event_file in iter
            for event in event_file
                event.what === nothing && continue
                !(event.what.value isa TensorBoardLogger.Summary) && continue

                for (tag, _) in event.what.value
                    if tag == "_hparams_/experiment"
                        found_one = true
                        break
                    end
                end
            end

            found_one && break
        end
        @test found_one
    end
end
