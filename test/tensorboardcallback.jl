@testset "TensorBoardCallback" begin
    tmpdir = mktempdir()
    mkpath(tmpdir)

    vns = DynamicPPL.TestUtils.varnames(demo_model)

    # Number of MCMC samples/steps
    num_samples = 100
    num_adapts = 50

    # Sampling algorithm to use
    alg = NUTS(num_adapts, 0.65)

    @testset "Correctness of values" begin
        # Create the callback
        callback = TensorBoardCallback(joinpath(tmpdir, "runs"))

        # Sample
        chain = sample(demo_model, alg, num_samples; callback=callback)

        # Extract the values.
        hist = convert(MVHistory, callback.logger)

        # Compare the recorded values to the chain.
        m_mean = last(last(hist["m/stat/Mean"]))
        s_mean = last(last(hist["s/stat/Mean"]))

        @test m_mean ≈ mean(chain[:m])
        @test s_mean ≈ mean(chain[:s])
    end

    @testset "Default" begin
        # Create the callback
        callback = TensorBoardCallback(
            joinpath(tmpdir, "runs");
        )

        # Sample
        chain = sample(demo_model, alg, num_samples; callback=callback)

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
        chain = sample(demo_model, alg, num_samples; callback=callback)

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
        chain = sample(demo_model, alg, num_samples; callback=callback)

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

    @testset "With hyperparams" begin
        @testset "$alg" for alg in [
            HMC(0.05, 10),
            HMCDA(num_adapts, 0.65, 1.0),
            NUTS(num_adapts, 0.65)
        ]

            # Create the callback
            callback = TensorBoardCallback(
                joinpath(tmpdir, "runs");
                include_hyperparams=true,
            )

            # Sample
            chain = sample(demo_model, alg, num_samples; callback=callback)

            # HACK: This touches internals so might just break at some point.
            # If it some point does, let's just remove this test.
            # Inspiration: https://github.com/JuliaLogging/TensorBoardLogger.jl/blob/3d9c1a554a08179785459ad7b83bce0177b90275/src/Deserialization/deserialization.jl#L244-L258
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
end
