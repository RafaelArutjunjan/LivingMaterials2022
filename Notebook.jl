### A Pluto.jl notebook ###
# v0.19.9

using Markdown
using InteractiveUtils

# This Pluto notebook uses @bind for interactivity. When running this notebook outside of Pluto, the following 'mock version' of @bind gives bound variables a default value (instead of an error).
macro bind(def, element)
    quote
        local iv = try Base.loaded_modules[Base.PkgId(Base.UUID("6e696c72-6542-2067-7265-42206c756150"), "AbstractPlutoDingetjes")].Bonds.initial_value catch; b -> missing; end
        local el = $(esc(element))
        global $(esc(def)) = Core.applicable(Base.get, el) ? Base.get(el) : iv(el)
        el
    end
end

# ╔═╡ 368a22f2-e727-11ec-0cd7-092914ead51b
using InformationGeometry, OrdinaryDiffEq, ModelingToolkit, Plots, PlutoUI, LinearAlgebra

# ╔═╡ fed6374f-c441-4855-b7dd-958e9f0f0f7f
using DiffEqCallbacks

# ╔═╡ ac2bd079-6aac-4c79-abf4-28417990e42d
md"""# Introduction to Dynamic Modelling in Julia

The aim of this notebook is to give a gentle introduction to implementing concrete modelling problems as a part of the workshop **Model-Driven Optimization of Biological Systems** at the conference on [Engineered Living Materials 2022 in Saarbrücken](https://www.livingmaterials2022.de/).

For more information, see [github.com/RafaelArutjunjan/LivingMaterials2022](https://www.github.com/RafaelArutjunjan/LivingMaterials2022). If you have further questions, feel free to message me at [rafael.arutjunjan@fdm.uni-freiburg.de](mailto:rafael.arutjunjan@fdm.uni-freiburg.de?subject=Re: LivingMaterials2022 - Model-Driven Optimization of Biological Systems).
"""

# ╔═╡ 7c48d58d-696c-4757-b0fc-2e13be224ebe
md"First, we need to load the libraries which we intend to use:"

# ╔═╡ 21d6bfa1-4150-4544-86e1-32cbaee190b2
md"""
As the names suggest, [**OrdinaryDiffEq.jl**](https://diffeq.sciml.ai/stable/tutorials/ode_example/) is used for numerically integrating ODEs while [**Plots.jl**](https://docs.juliaplots.org/latest/tutorial/) is required for visually displaying data.

While there are many ways to specify ODE systems in Julia, [**ModelingToolkit.jl**](https://mtk.sciml.ai/stable/) is particularly convenient, since it uses a symbolic syntax. For more details, see [**DifferentialEquations.jl**](https://diffeq.sciml.ai/stable/).

For fitting ODE systems to data, i.e. estimating its parameters such as reaction rates and initial concentrations, we will use the [**InformationGeometry.jl**](https://rafaelarutjunjan.github.io/InformationGeometry.jl/stable/) package.
"""

# ╔═╡ dce64479-39da-47a1-a54b-4474da01dbf9
md"""
## Baby Steps in Modelling
Let us consider a first-order reaction of the form $\dot{A} = -k \, A$, which describes a decay process. The solution to this ODE can be analytically computed as $A(t) = A_0 \, \exp(-k \, t)$.

Assuming that we experimentally measured the following data

| $t$      | $A$ | $\sigma$ |
| :-----------: | :-----------: | :-----------: |
| $0.2$ | $0.9$ | $0.1$ |
| $1$ | $0.4$ | $0.1$ |
| $2$ | $0.25$ | $0.1$ |

we can determine the initial concentration $A_0$ and the reaction rate $k$ from this data and subsequently use this model to make predictions for future measurements.
"""

# ╔═╡ ac8d160f-657b-4d8c-9f50-e1758cbf85bb
md"""
In the **InformationGeometry.jl** framework, datasets without missing values and without time-uncertainties can be constructed as:
"""

# ╔═╡ 328c20ea-1dc3-44f6-8a36-6dfb185232a7
DS = DataSet([0.2,1,2], [0.9,0.4,0.25], [0.1,0.1,0.1]; xnames=["t"], ynames=["A"])

# ╔═╡ 31706b80-5d7d-4dc2-b273-37ef10c11698
md"""
Next, a `DataModel` object is created using a dataset and a model function.
"""

# ╔═╡ a2dece9c-6e0c-4ce6-bf78-82176d5d352e
function DecayFunc(t, θ)
	A₀, k = θ
	return A₀ * exp(-k * t)
end

# ╔═╡ 392c1493-7e94-48bd-9a00-f3d08dae9d4e
DecayDM = DataModel(DS, DecayFunc);

# ╔═╡ 52727d42-b422-46e3-aabc-c979713b6dca
begin
	##### Custom code for creating data space visualizations
	using RecipesBase
	import SciMLBase: AbstractODESolution
	import InformationGeometry: meshgrid, ToCols, Unpack, DataspaceDim, GetConfnum

	Grid(Cube::HyperCube, N::Int=5) = [range(Cube.L[i], Cube.U[i]; length=N) for i in 1:length(Cube)]

	"""
	    VisualizeManifold(DM::AbstractDataModel, sols::AbstractODESolution, args...; kwargs...)
	    VisualizeManifold(DM::AbstractDataModel, sols::Vector{<:AbstractODESolution}; Padding::Real=0.1, kwargs...)
	    VisualizeManifold(DM::AbstractDataModel, sols::Vector{<:AbstractODESolution}, Cube::HyperCube; N::Int=50, kwargs...)
	    VisualizeManifold(DM::AbstractDataModel, sols::Union{<:Nothing,Vector{<:AbstractODESolution}}, points::AbstractVector{<:AbstractVector{<:Number}}; rescale::Bool=false, kwargs...)
	Visualizes a mapping of the parameter manifold into the data space for pdim=2 and dim(D) = 3.
	Providing confidence regions in the form of ODESolution objects also plots them in data space.
	The keyword `rescale=true` applies ``y \\longmapsto \\Sigma^{-1/2}(y - y_\\text{MLE})`` to all objects, thus showing the residual space.
	"""
	VisualizeManifold(DM::AbstractDataModel, Confnum::Real=1.; kwargs...) = VisualizeManifold(DM, nothing, LinearCuboid(DM, Confnum); kwargs...)
	VisualizeManifold(DM::AbstractDataModel, sols::AbstractODESolution, args...; kwargs...) = VisualizeManifold(DM, [sols], args...; kwargs...)
	VisualizeManifold(DM::AbstractDataModel, sols::Vector{<:AbstractODESolution}; Padding::Real=0.1, kwargs...) = VisualizeManifold(DM, sols, ConstructCube(sols; Padding=Padding); kwargs...)


	function VisualizeManifold(DM::AbstractDataModel, sols::Union{<:Nothing,Vector{<:AbstractODESolution}}, Cube::HyperCube; N::Int=50, kwargs...)
	    if pdim(DM) == 2
	        VisualizeManifold(DM, sols, collect(eachrow(reduce(hcat, meshgrid(Grid(Cube,N)...)))); kwargs...)
	    elseif pdim(DM) == 1
	        VisualizeManifold(DM, sols, [[x] for x in range(Cube.L[1], Cube.U[1]; length=N)]; kwargs...)
	    else
	        throw("Got pdim=$(pdim(DM)), can only handle pdim=1 or pdim=2.")
	    end
	end

	function VisualizeManifold(DM::AbstractDataModel, sols::Union{<:Nothing,Vector{<:AbstractODESolution}}, points::AbstractVector{<:AbstractVector{<:Number}}; rescale::Bool=false, OverWrite::Bool=true, kwargs...)
	    @assert DataspaceDim(DM) == 3
	    CoordTrafo = if rescale
	        C=cholesky(yInvCov(DM)).U
	        z::AbstractVector{<:Number}->C*(z-ydata(DM))
	    else
	        z::AbstractVector{<:Number}->z
	    end
	    Embedder(args...;) = Embedder([args...])
	    Embedder(x::AbstractVector{<:Number}) = try CoordTrafo(EmbeddingMap(DM, x)) catch; print("Dropped a point. "); nothing end
	    OverWrite && RecipesBase.plot()
	    p = if pdim(DM) == 1
	        RecipesBase.plot!(ToCols(Unpack(map(Embedder, points)))...; linestyle=:dash, markeralpha=0, linealpha=1, label="Model mfd.", markersize=0.8, color=:black, xlabel="y₁", ylabel="y₂", zlabel="y₃", kwargs...)
	    elseif pdim(DM) == 2
	        RecipesBase.plot!(map(Embedder, points); seriestype=:scatter, label="Model mfd.", markersize=0.9, color=:black, xlabel="y₁", ylabel="y₂", zlabel="y₃", kwargs...)
	    end
	    p = RecipesBase.plot!([CoordTrafo(ydata(DM))]; seriestype=:scatter, label="y_data", color=:red)
	    p = RecipesBase.plot!([Embedder(MLE(DM))]; seriestype=:scatter, label="MLE pred.", color=:blue)
	    if !isnothing(sols)
	        for sol in sols
	            p = RecipesBase.plot!(EmbeddedODESolution(Embedder, sol); vars=(1,2,3), label="$(round(GetConfnum(DM,sol);sigdigits=2)) sigma", linewidth=2, markeralpha=0, linealpha=1)
	        end
	    end;    p
	end
	function VisualizeManifoldParam(DM::AbstractDataModel, a::AbstractVector, Confnum::Real=1.0, Cube::HyperCube=LinearCuboid(DM, Confnum); kwargs...)
		p1 = VisualizeManifold(DM, Confnum)
		VisualizeManifoldParam(p1, DM, a, Confnum, Cube; kwargs...)
	end
	function VisualizeManifoldParam(p1::Plots.Plot, DM::AbstractDataModel, a::AbstractVector, Confnum::Real=1.0, Cube::HyperCube=LinearCuboid(DM, Confnum); kwargs...)
		BestPred = EmbeddingMap(DM, MLE(DM))
		Point = EmbeddingMap(DM, a)
		Q = Unpack([t*ydata(DM) + (1-t)*BestPred for t in 0:1])
		plot!(p1, view(Q,:,1), view(Q,:,2), view(Q,:,3); linewidth=2, markeralpha=0, line=:dash, label="")
		M = Unpack([t*ydata(DM) + (1-t)*Point for t in 0:1])
		plot!(p1, view(M,:,1), view(M,:,2), view(M,:,3); linewidth=2, linealpha=1, markeralpha=0, label="")
		plot!(p1, [Point]; marker=:hex, markercolor=:green, label="θ=$(round.(a; digits=2))", title="$(SymbolicModel(DM))")
		p1
	end
	function ManipulateParamVec(DM::AbstractDataModel, Confnum::Real, Cube::HyperCube, a::AbstractVector; kwargs...)
		p1 = VisualizeManifoldParam(DM, a, Confnum, Cube; kwargs...)
		p2 = plot(Data(DM));	PlotFit(DM; label="Best Fit", color=:blue, linewidth=1.5, line=:dash);	PlotFit(DM, a; label="Fit: θ=$(round.(a; digits=2))", color=:green, linewidth=3)
		plot(p1, p2; layout=(2,1), heights=[0.7,0.3], kwargs...)
	end
	#####

	# Cache first plot
	plotcache1 = VisualizeManifold(DecayDM, 2);

	md"(Btw, I am sneakily loading some code which is required for the visualizations in this cell.)"
end

# ╔═╡ 8fd86feb-4879-40fa-8668-747d6548d8b9
begin
	##### Need this for compatibility with Pluto notebooks, usually not required
	using Distributed
	@everywhere using ProgressMeter
	#####

	plot(ParameterProfile(DecayDM, 3.5; verbose=false, plot=false))
end

# ╔═╡ aa23ab43-cf3d-46b2-b0c3-fe59d2168fbb
plot(DecayDM)

# ╔═╡ c7ea5e84-aa72-471d-a241-34c882cfd401
md"""
During the construction of the `DataModel` object, the MLE is already computed and can be accessed via:
"""

# ╔═╡ c020d710-8819-4cfb-8d3d-eeddeb369c1d
MLE(DecayDM)

# ╔═╡ f8e3f1b5-d277-4dca-9e56-acb331d3b1aa
md"That is, the maximum likelihood estimate is given by ``A_0 \approx`` $(round(MLE(DecayDM)[1];sigdigits=4)) and ``k\approx`` $(round(MLE(DecayDM)[2];sigdigits=4))."

# ╔═╡ f0e660f4-defe-48c2-8b2c-1916df31bafc
md"""
## Geometric Interpretation of Parameter Inference
Given this simple example of a decay process with only 3 observations, we can visualize the data space $\mathcal{Y}^N$ in its entirety. In this data space, the collection of all observations $y_\text{data} := (y_1, y_2, y_3)$ constitutes a single point.

Similarly, the model predictions at the corresponding observation times $y_\text{pred} = \big( y_\text{model}(t_1, \theta), y_\text{model}(t_2, \theta), y_\text{model}(t_3, \theta) \big)$ for some parameter configuration $\theta=(A_0, k)$, which are to be compared to the measured data, also constitute a single point.

The likelihood $L(\theta)$ can then be interpreted as quantifying a measure of "distance" between the data $y_\text{data}$ and predictions $y_\text{pred}$, which is to be minimized by finding appropriate values for $\theta=(A_0, k)$.
"""

# ╔═╡ 68da6271-ce03-4ee0-a903-e9cfd32b86e6
begin
	DecayCube = LinearCuboid(DecayDM, 2)
	md"A₀ = $(@bind P1 Slider(range(DecayCube[1]...; length=50); default=1.))	k = $(@bind P2 Slider(range(DecayCube[2]...; length=50); default=0.6))"
end

# ╔═╡ c47b633c-6a74-4f68-9442-34db95d876ce
# VisualizeManifoldParam(DecayDM, [P1,P2], 2, DecayCube)
VisualizeManifoldParam(deepcopy(plotcache1), DecayDM, [P1,P2], 2, DecayCube)

# ╔═╡ 705edf55-7664-473a-91a8-725ee479b5b4
begin
	plot(Data(DecayDM))
	PlotFit(DecayDM; label="Best Fit", color=:blue, linewidth=1.5, line=:dash);	PlotFit(DecayDM, [P1,P2]; label="Fit: θ=$(round.([P1,P2]; digits=2))", color=:green, linewidth=3)
end

# ╔═╡ a8696ecb-62c0-4d67-8244-5371a2d19315
md"""
The surface indicated by the black dots constitutes an embedding of the "model manifold" $\mathcal{M}$ into the data space, i.e. the set of all possible model predictions for the observed time points in the dataset.

The non-linear dependence of a model on its parameters can be seen here from the fact that the embedded model manifold does not constitute a flat plane, but instead exhibits a curved shape.

It is unsurprising that the model manifold is a 2D surface in this example, since our model has 2 parameters $A_0$ and $k$. Although we can no longer visualize the data space if we have more than 3 data points, the underlying idea and mathematical formalism remain the same.
"""

# ╔═╡ 5f5127ab-81ca-4ac0-8162-c14b93aa2abb
md"""
If the model only had a single free parameter, e.g. if we already knew from some other experiment that $A_0 = 1$, the parameter manifold would be one-dimensional.
"""

# ╔═╡ 8705b9d3-7fc1-4847-96fa-4314b4cff4af
DecayDMsmall = DataModel(DS, (t,p)->DecayFunc(t, [1.0, p[1]]));

# ╔═╡ 11e0f82d-cb9a-4b15-ad45-489981471303
begin
	DecayCubeSmall = LinearCuboid(DecayDMsmall, 2)
	md"k = $(@bind S1 Slider(range(DecayCubeSmall[1]...; length=100); default=0.7))"
end

# ╔═╡ 77155256-b6af-4dfc-9f71-994f99319260
VisualizeManifoldParam(DecayDMsmall, [S1], 2)

# ╔═╡ b7f5585a-2078-4bb0-8a9f-44f441325599
begin
	plot(Data(DecayDMsmall))
	PlotFit(DecayDMsmall; label="Best Fit", color=:blue, linewidth=1.5, line=:dash)
	PlotFit(DecayDMsmall, [S1]; label="Fit: θ=$(round.([S1]; digits=2))", color=:green, linewidth=3)
end

# ╔═╡ f5ad0a25-28b2-4905-b691-f3d91230b39d
md"""
## Quantifying Parameter Uncertainty
As discussed, there are many inequivalent ways in which parameter uncertainties are specified and communicated in the literature.

The most commonly used and incidentally also worst (i.e. most misleading) way to specify parameter uncertainties is to take them as the square root of the diagonal of the inverse Fisher information matrix. This not only assumes that the uncertainty in the parameters is symmetric but also that the value of any given parameter is completely independent of all the other parameters.
"""

# ╔═╡ cec1b102-7df1-4d13-8fd4-ddae5a0c8ff6
InformationGeometry.MLEuncert(DecayDM)

# ╔═╡ 75f9228f-f6f3-4e85-8e79-936e2c66d71d
md"""
A slightly more nuanced view is to acknowledge the interactive and compensatory effects between parameters (i.e. their covariance) via the off-diagonal elements of the inverse Fisher information:
"""

# ╔═╡ 73cfdd08-aa15-4078-b752-02a8cf95ae48
inv(FisherMetric(DecayDM, MLE(DecayDM)))

# ╔═╡ 2176e5d4-7925-4ee3-9e87-ce829fdff052
md"""
From this we can see that covariance between the parameters $A_0$ and $k$ not only exists, but that the off-diagonal term is similar in magnitude to the diagonal terms!

For models which are linear with respect to all of their parameters, the confidence regions constitute perfect ellipsoids around the MLE. Therefore, their shape is fully encoded in the (inverse) Fisher information matrix and no further tools are required.

However, with the exception of a few simplistic examples, **all ODE-based models depend non-linearly on their parameters** such as reaction rates and initial concentrations. As a result, they are guaranteed to exhibit non-linear parameter space distortions and more elaborate analyses of the parameter uncertainty are required.
"""

# ╔═╡ 227e52af-cbb5-4b9c-bd67-3138c517c84a
md"""
The following illustrates a rescaled version of the profile likelihood in terms of confidence level. For a given confidence level, say $2σ \,\hat{=}\, 95\%$, all parameter values for which this line is below the horizontal threshold of $2σ \,\hat{=}\, 95\%$ are compatible with the data up to the given confidence level (if the remaining parameters are suitably chosen).
"""

# ╔═╡ 74af9876-f57a-4f72-9ecc-0de1f7229ee3
md"""
Although the distortion of the parameter space is not incredibly pronounced in this example, there is a visible asymmetry particularly in the profile of the parameter $k$, which stems from the fact that it appears inside an exponential function in the model.

Overall, the profile likelihood is a robust method which faithfully captures the asymmetries in the parameter uncertainties that are induced by non-linearity. For this reason, profile likelihood analyses are considered by many to be the gold standard when it comes to investigating parameter uncertainty and are somewhat of a speciality at the Jetilab Freiburg.
"""

# ╔═╡ 2218cb5c-1050-41ca-af92-f572c4e378ae
md"""
Lastly, for models which are sufficiently "nice" in that they are structurally identifiable and have a managable number of parameters $\lesssim 10$, even more in-depth investigations of the full confidence region become feasible.
"""

# ╔═╡ 4caabc84-5810-476e-88c1-0f61d19396e5
begin
	Decaysols = ConfidenceRegions(DecayDM, 1:3)
	VisualizeSols(DecayDM, Decaysols; xlabel="A₀", ylabel="k", title="Confidence Regions 1σ-3σ")
end

# ╔═╡ aa54a793-7fd1-46be-8d35-d8ea2af60abb
md"""
If confidence regions can be constructed explicitly in this manner, confidence bands around the predictions of a given confdence level can immediately be obtained by evaluating the model on configurations belonging to the associated confidence region.
"""

# ╔═╡ 0bdd5e46-840d-4900-adc2-276ef08c4554
begin
	plot(DecayDM)
	M = ConfidenceBands(DecayDM, Decaysols[2]; N=500, plot=false)
	plot!(view(M,:,1), view(M,:,2), color=:blue, label="2σ Conf Bands")
	plot!(view(M,:,1), view(M,:,3), color=:blue, label="")
end

# ╔═╡ c8fe19ad-a90c-468c-be47-7cf0b78c98e6
md"""
At every time $t$ there is a $95\%$ probability that the prediction corresponding to the true parameters $\theta_\text{true}$ lies between the $95\%$ confidence bands (in the frequentist sense). The non-trivial shape of the obtained confidence bands nicely illustrates the benefit of this precise approach since it accounts for the varying flexibility in the model across different times $t$.

Consequently, the width of the confidence bands can also be used in experimental design since recording more data in time ranges where the bands are widest will be most informative for further constraining the parameter values of the model.
"""

# ╔═╡ 94f42b08-f105-4f92-804c-fcefca4df0fb
md"""
## Constructing ODE models

We are finally ready to graduate to a real-world example of modelling in synthetic biology.
"""

# ╔═╡ 5e019ff9-b355-4c5c-93b9-73e37dd0e9e2
md"""
### Repressible Promoter Model

This model is adapted from [Kramer, B., Weber, W. et al. An engineered epigenetic transgene switch in mammalian cells. Nature Biotechnol 22, 867–870 (2004).](https://doi.org/10.1038/nbt980)

For simplicity, the concentrations of the expression vector $v$, activator $a$ and repressor $r$ can be assumed to stay nearly constant in equilibrium.
"""

# ╔═╡ c110adfd-5f0a-4a2b-a104-829a0a218b13
begin
	@parameters t k₀ k₁ k₂ k₃ k₄ n
	@variables mRNA(t) v(t) a(t) r(t)
	Dt = Differential(t)

	eqs = [Dt(mRNA) ~ (v*k₀ + v*k₁*a^n / (k₂^n + a^n)) / (1 + (k₃ * r)^n) - k₄ * mRNA,
		   Dt(v) ~ 0,  Dt(a) ~ 0,  Dt(r) ~ 0]
	@named PromoterSystem = ODESystem(eqs, t, [mRNA, v, a, r], [k₀, k₁, k₂, k₃, k₄, n])
end

# ╔═╡ a280b180-ed70-41ee-ac63-ed3c2391c5a2
md"""
The above code illustrates the specification of an ODE system.
First, symbolic variables are created for the ODE parameters using the `@parameters` macro and for the time-dependent states of the ODE via the `@variables` macro. Next, a vector of equations for the various states is created, where the tilde symbol `~` is used to represent equals signs.

Finally the symbolic equations, time variable, state vector and parameter vector are passed to the `ODESystem` constructor.
"""

# ╔═╡ fa852d24-268c-4377-be33-774914be189c
md"""
Even without any data, the time evolution of the ODE system can be simulated for given initial concentrations and parameters as follows:
"""

# ╔═╡ 84bf2ba8-4889-4a24-9556-6f25b890cec7
begin
	InitialStates = [0, 1.0, 1.5, 0.0]
	timespan = (0., 10.)
	InitialParameters = [1, 20, 2, 2, 2, 2.]

	prob = ODEProblem(PromoterSystem, InitialStates, timespan, InitialParameters)
end

# ╔═╡ 1feba59c-b55c-4f08-9a9f-550653ffc23c
md"Note that the order of the states and parameters corresponds to the order in which they were specified in the `ODESystem` object."

# ╔═╡ df864dcf-9885-48d5-bcd8-9b658a8b650f
sol = solve(prob, Tsit5(); reltol=1e-4, abstol=1e-4);

# ╔═╡ cd472569-6b92-47d6-92a8-faede05b560d
md"""
The `solve` command is used to compute concrete numerical solutions to the ODE problem up to a specified tolerance, i.e. accuracy. `Tsit5()` specifies the use of the so-called Tsitouras 5th order Runge--Kutta method. More information on ODE solvers can be found [here](https://diffeq.sciml.ai/stable/solvers/ode_solve/).
"""

# ╔═╡ cf6406e5-d0fb-4c3f-be1a-a6b4962fa096
plot(sol; vars=(0,1), ylabel="mRNA(t)", label="a(t=0) = 1.5")

# ╔═╡ 09efd4a7-8e8f-469f-b57c-dc5c23b1d7dc
begin
	### Cache concentration plot here
	Sol(a₀::Real) = solve(ODEProblem(PromoterSystem, [0,1,a₀,0.0], timespan, InitialParameters), Tsit5(); reltol=1e-4, abstol=1e-4)

	plotcache3 = plot(Sol(1); vars=(0,1), label="a(t=0) = $(1)");
	plot!(plotcache3, Sol(3); vars=(0,1), label="a(t=0) = $(3)");
	plot!(plotcache3, Sol(5); vars=(0,1), label="a(t=0) = $(5)");
	plot!(plotcache3, Sol(8); vars=(0,1), label="a(t=0) = $(8)");
	###


	md"""
	The keyword argument `vars=(0,1)` indicates that we wish to plot the first state of the ODE (i.e. the state which describes the mRNA concentration) against the "zeroth" state, which is time.
	"""
end

# ╔═╡ 4703f4a0-887a-40bc-8593-a805f6a04e9a
md"""
We can get a more intuitive feeling for the dependence of the `mRNA` on the initial concentration `a₀` by changing its value:
$(@bind a₀ Slider(0.0:0.1:10; default=2))
"""

# ╔═╡ 5a932d80-4ddb-4fd3-9d33-c7d9fe00102e
## Cached earlier
# Sol(a₀::Real) = solve(ODEProblem(PromoterSystem, [0,1,a₀,0.0], timespan, InitialParameters), Tsit5(); reltol=1e-4, abstol=1e-4)

# plot(Sol(1); vars=(0,1), label="a(t=0) = $(1)");
# plot!(Sol(3); vars=(0,1), label="a(t=0) = $(3)");
# plot!(Sol(5); vars=(0,1), label="a(t=0) = $(5)");
# plot!(Sol(8); vars=(0,1), label="a(t=0) = $(8)");
# plot!(Sol(a₀); linewidth=3, vars=(0,1), ylabel="mRNA(t)", label="a(t=0) = $(a₀)")

plot!(deepcopy(plotcache3), Sol(a₀); linewidth=3, vars=(0,1), ylabel="mRNA(t)", label="a(t=0) = $(a₀)")

# ╔═╡ 3ed68717-66ce-4378-b650-72b856309d57
md"""
### Light-Induced Gene Expression Model
This model is based on the *Arabidopsis thaliana* red / far-red light-sensing systems discussed in [Müller, K., Weber, W. et al. A red/far-red light-responsive bi-stable toggle switch to control gene expression in mammalian cells, Nucleic Acids Research, Volume 41, Issue 7, (2013)](https://doi.org/10.1093/nar/gkt002).
"""

# ╔═╡ b11d2e7d-b5f1-494e-947b-8e26d3d48d08
begin
	# Symbols for t and mRNA already defined earlier
	@parameters k_deact k_act k_red_basal
	@parameters k_mRNA_basal k_mRNA_max k_mRNA_M k_mRNA_deg
	@parameters k_Prot k_Prot_deg
	@variables PVP_act(t) PVP_inact(t) Prot(t) I740(t) I660(t)

	Eqs = [ (Dt(PVP_act) ~ -k_deact*I740*PVP_act + k_act*I660*PVP_inact),
			(Dt(PVP_inact) ~ k_deact*I740*PVP_act -k_act*I660*PVP_inact),
			(Dt(mRNA) ~ k_mRNA_basal + k_mRNA_max * PVP_act / (k_mRNA_M + PVP_act) - k_mRNA_deg*mRNA),
			(Dt(Prot) ~ k_Prot * mRNA - k_Prot_deg*Prot),
			(Dt(I740) ~ 0), (Dt(I660) ~ 0)
	]
	@named LightControlSys = ODESystem(Eqs, t, [PVP_act, PVP_inact, mRNA, Prot, I740, I660], [k_deact, k_act, k_red_basal, k_mRNA_basal, k_mRNA_max, k_mRNA_M, k_mRNA_deg, k_Prot, k_Prot_deg])
end

# ╔═╡ 3f727356-479b-476d-90c5-701ed3faf985
md"""
For larger reaction systems in particular, the [**Catalyst.jl**](https://github.com/SciML/Catalyst.jl) package offers convenient macros which convert from a special syntax for chemical reactions into the appropriate system of ODE or SDE equations.
"""

# ╔═╡ b358465a-3f38-4d9e-9bf4-4835a759adfa
InitialDict = Dict(
	PVP_inact=>0,
	PVP_act=>15,
	mRNA=>10^(-0.625) / 10^(-0.551), # k_mRNA_basal/k_mRNA_deg,
	Prot=>10^(1.029),
	I740 => 1,
	I660 => 0,
)

# ╔═╡ 1a19e249-31dc-4e7f-a3cc-1387836ed746
ParamDict = Dict(
	k_Prot => 10^0.866,
	k_Prot_deg => 10^(-1.035),

	k_mRNA_basal => 10^(-0.625),
	k_mRNA_deg => 10^(-0.551),
	k_mRNA_max => 3, # k_tc*k_light, k_light = 0 oder 1 zwischen 1-7h
	k_mRNA_M => 10^(-0.196),

	k_deact => 10^(-0.756), # 10^(-0.158)
	k_act => 5., # 3600*2.8e-2,
)

# ╔═╡ 5ac31ba6-af9b-4666-b02e-2be74dcb7b29
begin
	NumToFloat(X::AbstractVector{<:Num}) = parse.(Float64,string.(X))

	# Construct vectors with values for the parameters and initial conditions
	U0 = substitute([PVP_act, PVP_inact, mRNA, Prot, I740, I660], InitialDict) |> NumToFloat

	P0 = substitute([k_deact, k_act, k_mRNA_basal, k_mRNA_max, k_mRNA_M, k_mRNA_deg, k_Prot, k_Prot_deg], ParamDict) |> NumToFloat
end

# ╔═╡ 33ce89b8-1328-45a0-b939-360bfaee060c
plot(
	solve(ODEProblem(ODEFunction(LightControlSys), U0, (0,10.), P0), Tsit5(); reltol=1e-5, abstol=1e-5)
)

# ╔═╡ 419f152d-866e-49e1-b0dd-66f1b2780496
md"""
We can also add special events to our ODE system, such as the on-switching or off-switching of light through the $I740$ and $I660$ states via the `PresetTimeCallback` function from the [**DiffEqCallbacks.jl**](https://github.com/SciML/DiffEqCallbacks.jl) library. For further details and examples, [see here](https://diffeq.sciml.ai/stable/features/callback_library/#PresetTimeCallback).
"""

# ╔═╡ f3f78d8b-c28f-4a72-a3b0-c1f1806a5d94
plot(
	solve(ODEProblem(ODEFunction(LightControlSys), U0, (0,10.), P0), Tsit5();
		reltol=1e-5, abstol=1e-5,
		callback=CallbackSet(
			# Switch I740 on at t=1h
			PresetTimeCallback([1.], integrator->integrator.u[5] += 1),
			# Switch off at t=7h
			PresetTimeCallback([7.], integrator->integrator.u[5] -= 1),
		)
	)
)

# ╔═╡ caff9d73-7675-4686-b20c-1f8f511baec3
md"""
## Tuning Biological Systems with the Help of ODE Models

Once the reaction rates governing a biological system have been sufficiently determined from previous experiments, the associated mathematical model can be used to make predictions for the outcomes of future experiments.

Typically, only the initial concentrations are experimentally tunable whereas the reaction rates are chemically fixed. Therefore, the model can be used to determine which particular initial concentrations are required to achieve a desired behaviour in the given biological system.
"""

# ╔═╡ 28f641ca-e392-4e2d-ba30-189db1cc435d
md"""
The [**InformationGeometry.jl**](https://github.com/RafaelArutjunjan/InformationGeometry.jl) package uses the following syntax to convert the ODE system into a model:
"""

# ╔═╡ 011dc665-d3db-4c4f-b863-e323e78cd547
PromoterModel = GetModel(PromoterSystem, θ->([0, 1.0, θ[1], 0.0], InitialParameters), 1)

# ╔═╡ 3c59c269-0be4-4aa0-b7be-18fbb7649cfe
md"""
The second argument is a function which splits the parameter vector into a vector of initial conditions for the ODE system and remaining parameters specific to the ODE system (i.e. reaction rates). Since the initial conditions are therefore also parameters of the model, this makes it possible to estimate them from data.

The third argument indicates which states of the ODE system are observed in the dataset. Instead of (a subset of) the ODE system states, it is also possible to specify an observation function which depends on the states. For more examples of this, see the [**InformationGeometry.jl** documentation](https://rafaelarutjunjan.github.io/InformationGeometry.jl/stable/ODEmodels/)
"""

# ╔═╡ 5b4052ce-17c0-4bdb-b7c9-97f03b056ed2
PromoterDM = DataModel(DataSet([0.,6], [0, 3.6], 0.1), PromoterModel)

# ╔═╡ 5a0c26c5-eb62-4941-9fd5-bd1ca5f5ea60
plot(PromoterDM)

# ╔═╡ f7e46c68-1be5-4562-ad74-a1169bf78087
md"""
The MLE for the initial concentration of the activator is found as ``a_0 \approx~``$(round(MLE(PromoterDM)[1]; sigdigits=3)).
"""

# ╔═╡ 00000000-0000-0000-0000-000000000001
PLUTO_PROJECT_TOML_CONTENTS = """
[deps]
DiffEqCallbacks = "459566f4-90b8-5000-8ac3-15dfb0a30def"
Distributed = "8ba89e20-285c-5b6f-9357-94700520ee1b"
InformationGeometry = "ee9d1287-bbdf-432c-9920-c447cf97a828"
LinearAlgebra = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
ModelingToolkit = "961ee093-0014-501f-94e3-6117800e7a78"
OrdinaryDiffEq = "1dea7af3-3e70-54e6-95c3-0bf5283fa5ed"
Plots = "91a5bcdd-55d7-5caf-9e0b-520d859cae80"
PlutoUI = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
ProgressMeter = "92933f4c-e287-5a05-a399-4b506db050ca"
RecipesBase = "3cdcf5f2-1ef4-517c-9805-6587b60abb01"
SciMLBase = "0bca4576-84f4-4d90-8ffe-ffa030f20462"

[compat]
DiffEqCallbacks = "~2.23.1"
InformationGeometry = "~1.12.3"
ModelingToolkit = "~8.14.0"
OrdinaryDiffEq = "~6.15.0"
Plots = "~1.30.0"
PlutoUI = "~0.7.39"
ProgressMeter = "~1.7.2"
RecipesBase = "~1.2.1"
SciMLBase = "~1.40.1"
"""

# ╔═╡ 00000000-0000-0000-0000-000000000002
PLUTO_MANIFEST_TOML_CONTENTS = """
"""

# ╔═╡ Cell order:
# ╟─ac2bd079-6aac-4c79-abf4-28417990e42d
# ╟─7c48d58d-696c-4757-b0fc-2e13be224ebe
# ╠═368a22f2-e727-11ec-0cd7-092914ead51b
# ╟─21d6bfa1-4150-4544-86e1-32cbaee190b2
# ╟─dce64479-39da-47a1-a54b-4474da01dbf9
# ╟─ac8d160f-657b-4d8c-9f50-e1758cbf85bb
# ╠═328c20ea-1dc3-44f6-8a36-6dfb185232a7
# ╟─31706b80-5d7d-4dc2-b273-37ef10c11698
# ╠═a2dece9c-6e0c-4ce6-bf78-82176d5d352e
# ╠═392c1493-7e94-48bd-9a00-f3d08dae9d4e
# ╠═aa23ab43-cf3d-46b2-b0c3-fe59d2168fbb
# ╟─c7ea5e84-aa72-471d-a241-34c882cfd401
# ╠═c020d710-8819-4cfb-8d3d-eeddeb369c1d
# ╟─f8e3f1b5-d277-4dca-9e56-acb331d3b1aa
# ╟─f0e660f4-defe-48c2-8b2c-1916df31bafc
# ╟─52727d42-b422-46e3-aabc-c979713b6dca
# ╟─68da6271-ce03-4ee0-a903-e9cfd32b86e6
# ╟─c47b633c-6a74-4f68-9442-34db95d876ce
# ╟─705edf55-7664-473a-91a8-725ee479b5b4
# ╟─a8696ecb-62c0-4d67-8244-5371a2d19315
# ╟─5f5127ab-81ca-4ac0-8162-c14b93aa2abb
# ╠═8705b9d3-7fc1-4847-96fa-4314b4cff4af
# ╟─11e0f82d-cb9a-4b15-ad45-489981471303
# ╟─77155256-b6af-4dfc-9f71-994f99319260
# ╟─b7f5585a-2078-4bb0-8a9f-44f441325599
# ╟─f5ad0a25-28b2-4905-b691-f3d91230b39d
# ╠═cec1b102-7df1-4d13-8fd4-ddae5a0c8ff6
# ╟─75f9228f-f6f3-4e85-8e79-936e2c66d71d
# ╠═73cfdd08-aa15-4078-b752-02a8cf95ae48
# ╟─2176e5d4-7925-4ee3-9e87-ce829fdff052
# ╟─227e52af-cbb5-4b9c-bd67-3138c517c84a
# ╠═8fd86feb-4879-40fa-8668-747d6548d8b9
# ╟─74af9876-f57a-4f72-9ecc-0de1f7229ee3
# ╟─2218cb5c-1050-41ca-af92-f572c4e378ae
# ╠═4caabc84-5810-476e-88c1-0f61d19396e5
# ╟─aa54a793-7fd1-46be-8d35-d8ea2af60abb
# ╟─0bdd5e46-840d-4900-adc2-276ef08c4554
# ╟─c8fe19ad-a90c-468c-be47-7cf0b78c98e6
# ╟─94f42b08-f105-4f92-804c-fcefca4df0fb
# ╟─5e019ff9-b355-4c5c-93b9-73e37dd0e9e2
# ╠═c110adfd-5f0a-4a2b-a104-829a0a218b13
# ╟─a280b180-ed70-41ee-ac63-ed3c2391c5a2
# ╟─fa852d24-268c-4377-be33-774914be189c
# ╠═84bf2ba8-4889-4a24-9556-6f25b890cec7
# ╟─1feba59c-b55c-4f08-9a9f-550653ffc23c
# ╠═df864dcf-9885-48d5-bcd8-9b658a8b650f
# ╟─cd472569-6b92-47d6-92a8-faede05b560d
# ╠═cf6406e5-d0fb-4c3f-be1a-a6b4962fa096
# ╟─09efd4a7-8e8f-469f-b57c-dc5c23b1d7dc
# ╟─4703f4a0-887a-40bc-8593-a805f6a04e9a
# ╟─5a932d80-4ddb-4fd3-9d33-c7d9fe00102e
# ╟─3ed68717-66ce-4378-b650-72b856309d57
# ╠═b11d2e7d-b5f1-494e-947b-8e26d3d48d08
# ╟─3f727356-479b-476d-90c5-701ed3faf985
# ╠═b358465a-3f38-4d9e-9bf4-4835a759adfa
# ╟─1a19e249-31dc-4e7f-a3cc-1387836ed746
# ╠═5ac31ba6-af9b-4666-b02e-2be74dcb7b29
# ╠═33ce89b8-1328-45a0-b939-360bfaee060c
# ╟─419f152d-866e-49e1-b0dd-66f1b2780496
# ╠═fed6374f-c441-4855-b7dd-958e9f0f0f7f
# ╠═f3f78d8b-c28f-4a72-a3b0-c1f1806a5d94
# ╟─caff9d73-7675-4686-b20c-1f8f511baec3
# ╟─28f641ca-e392-4e2d-ba30-189db1cc435d
# ╠═011dc665-d3db-4c4f-b863-e323e78cd547
# ╟─3c59c269-0be4-4aa0-b7be-18fbb7649cfe
# ╠═5b4052ce-17c0-4bdb-b7c9-97f03b056ed2
# ╟─5a0c26c5-eb62-4941-9fd5-bd1ca5f5ea60
# ╟─f7e46c68-1be5-4562-ad74-a1169bf78087
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
