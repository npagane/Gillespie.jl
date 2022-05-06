"A type storing the status at the end of a call to `ssa`:

- **termination_status** : whether the simulation stops at the final time (`finaltime`) or early due to zero propensity function (`zeroprop`)
- **nsteps** : the number of steps taken during the simulation.

"
struct SSAStats
    termination_status::String
    nsteps::Int64
end

"A type storing the call to `ssa`:

- **x0** : a `Vector` of `Int64`, representing the initial states of the system.
- **F** : a `Function` or a callable type, which itself takes two arguments; x, a `Vector` of `Int64` representing the states, and parms, a `Vector` of `Float64` representing the parameters of the system.
- **nu** : a `Matrix` of `Int64`, representing the transitions of the system, organised by row.
- **parms** : a `Vector` of `Float64` representing the parameters of the system.
- **tf** : the final simulation time (`Float64`).
- **alg** : the algorithm used (`Symbol`, either `:gillespie`, `jensen`, or `tjc`).
- **tvc** : whether rates are time varying.
"
struct SSAArgs{X,Ftype,N,P}
    x0::X
    F::Ftype
    nu::N
    parms::P
    tf::Float64
    alg::Symbol
    tvc::Bool
end

"
This type stores the output of `ssa`, and comprises of:

- **time** : a `Vector` of `Float64`, containing the times of simulated events.
- **data** : a `Matrix` of `Int64`, containing the simulated states.
- **stats** : an instance of `SSAStats`.
- **args** : arguments passed to `ssa`.

"
struct SSAResult
    time::Vector{Float64}
    data::Matrix{Int64}
    stats::SSAStats
    args::SSAArgs
end

"
This function calculates the survival probability of a process `ψᵢ` occuring after time `τ`
"
survival(ψᵢ::Distribution{ArrayLikeVariate{0}, Continuous}, τ::Float64) = 1.0 - cdf(ψᵢ, τ)

"
This function defines the conditional survival probability of a process `ψᵢ` at elapsed time `tᵢ` occuring after time `τ` 
"
survival(ψᵢ::Distribution{ArrayLikeVariate{0}, Continuous}, τ::Float64, tᵢ::Float64) = survival(ψᵢ, τ + tᵢ) / survival(ψᵢ, tᵢ) 

"
This function defines the probability of `τ`, i.e. the probability that no reaction `ψₖ` occurs between any `tₖ + τ`
"
function survival(ψₖ::AbstractArray{<:Distribution, 1}, τ::Float64, tₖ::AbstractArray{Float64,1}, n::Int64)
    Φ = 1
    for j in 1:n
        @inbounds Φ *= survival(ψₖ[j], τ + tₖ[j]) / survival(ψₖ[j], tₖ[j])
    end
    return Φ
end

" 
This function defines the joint probability that given times `tₖ` elapsed since the last occurence of each process up to a given point in time `t`,
    the next event taking place corresponds to process `ψᵢ` and will occur at time  `t + τ`
"
function joint(i::Int64, ψₖ::AbstractArray{<:Distribution, 1}, τ::Float64, tₖ::AbstractArray{Float64,1}, n::Int64) 
    return pdf(ψₖ[i], τ + tₖ[i]) / survival(ψₖ[i], τ + tₖ[i]) * survival(ψₖ, τ, tₖ, n)
end

" 
This process defines the instantaneous (hazard) rate of process `ψᵢ` at time `τ`
"
hazard(ψᵢ::Distribution{ArrayLikeVariate{0}, Continuous}, τ::Float64) = pdf(ψᵢ, τ) / survival(ψᵢ, τ)

"
This function defines the probability that the next event belongs to process `ψᵢ` given the occurrence time `τ`
"
function process(i::Int64, ψₖ::AbstractArray{<:Distribution, 1}, τ::Float64, tₖ::AbstractArray{Float64,1}, n::Int64)
    # calculate 
    Π = 0
    for j in 1:n
        @inbounds Π += hazard(ψₖ[j], tₖ[j] + τ)
    end
    return hazard(ψₖ[i], tₖ[i] + τ) / Π
end

"
This function calculates the exact sampling of the next event `ψᵢ` at `τ = 0`
"
function process_sample_exact(ψₖ::AbstractArray{<:Distribution, 1}, τ::Float64, tₖ::AbstractArray{Float64,1}, n::Int64, ignore_index::Int64)
    # make empirical pdf
    empdf = [process(i, ψₖ, τ, tₖ, n) for i in 1:n]
    sumpdf = sum(empdf)
    # solve Π(i, ψ, 0, t, n) = u to get new event i 
    i = pfsample(empdf,sumpdf,n)
    return i
end

"
This function calculates the exact sampling of the next event `ψᵢ` at `τ = 0`
"
function process_sample_approximate(ψₖ::AbstractArray{<:Distribution, 1}, τ::Float64, tₖ::AbstractArray{Float64,1}, n::Int64, ignore_index::Int64)
    # make empirical pdf
    empdf = [process(i, ψₖ, 0.0, tₖ, n) for i in 1:n]
    sumpdf = sum(empdf)
    # solve Π(i, ψ, 0, t, n) = u to get new event i 
    i = ignore_index
    while i == ignore_index
        i = pfsample(empdf,sumpdf,n)
    end
    return i
end

"
This funtion calculates the exact (albeit numerical) sampling of the survival probability of `τ`.
The default stepsize `h₀ = 1e-5` and the adaptive selection of `h` could be improved.
"
function survival_sample_exact(ψₖ::AbstractArray{<:Distribution, 1}, tₖ::AbstractArray{Float64,1}, n::Int64; h = 1e-8, cutoff = 1e-3)
    # find intital time starting point and bin size h
    while 1 - survival(ψₖ, h, tₖ, n) < cutoff
        h *= 2
    end
    # determine size of array for given h
    nsize = 2
    while survival(ψₖ, nsize*h, tₖ, n) > cutoff
        nsize *= 2
    end
    τrange = h:h:nsize*h
    empdf = [survival(ψₖ, τ, tₖ, n) for τ in τrange]
    sumpdf = sum(empdf)
    # solve Θ(ψ, τ, t, n) = u to get new time τ
    i = pfsample(empdf,sumpdf,length(τrange))
    return τrange[i]
end

"
This funtion calculates the `n→∞` sampling approximation of the survival probability of `τ` 
"
function survival_sample_approximate(ψₖ::AbstractArray{<:Distribution, 1}, tₖ::AbstractArray{Float64,1}, n::Int64)
    Φ = 0
    for j in 1:n
        @inbounds Φ += hazard(ψₖ[j], tₖ[j])
    end
    return rand(Exponential(1/Φ))
end

"
This function is a substitute for `StatsBase.sample(wv::WeightVec)`, which avoids recomputing the sum and size of the weight vector, as well as a type conversion of the propensity vector. It takes the following arguments:

- **w** : an `Array{Float64,1}`, representing propensity function weights.
- **s** : the sum of `w`.
- **n** : the length of `w`.

"
function pfsample(w::AbstractArray{Float64,1},s::Float64,n::Int64)
    t = rand() * s
    i = 1
    cw = w[1]
    while cw < t && i < n
        i += 1
        @inbounds cw += w[i]
    end
    return i
end

"
This function performs Gillespie's stochastic simulation algorithm. It takes the following arguments:

- **x0** : a `Vector` of `Int64`, representing the initial states of the system.
- **F** : a `Function` or a callable type, which itself takes two arguments; x, a `Vector` of `Int64` representing the states, and parms, a `Vector` of `Float64` representing the parameters of the system.
- **nu** : a `Matrix` of `Int64`, representing the transitions of the system, organised by row.
- **parms** : a `Vector` of `Float64` representing the parameters of the system.
- **tf** : the final simulation time (`Float64`).
"
function gillespie(x0::AbstractVector{Int64},F::Base.Callable,nu::AbstractMatrix{Int64},parms::AbstractVector{Float64},tf::Float64)
    # Args
    args = SSAArgs(x0,F,nu,parms,tf,:gillespie,false)
    # Set up time array
    ta = Vector{Float64}()
    t = 0.0
    push!(ta,t)
    # Set up initial x
    nstates = length(x0)
    x = copy(x0')
    xa = copy(Array(x0))
    # Number of propensity functions
    numpf = size(nu,1)
    # Main loop
    termination_status = "finaltime"
    nsteps = 0
    while t <= tf
        pf = F(x,parms)
        # Update time
        sumpf = sum(pf)
        if sumpf == 0.0
            termination_status = "zeroprop"
            break
        end
        dt = rand(Exponential(1/sumpf))
        t += dt
        push!(ta,t)
        # Update event
        ev = pfsample(pf,sumpf,numpf)
        if x isa SVector
            @inbounds x[1] += nu[ev,:]
        else
            deltax = view(nu,ev,:)
            for i in 1:nstates
                @inbounds x[1,i] += deltax[i]
            end
        end
        for xx in x
            push!(xa,xx)
        end
        # update nsteps
        nsteps += 1
    end
    stats = SSAStats(termination_status,nsteps)
    xar = transpose(reshape(xa,length(x),nsteps+1))
    return SSAResult(ta,xar,stats,args)
end

"
This function performs the true jump method for piecewise deterministic Markov processes. It takes the following arguments:

- **x0** : a `Vector` of `Int64`, representing the initial states of the system.
- **F** : a `Function` or a callable type, which itself takes three arguments; x, a `Vector` of `Int64` representing the states, parms, a `Vector` of `Float64` representing the parameters of the system, and t, a `Float64` representing the time of the system.
- **nu** : a `Matrix` of `Int64`, representing the transitions of the system, organised by row.
- **parms** : a `Vector` of `Float64` representing the parameters of the system.
- **tf** : the final simulation time (`Float64`).
"
function truejump(x0::AbstractVector{Int64},F::Base.Callable,nu::AbstractMatrix{Int64},parms::AbstractVector{Float64},tf::Float64)
    # Args
    args = SSAArgs(x0,F,nu,parms,tf,:tjm,true)
    # Set up time array
    ta = Vector{Float64}()
    t = 0.0
    push!(ta,t)
    # Set up initial x
    nstates = length(x0)
    x = x0'
    xa = copy(x0)
    # Number of propensity functions
    numpf = size(nu,1)
    # Main loop
    termination_status = "finaltime"
    nsteps = 0
    while t <= tf
        ds = rand(Exponential(1.0))
        f = (u)->(quadgk((u)->sum(F(x,parms,u)),t,u)[1]-ds)
        newt = fzero(f,t)
        if newt>tf
          break
        end
        t=newt
        pf = F(x,parms,t)
        # Update time
        sumpf = sum(pf)
        if sumpf == 0.0
            termination_status = "zeroprop"
            break
        end
        push!(ta,t)
        # Update event
        ev = pfsample(pf,sumpf,numpf)
        if x isa SVector
            @inbounds x[1] += nu[ev,:]
        else
            deltax = view(nu,ev,:)
            for i in 1:nstates
                @inbounds x[1,i] += deltax[i]
            end
        end
        for xx in x
            push!(xa,xx)
        end
        # update nsteps
        nsteps += 1
    end
    stats = SSAStats(termination_status,nsteps)
    xar = transpose(reshape(xa,length(x),nsteps+1))
    return SSAResult(ta,xar,stats,args)
end

"
This function performs stochastic simulation using thinning/uniformization/Jensen's method, returning only the thinned jumps. It takes the following arguments:

- **x0** : a `Vector` of `Int64`, representing the initial states of the system.
- **F** : a `Function` or a callable type, which itself takes two arguments; x, a `Vector` of `Int64` representing the states, and parms, a `Vector` of `Float64` representing the parameters of the system. In the case of time-varying systems, a third argument, a `Float64` representing the time of the system should be added
- **nu** : a `Matrix` of `Int64`, representing the transitions of the system, organised by row.
- **parms** : a `Vector` of `Float64` representing the parameters of the system.
- **tf** : the final simulation time (`Float64`).
- **max_rate**: the maximum rate (`Float64`).
"
function jensen(x0::AbstractVector{Int64},F::Base.Callable,nu::AbstractMatrix{Int64},parms::AbstractVector{Float64},tf::Float64,max_rate::Float64,thin::Bool=true)
    if thin==false
      return jensen_alljumps(x0::AbstractVector{Int64},F::Base.Callable,nu::Matrix{Int64},parms::AbstractVector{Float64},tf::Float64,max_rate::Float64)
    end
    tvc=true
    try
      F(x0,parms,0.0)
    catch
      tvc=false
    end
    # Args
    args = SSAArgs(x0,F,nu,parms,tf,:jensen,tvc)
    # Set up time array
    ta = Vector{Float64}()
    t = 0.0
    push!(ta,t)
    # Set up initial x
    nstates = length(x0)
    x = copy(x0')
    xa = copy(x0)
    # Number of propensity functions; one for no event
    numpf = size(nu,1)+1
    # Main loop
    termination_status = "finaltime"
    nsteps = 0
    while t <= tf
        dt = rand(Exponential(1/max_rate))
        t += dt
        if tvc
          pf = F(x,parms,t)
        else
          pf = F(x,parms)
        end
        # Update time
        sumpf = sum(pf)
        if sumpf == 0.0
            termination_status = "zeroprop"
            break
        end
        if sumpf > max_rate
            termination_status = "upper_bound_exceeded"
            break
        end
        # Update event
        ev = pfsample([pf; max_rate-sumpf],max_rate,numpf+1)
        if ev < numpf
            if x isa SVector
                @inbounds x[1] += nu[ev,:]
            else
                deltax = view(nu,ev,:)
                for i in 1:nstates
                    @inbounds x[1,i] += deltax[i]
                end
            end
          for xx in x
            push!(xa,xx)
          end
          push!(ta,t)
          # update nsteps
          nsteps += 1
        end
    end
    stats = SSAStats(termination_status,nsteps)
    xar = transpose(reshape(xa,length(x),nsteps+1))
    return SSAResult(ta,xar,stats,args)
end

"
This function performs stochastic simulation using thinning/uniformization/Jensen's method, returning all the jumps, both real and 'virtual'. It takes the following arguments:

- **x0** : a `Vector` of `Int64`, representing the initial states of the system.
- **F** : a `Function` or a callable type, which itself takes two arguments; x, a `Vector` of `Int64` representing the states, and parms, a `Vector` of `Float64` representing the parameters of the system. In the case of time-varying systems, a third argument, a `Float64` representing the time of the system should be added
- **nu** : a `Matrix` of `Int64`, representing the transitions of the system, organised by row.
- **parms** : a `Vector` of `Float64` representing the parameters of the system.
- **tf** : the final simulation time (`Float64`).
- **max_rate**: the maximum rate (`Float64`).
"
function jensen_alljumps(x0::AbstractVector{Int64},F::Base.Callable,nu::AbstractMatrix{Int64},parms::AbstractVector{Float64},tf::Float64,max_rate::Float64)
    # Args
    tvc=true
    try
      F(x0,parms,0.0)
    catch
      tvc=false
    end
    # Args
    args = SSAArgs(x0,F,nu,parms,tf,:jensen,tvc)
    # Set up time array
    ta = Vector{Float64}()
    t = 0.0
    push!(ta,t)
    while t < tf
      dt = rand(Exponential(1/max_rate))
      t += dt
      push!(ta,t)
    end
    nsteps=length(ta)-1
    # Set up initial x
    nstates = length(x0)
    x = copy(x0')
    xa = Array{Int64,1}(undef, (nsteps+1)*nstates)
    xa[1:nstates] = x
    # Number of propensity functions; one for no event
    numpf = size(nu,1)+1
    # Main loop
    termination_status = "finaltime"
    k=1 # step counter
    while k <= nsteps
        if tvc
          t=ta[k]
          pf=F(x,parms,t)
        else
          pf = F(x,parms)
        end
        sumpf = sum(pf)
        if sumpf == 0.0
            termination_status = "zeroprop"
            break
        end
        if sumpf > max_rate
            termination_status = "upper_bound_exceeded"
            break
        end
        # Update event
        ev = pfsample([pf; max_rate-sumpf],max_rate,numpf+1)
        if ev < numpf # true jump
          deltax = view(nu,ev,:)
          for i in 1:nstates
              @inbounds xa[k*nstates+i] = xa[(k-1)*nstates+i]+deltax[i]
          end
        else
          for i in 1:nstates
              @inbounds xa[k*nstates+i] = xa[(k-1)*nstates+i]
          end
        end
        k +=1
    end
    stats = SSAStats(termination_status,nsteps)
    xar = transpose(reshape(xa,length(x),nsteps+1))
    return SSAResult(ta,xar,stats,args)
end

"
This function performs the generalized Gillespie (non-Markovian) stochastic simulation algorithm. 
It takes the following arguments:

- **x0** : a `Vector` of `Int64`, representing the initial states of the system.
- **F** : a `Function` or a callable type, which itself takes two arguments; x, a `Vector` of `Int64` representing the states, and parms, a `Vector` of `Float64` representing the parameters of the system.
- **nu** : a `Matrix` of `Int64`, representing the transitions of the system, organised by row.
- **ψₖ** : a `Vector` of `Distribution` representing the response times of the system.
- **tf** : the final simulation time (`Float64`).
"
function nonmarkov(x0::AbstractVector{Int64},F::Base.Callable,nu::AbstractMatrix{Int64},parms::AbstractVector{Float64},tf::Float64; napprox = 50)
    # Args
    args = SSAArgs(x0,F,nu,parms,tf,:nonmarkov,false)
    # Set up time and elapsed time arrays
    ta = Vector{Float64}()
    n = size(nu,1)
    tₖ = zeros(n)
    t = 0.0
    push!(ta,t)
    # Set up initial x
    nstates = length(x0)
    x = copy(x0')
    xa = copy(Array(x0))
    # Main loop
    termination_status = "finaltime"
    nsteps = 0; ev = 0
    # warm up (get every process sampled so that tₖ vector is not zeros)
    while t <= tf && sum(tₖ .== 0) > 1
        ψₖ=F(x,parms,t,tₖ)
        if sum(survival.(ψₖ, Inf) .!= 0) == n
            termination_status = "zeroprop"
            break
        end
        # Update time
        dt = survival_sample_exact(ψₖ, tₖ, n)
        t += dt
        push!(ta,t)
        # Update event
        ev = process_sample_exact(ψₖ, dt, tₖ, n, 0)
        if x isa SVector
            @inbounds x[1] += nu[ev,:]
        else
            deltax = view(nu,ev,:)
            for i in 1:nstates
                @inbounds x[1,i] += deltax[i]
            end
        end
        for xx in x
            push!(xa,xx)
        end
        # update elapsed times of all processes
        tₖ[1:end .!= ev] .+= dt
        tₖ[ev] = 0
        # update nsteps
        nsteps += 1
    end
    # determine whether the number of processes is in the approximation limit
    if n > napprox
        survival_sampler = survival_sample_approx
        process_sampler = process_sample_approx
    else
        survival_sampler = survival_sample_exact
        process_sampler = process_sample_exact
    end
    # continue sampling but potentially with approx function
    while t <= tf
        ψₖ=F(x,parms,t,tₖ)
        if sum(survival.(ψₖ, Inf) .!= 0) == n
            termination_status = "zeroprop"
            break
        end
        # Update time
        dt = survival_sampler(ψₖ, tₖ, n)
        t += dt
        push!(ta,t)
        # Update event
        ev = process_sampler(ψₖ, dt, tₖ, n, ev)
        if x isa SVector
            @inbounds x[1] += nu[ev,:]
        else
            deltax = view(nu,ev,:)
            for i in 1:nstates
                @inbounds x[1,i] += deltax[i]
            end
        end
        for xx in x
            push!(xa,xx)
        end
        # update elapsed times of all processes
        tₖ[1:end .!= ev] .+= dt
        tₖ[ev] = 0
        # update nsteps
        nsteps += 1
    end
    stats = SSAStats(termination_status,nsteps)
    xar = transpose(reshape(xa,length(x),nsteps+1))
    return SSAResult(ta,xar,stats,args)
end

"This takes a single argument of type `SSAResult` and returns a `DataFrame`."
function ssa_data(s::SSAResult)
    hcat(DataFrame(time=s.time),DataFrame(s.data, :auto))
end
