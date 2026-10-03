# Blaom's model battle (TumorGrowth.jl docs/src/examples/04_model_battle), classical models only,
# plus the naive "last calibration value carried forward" error on the exact same records and holdouts.
using Pkg; Pkg.activate(@__DIR__)
using TumorGrowth, Statistics
records = filter(r -> r.readings >= 6, patient_data())
models = [exponential, gompertz, logistic, classical_bertalanffy, bertalanffy]
names = ["exponential", "gompertz", "logistic", "classical_bertalanffy", "bertalanffy"]
holdouts = 2
out = open(joinpath(@__DIR__, "errors_jl.csv"), "w")
println(out, join(["i", "id", "n", names..., "last_value"], ","))
println("records ", length(records), " fields ", keys(records[1])); flush(stdout)
for (i, r) in enumerate(records)
    t, v = r.T_weeks, r.Lesion_normvol
    e = try
        TumorGrowth.errors(compare(t, v, models; holdouts))
    catch err
        fill(NaN, length(models))
    end
    naive = mean(abs.(v[end-holdouts] .- v[end-holdouts+1:end]))
    id = hasproperty(r, :Pt_hashID) ? string(r.Pt_hashID) : string(i)
    println(out, join([string(i), id, string(length(v)), string.(e)..., string(naive)], ","))
    flush(out)
    i % 25 == 0 && (println(i, "/", length(records)); flush(stdout))
end
close(out); println("done")
