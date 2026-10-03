using Pkg; Pkg.activate(@__DIR__); Pkg.add("TumorGrowth"); Pkg.precompile()
using TumorGrowth; println(pkgversion(TumorGrowth)); println(length(patient_data()))
