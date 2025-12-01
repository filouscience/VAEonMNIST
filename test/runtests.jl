using VAEonMNIST
using Test
using Aqua

@testset "VAEonMNIST" begin
	@testset "Code quality (Aqua.jl)" begin
		Aqua.test_all(VAEonMNIST)
	end
	# custom test go here:
	#
end
