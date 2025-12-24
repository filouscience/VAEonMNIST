module VAEonMNIST

#export

import MLDatasets;
using  Flux;

# development purposes
nll(model, x, y) = Flux.binarycrossentropy(model(x), y, agg=sum) / (size(x)[end]);
one_hot(y::Vector{<:Integer}) = Flux.OneHotArrays.onehotbatch(y, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]);

function load_dataset()
    train_x, train_y = MLDatasets.MNIST(Float32, split=:train)[:];
    test_x,  test_y  = MLDatasets.MNIST(Float32, split=:test)[:];
    train_x = reshape(train_x, 28,28,1,:);
    test_x = reshape(test_x, 28,28,1,:);
    return (train_x, train_y, test_x, test_y);
end

function train_epoch!(model, loss, train_x, train_y; batch_size=16)
    optimiser = Flux.Adam();
    opt_state = Flux.setup(optimiser, model);
    batches = Flux.DataLoader((train_x, train_y); batchsize = batch_size, shuffle = true);
    Flux.train!(loss, model, batches, opt_state);
end

function one_hot(y::Integer; zero_pad=0)
    y >= 0 || println("warning: negative numbers not supported. taking abs.");
    y = abs(y);
    y1 = y;
    if (y > 9 || zero_pad > 0)
        n = Integer(floor(log10(y)+1));
        y1 = zeros(Integer, n+zero_pad); # vector
        for i in 1:n
            y1[i] = y % 10;
            y = Integer(floor(y / 10));
        end
        y1 = y1 |> reverse;
    end
    return Flux.OneHotArrays.onehotbatch(y1, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]);
end

function prep4plot(whcn::Array{Float32}; row_length=100)
    len = size(whcn)[end];
    mat = ones(Float32, Integer(ceil(len / row_length))*28, min(len, row_length)*28);
    for itr in 1:len
        i = Integer(ceil(itr / row_length));
        j = (itr-1) % row_length + 1;
        mat[(i-1)*28+1 : i*28, (j-1)*28+1 : j*28] = -reverse( whcn[:,:,1,itr]; dims=2)' .+ 1.f0;
    end
    return mat;
end
    
struct Reshaper
    dims::Tuple
    function Reshaper(dims...)
        return new(dims);
    end
end
function (r::Reshaper)(x) # make the struct callable
    return reshape(x, r.dims...,:);
end
Flux.@layer Reshaper;
Flux.trainable(r::Reshaper) = (;);  # empty NamedTuple of trainable parameters

struct Decoder{T <: Chain}
    m::T
    function Decoder(latent_dim::Integer)
        return new{Chain}(Chain(
            Dense(latent_dim => 64, relu),                                       # LxN => 64xN
            Dense(64 => 288, relu),                                              # 64xN => 288xN
            Reshaper(3,3,32),                                                    # 288xN => 3x3x32xN
            ConvTranspose( (3,3), 32 => 16, relu; stride=2, pad=0, outpad=0),    # 3x3x32xN => 7x7x16xN
            ConvTranspose( (3,3), 16 => 8, relu; stride=2, pad=1, outpad=1),     # 7x7x16xN => 14x14x8xN
            ConvTranspose( (3,3), 8 => 1, sigmoid; stride=2, pad=1, outpad=1)    # 14x14x8xN => 28x28x1xN
        ));
    end
    # constructor to reconstruct the trainable model from fields (Flux internal needs)
    function Decoder(m::Chain)
        return new{Chain}(m);
    end
end
(d::Decoder)(x) = d.m(x);
Flux.@layer Decoder;
Flux.trainable(d::Decoder) = (; m = d.m); # NamedTuple (names the same as field names of trainables)


include("ae_module.jl"); # vanilla AutoEncoder
using .ae_module
include("vae_module.jl"); # Variational AutoEncoder (VAE)
using .vae_module
include("cvae_module.jl"); # Conditional VAE (CVAE)
using .cvae_module


end # module VAEonMNIST
