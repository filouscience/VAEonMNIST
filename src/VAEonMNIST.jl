module VAEonMNIST


import MLDatasets;
using  Flux;

show_trainable(x) = Flux.trainable(x); # development purposes
nll(model, x, y) = Flux.binarycrossentropy(model(x), y, agg=sum) / (size(x)[end]);

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



end # module VAEonMNIST
