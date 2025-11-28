module ae_module

export AE, ae_loss

import VAEonMNIST:Reshaper, Decoder
using  Flux

struct Encoder{T <: Chain}
    m::T
    function Encoder(latent_dim::Integer)
        return new{Chain}(Chain(
            Conv( (3,3), 1 => 8, relu; stride=2, pad=1),   # 28x28x1xN => 14x14x8xN (stride=2 instead of MaxPool((2,2)) layer)
            Conv( (3,3), 8 => 16, relu; stride=2, pad=1),  # 14x14x8xN => 7x7x16xN
            Conv( (3,3), 16 => 32, relu; stride=2, pad=0), # 7x7x16xN => 3x3x32xN
            Reshaper(288),                                 # 3x3x32xN => 288xN
            Dense(288 => 64, relu),                        # 288xN => 64xN
            Dense(64 => latent_dim, identity)              # 64xN => LxN
        ));
    end
    function Encoder(m::Chain)
        return new{Chain}(m);
    end
end
function (e::Encoder)(x)
    return e.m(x);
end
Flux.@layer Encoder;
function Flux.trainable(e::Encoder)
    return (; m = e.m);
end

struct AE{E <: Encoder, D <: Decoder}
    enco::E
    deco::D
    function AE(latent_dim::Integer)
        return new{Encoder, Decoder}(Encoder(latent_dim), Decoder(latent_dim));
    end
    function AE(e::Encoder, d::Decoder)
        return new{Encoder, Decoder}(e,d);
    end
end
function (ae::AE)(x) 
    return x |> ae.enco |> ae.deco;
end
Flux.@layer AE;
function Flux.trainable(ae::AE)
    return (; enco = ae.enco, deco = ae.deco ); # NamedTuple (names the same as field names of trainables)
end

function ae_loss(model, x, y)
    return Flux.binarycrossentropy(model(x), y);
end


end # module ae_module