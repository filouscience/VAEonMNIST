module cvae_module

export CVAE, CVAE_loss

import VAEonMNIST: Reshaper, Decoder
using  Flux
import Flux.OneHotArrays as OneHot
import Flux.ChainRulesCore as ChainRules
using  Random

struct CVarEncoder
    encode_feat::Chain
    encode_mean::Dense
    encode_logsig2::Dense
    encode_class::Dense
    function CVarEncoder(latent_dim::Integer)
        return new(
            Chain(
                Conv( (3,3), 1 => 8, relu; stride=2, pad=1),   # 28x28x1xN => 14x14x8xN (stride=2 instead of MaxPool((2,2)) layer)
                Conv( (3,3), 8 => 16, relu; stride=2, pad=1),  # 14x14x8xN => 7x7x16xN
                Conv( (3,3), 16 => 32, relu; stride=2, pad=0), # 7x7x16xN => 3x3x32xN
                Reshaper(288),                                 # 3x3x32xN => 288xN
                Dense(288 => 64, relu),                        # 288xN => 64xN
            ),
            Dense(64 => latent_dim, identity),                 # 64xN => LxN
            Dense(64 => latent_dim, identity),                 # 64xN => LxN
            Dense(64 => 10, identity) # softmax applied next   # 64xN => 10xN
        );
    end
    function CVarEncoder(ef::Chain, em::Dense, es::Dense, ec::Dense)
        return new(ef, em, es, ec);
    end
end
function (e::CVarEncoder)(x)
    feat = x |> e.encode_feat;
    return (feat |> e.encode_mean,
            feat |> e.encode_logsig2,
            feat |> e.encode_class |> softmax);
end
Flux.@layer CVarEncoder;
function Flux.trainable(e::CVarEncoder)
    # NamedTuple (names the same as field names of trainables)
    return (; encode_feat = e.encode_feat,
              encode_mean = e.encode_mean,
              encode_logsig2 = e.encode_logsig2,
              encode_class = e.encode_class);
end

struct CVAE
    enco::CVarEncoder
    deco::Decoder
    function CVAE(latent_dim::Integer)
        return new(CVarEncoder(latent_dim), Decoder(latent_dim + 10));
    end
    function CVAE(e::CVarEncoder, d::Decoder)
        return new(e,d);
    end
end
function (cvae::CVAE)(x) # forward pass
    (mean, logsig2, class) = x |> cvae.enco;
    # reparametrization trick:
    z::Matrix{Float32} = mean .+ exp.(logsig2 ./ 2) .* randn(Float32, size(logsig2) );
    return [z; class] |> cvae.deco;
end
Flux.@layer CVAE;
function Flux.trainable(cvae::CVAE)
    # NamedTuple (names the same as field names of trainables)
    return (; enco = cvae.enco, deco = cvae.deco );
end

# Kullback-Leibler divergence: KL( q(z|x) || p(z) )
function KL(model, x)
    # latent space
    (mean, logsig2, class) = x |> model.enco;
    return 0.5f0 * sum( mean .^2 .+ exp.(logsig2) .- logsig2 .- 1.0f0 ) / (size(x)[end]); # batch-average
end

function class_loss(model, x, y)
    (mean, logsig2, class) = x |> model.enco;
    one_hot = OneHot.onehotbatch(y, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]);
    return Flux.crossentropy(class, one_hot, agg=sum) / (size(x)[end]);
end

struct CVAE_loss
    β::Float32
    #γ::Float32
    CVAE_loss() = return new(1.f0);
    CVAE_loss(b::Float32) = return new(b);
end
function (loss::CVAE_loss)(model, x, y)

    # encoder AND classifier
    (mean, logsig2, class) = x |> model.enco;
    
    # classification loss
    one_hot = OneHot.onehotbatch(y, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]);
    c_loss = Flux.crossentropy(class, one_hot, agg=sum) / (size(x)[end]); # batch-average
    
    # reparametrization trick:
    z::Matrix{Float32} = mean .+ exp.(logsig2 ./ 2) .* randn(Float32, size(logsig2) );
    
    # conditional decoder (do not backpropagate reconstruction loss through class condition)
    r_x =  [z; ChainRules.ignore_derivatives(class)] |> model.deco;
    
    # reconstruction loss
    r_loss = Flux.binarycrossentropy(r_x, x, agg=sum) / (size(x)[end]); # batch-average

    # total loss
    return r_loss + c_loss + loss.β * KL(model, x);
end


end # module cvae_module
