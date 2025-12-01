module vae_module

export VAE, vae_loss

import VAEonMNIST: Reshaper, Decoder
using  Flux
using  Random

struct VarEncoder
    encode_feat::Chain
    encode_mean::Dense
    encode_logsig2::Dense
    function VarEncoder(latent_dim::Integer)
        return new(
            Chain(
                Conv( (3,3), 1 => 8, relu; stride=2, pad=1),   # 28x28x1xN => 14x14x8xN (stride=2 instead of MaxPool((2,2)) layer)
                Conv( (3,3), 8 => 16, relu; stride=2, pad=1),  # 14x14x8xN => 7x7x16xN
                Conv( (3,3), 16 => 32, relu; stride=2, pad=0), # 7x7x16xN => 3x3x32xN
                Reshaper(288),                                 # 3x3x32xN => 288xN
                Dense(288 => 64, relu),                        # 288xN => 64xN
            ),
            Dense(64 => latent_dim, identity),                 # 64xN => LxN
            Dense(64 => latent_dim, identity)                  # 64xN => LxN
        );
    end
    function VarEncoder(ef::Chain, em::Dense, es::Dense)
        return new(ef, em, es);
    end
end
function (e::VarEncoder)(x)
    feat = x |> e.encode_feat;
    return (feat |> e.encode_mean, feat |> e.encode_logsig2);
end
Flux.@layer VarEncoder;
function Flux.trainable(e::VarEncoder)
    # NamedTuple (names the same as field names of trainables)
    return (; encode_feat = e.encode_feat, encode_mean = e.encode_mean, encode_logsig2 = e.encode_logsig2);
end

struct VAE
    enco::VarEncoder
    deco::Decoder
    function VAE(latent_dim::Integer)
        return new(VarEncoder(latent_dim), Decoder(latent_dim));
    end
    function VAE(e::VarEncoder, d::Decoder)
        return new(e,d);
    end
end
function (vae::VAE)(x) # forward pass
    (mean, logsig2) = x |> vae.enco;
    # reparametrization trick:
    z::Matrix{Float32} = mean .+ exp.(logsig2 ./ 2) .* randn(Float32, size(logsig2) );
    return z |> vae.deco;
end
Flux.@layer VAE;
function Flux.trainable(vae::VAE)
    # NamedTuple (names the same as field names of trainables)
    return (; enco = vae.enco, deco = vae.deco );
end

# Kullback-Leibler divergence: KL( q(z|x) || p(z) )
function KL(model, x)
    # latent space
    (mean, logsig2) = x |> model.enco;
    return 0.5 * sum( mean .^2 .+ exp.(logsig2) .- logsig2 .- 1.0 ) / (size(x)[end]); # batch-average
end

function vae_loss(model, x, y)
    # negative ELBO:
    # reconstruction loss (neg log likelihood) + KL divergence
    nll = Flux.binarycrossentropy(model(x), y, agg=sum) / (size(x)[end]); # batch-average
    return nll + KL(model, x);
end


end # module vae_module
