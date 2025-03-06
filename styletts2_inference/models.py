#coding:utf-8

import torch
import librosa
import yaml
import math
import torch
import torchaudio
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm, spectral_norm
from transformers import AlbertConfig, AlbertModel

from huggingface_hub import hf_hub_download

from .Modules.diffusion.sampler import KDiffusion, LogNormalDistribution, DiffusionSampler, ADPM2Sampler, KarrasSchedule
from .Modules.diffusion.modules import Transformer1d, StyleTransformer1d, AdaLayerNorm
from .Modules.diffusion.diffusion import AudioDiffusionConditional

from munch import Munch

def recursive_munch(d):
    if isinstance(d, dict):
        return Munch((k, recursive_munch(v)) for k, v in d.items())
    elif isinstance(d, list):
        return [recursive_munch(v) for v in d]
    else:
        return d

class PLBert(AlbertModel):
    def forward(self, *args, **kwargs):
        # Call the original forward method
        outputs = super().forward(*args, **kwargs)

        # Only return the last_hidden_state
        return outputs.last_hidden_state


class LearnedDownSample(nn.Module):
    def __init__(self, layer_type, dim_in):
        super().__init__()
        self.layer_type = layer_type

        if self.layer_type == 'none':
            self.conv = nn.Identity()
        elif self.layer_type == 'timepreserve':
            self.conv = spectral_norm(nn.Conv2d(dim_in, dim_in, kernel_size=(3, 1), stride=(2, 1), groups=dim_in, padding=(1, 0)))
        elif self.layer_type == 'half':
            self.conv = spectral_norm(nn.Conv2d(dim_in, dim_in, kernel_size=(3, 3), stride=(2, 2), groups=dim_in, padding=1))
        else:
            raise RuntimeError('Got unexpected donwsampletype %s, expected is [none, timepreserve, half]' % self.layer_type)
            
    def forward(self, x):
        return self.conv(x)


class DownSample(nn.Module):
    def __init__(self, layer_type):
        super().__init__()
        self.layer_type = layer_type

    def forward(self, x):
        if self.layer_type == 'none':
            return x
        elif self.layer_type == 'timepreserve':
            return F.avg_pool2d(x, (2, 1))
        elif self.layer_type == 'half':
            if x.shape[-1] % 2 != 0:
                x = torch.cat([x, x[..., -1].unsqueeze(-1)], dim=-1)
            return F.avg_pool2d(x, 2)
        else:
            raise RuntimeError('Got unexpected donwsampletype %s, expected is [none, timepreserve, half]' % self.layer_type)


class ResBlk(nn.Module):
    def __init__(self, dim_in, dim_out, actv=nn.LeakyReLU(0.2),
                 normalize=False, downsample='none'):
        super().__init__()
        self.actv = actv
        self.normalize = normalize
        self.downsample = DownSample(downsample)
        self.downsample_res = LearnedDownSample(downsample, dim_in)
        self.learned_sc = dim_in != dim_out
        self._build_weights(dim_in, dim_out)

    def _build_weights(self, dim_in, dim_out):
        self.conv1 = spectral_norm(nn.Conv2d(dim_in, dim_in, 3, 1, 1))
        self.conv2 = spectral_norm(nn.Conv2d(dim_in, dim_out, 3, 1, 1))
        if self.normalize:
            self.norm1 = nn.InstanceNorm2d(dim_in, affine=True)
            self.norm2 = nn.InstanceNorm2d(dim_in, affine=True)
        if self.learned_sc:
            self.conv1x1 = spectral_norm(nn.Conv2d(dim_in, dim_out, 1, 1, 0, bias=False))

    def _shortcut(self, x):
        if self.learned_sc:
            x = self.conv1x1(x)
        if self.downsample:
            x = self.downsample(x)
        return x

    def _residual(self, x):
        if self.normalize:
            x = self.norm1(x)
        x = self.actv(x)
        x = self.conv1(x)
        x = self.downsample_res(x)
        if self.normalize:
            x = self.norm2(x)
        x = self.actv(x)
        x = self.conv2(x)
        return x

    def forward(self, x):
        x = self._shortcut(x) + self._residual(x)
        return x / math.sqrt(2)  # unit variance


class StyleEncoder(nn.Module):
    def __init__(self, dim_in=48, style_dim=48, max_conv_dim=384):
        super().__init__()
        blocks = []
        blocks += [spectral_norm(nn.Conv2d(1, dim_in, 3, 1, 1))]

        repeat_num = 4
        for _ in range(repeat_num):
            dim_out = min(dim_in*2, max_conv_dim)
            blocks += [ResBlk(dim_in, dim_out, downsample='half')]
            dim_in = dim_out

        blocks += [nn.LeakyReLU(0.2)]
        blocks += [spectral_norm(nn.Conv2d(dim_out, dim_out, 5, 1, 0))]
        blocks += [nn.AdaptiveAvgPool2d(1)]
        blocks += [nn.LeakyReLU(0.2)]
        self.shared = nn.Sequential(*blocks)

        self.unshared = nn.Linear(dim_out, style_dim)

    def forward(self, x):
        h = self.shared(x)
        h = h.view(h.size(0), -1)
        s = self.unshared(h)
    
        return s


class LinearNorm(torch.nn.Module):
    def __init__(self, in_dim, out_dim, bias=True, w_init_gain='linear'):
        super(LinearNorm, self).__init__()
        self.linear_layer = torch.nn.Linear(in_dim, out_dim, bias=bias)

        torch.nn.init.xavier_uniform_(
            self.linear_layer.weight,
            gain=torch.nn.init.calculate_gain(w_init_gain))

    def forward(self, x):
        return self.linear_layer(x)


class LayerNorm(nn.Module):
    def __init__(self, channels, eps=1e-5):
        super().__init__()
        self.channels = channels
        self.eps = eps

        self.gamma = nn.Parameter(torch.ones(channels))
        self.beta = nn.Parameter(torch.zeros(channels))

    def forward(self, x):
        x = x.transpose(1, 2)
        x = F.layer_norm(x, (self.channels,), self.gamma, self.beta, self.eps)
        return x.transpose(1, 2)


class TextEncoder(nn.Module):
    def __init__(self, channels, kernel_size, depth, n_symbols, actv=nn.LeakyReLU(0.2)):
        super().__init__()
        self.embedding = nn.Embedding(n_symbols, channels)
        self.lstm = nn.LSTM(channels, channels//2, 1, batch_first=True, bidirectional=True)

        padding = (kernel_size - 1) // 2
        self.cnn = nn.ModuleList()
        for _ in range(depth):
            self.cnn.append(nn.Sequential(
                weight_norm(nn.Conv1d(channels, channels, kernel_size=kernel_size, padding=padding)),
                LayerNorm(channels),
                actv,
                nn.Dropout(0.2),
            ))
        # self.cnn = nn.Sequential(*self.cnn)
    
    def forward(self, x, input_lengths, m):
        x = self.embedding(x)  # [B, T, emb]
        x = x.transpose(1, 2)  # [B, emb, T]
        m = m.to(input_lengths.device).unsqueeze(1)
        x.masked_fill_(m, 0.0)
        
        for c in self.cnn:
            x = c(x)
            x.masked_fill_(m, 0.0)
            
        x = x.transpose(1, 2)  # [B, T, chn]

        input_lengths = input_lengths
        x = nn.utils.rnn.pack_padded_sequence(
            x, input_lengths, batch_first=True, enforce_sorted=False)

        self.lstm.flatten_parameters()
        x, _ = self.lstm(x)
        x, _ = nn.utils.rnn.pad_packed_sequence(
            x, batch_first=True)
                
        x = x.transpose(2, 1)
        x_pad = torch.zeros([x.shape[0], x.shape[1], m.shape[-1]])

        x_pad[:, :, :x.shape[-1]] = x
        x = x_pad.to(x.device)
        
        x.masked_fill_(m, 0.0)
        
        return x

    def inference(self, x):
        x = self.embedding(x)
        x = x.transpose(1, 2)
        x = self.cnn(x)
        x = x.transpose(1, 2)
        self.lstm.flatten_parameters()
        x, _ = self.lstm(x)
        return x
    
    def length_to_mask(self, lengths):
        mask = torch.arange(lengths.max()).unsqueeze(0).expand(lengths.shape[0], -1).type_as(lengths)
        mask = torch.gt(mask+1, lengths.unsqueeze(1))
        return mask


class AdaIN1d(nn.Module):
    def __init__(self, style_dim, num_features):
        super().__init__()
        self.norm = nn.InstanceNorm1d(num_features, affine=True)
        self.fc = nn.Linear(style_dim, num_features*2)

    def forward(self, x, s):
        h = self.fc(s)
        h = h.view(h.size(0), h.size(1), 1)
        gamma, beta = torch.chunk(h, chunks=2, dim=1)
        return (1 + gamma) * self.norm(x) + beta


class UpSample1d(nn.Module):
    def __init__(self, layer_type):
        super().__init__()
        self.layer_type = layer_type

    def forward(self, x):
        if self.layer_type == 'none':
            return x
        else:
            return F.interpolate(x, scale_factor=2, mode='nearest')


class AdainResBlk1d(nn.Module):
    def __init__(self, dim_in, dim_out, style_dim=64, actv=nn.LeakyReLU(0.2),
                 upsample='none', dropout_p=0.0):
        super().__init__()
        self.actv = actv
        self.upsample_type = upsample
        self.upsample = UpSample1d(upsample)
        self.learned_sc = dim_in != dim_out
        self._build_weights(dim_in, dim_out, style_dim)
        self.dropout = nn.Dropout(dropout_p)
        
        if upsample == 'none':
            self.pool = nn.Identity()
        else:
            self.pool = weight_norm(nn.ConvTranspose1d(dim_in, dim_in, kernel_size=3, stride=2, groups=dim_in, padding=1, output_padding=1))
        
        
    def _build_weights(self, dim_in, dim_out, style_dim):
        self.conv1 = weight_norm(nn.Conv1d(dim_in, dim_out, 3, 1, 1))
        self.conv2 = weight_norm(nn.Conv1d(dim_out, dim_out, 3, 1, 1))
        self.norm1 = AdaIN1d(style_dim, dim_in)
        self.norm2 = AdaIN1d(style_dim, dim_out)
        if self.learned_sc:
            self.conv1x1 = weight_norm(nn.Conv1d(dim_in, dim_out, 1, 1, 0, bias=False))

    def _shortcut(self, x):
        x = self.upsample(x)
        if self.learned_sc:
            x = self.conv1x1(x)
        return x

    def _residual(self, x, s):
        x = self.norm1(x, s)
        x = self.actv(x)
        x = self.pool(x)
        x = self.conv1(self.dropout(x))
        x = self.norm2(x, s)
        x = self.actv(x)
        x = self.conv2(self.dropout(x))
        return x

    def forward(self, x, s):
        out = self._residual(x, s)
        out = (out + self._shortcut(x)) / math.sqrt(2)
        return out


class ProsodyPredictor(nn.Module):

    def __init__(self, style_dim, d_hid, nlayers, max_dur=50, dropout=0.1):
        super().__init__() 
        
        self.add_module('text_encoder', DurationEncoder(sty_dim=style_dim, 
                                            d_model=d_hid,
                                            nlayers=nlayers, 
                                            dropout=dropout))

        self.lstm = nn.LSTM(d_hid + style_dim, d_hid // 2, 1, batch_first=True, bidirectional=True)
        self.duration_proj = LinearNorm(d_hid, max_dur)
        
        self.shared = nn.LSTM(d_hid + style_dim, d_hid // 2, 1, batch_first=True, bidirectional=True)
        self.F0 = nn.ModuleList()
        self.F0.append(AdainResBlk1d(d_hid, d_hid, style_dim, dropout_p=dropout))
        self.F0.append(AdainResBlk1d(d_hid, d_hid // 2, style_dim, upsample=True, dropout_p=dropout))
        self.F0.append(AdainResBlk1d(d_hid // 2, d_hid // 2, style_dim, dropout_p=dropout))

        self.N = nn.ModuleList()
        self.N.append(AdainResBlk1d(d_hid, d_hid, style_dim, dropout_p=dropout))
        self.N.append(AdainResBlk1d(d_hid, d_hid // 2, style_dim, upsample=True, dropout_p=dropout))
        self.N.append(AdainResBlk1d(d_hid // 2, d_hid // 2, style_dim, dropout_p=dropout))
        
        self.F0_proj = nn.Conv1d(d_hid // 2, 1, 1, 1, 0)
        self.N_proj = nn.Conv1d(d_hid // 2, 1, 1, 1, 0)


    def forward(self, texts, style, text_lengths, alignment, m):
        d = self.text_encoder(texts, style, text_lengths, m)
        
        batch_size = d.shape[0]
        text_size = d.shape[1]
        
        # predict duration
        input_lengths = text_lengths
        x = nn.utils.rnn.pack_padded_sequence(
            d, input_lengths, batch_first=True, enforce_sorted=False)
        
        m = m.to(text_lengths.device).unsqueeze(1)
        
        self.lstm.flatten_parameters()
        x, _ = self.lstm(x)
        x, _ = nn.utils.rnn.pad_packed_sequence(
            x, batch_first=True)
        
        x_pad = torch.zeros([x.shape[0], m.shape[-1], x.shape[-1]])

        x_pad[:, :x.shape[1], :] = x
        x = x_pad.to(x.device)
                
        duration = self.duration_proj(nn.functional.dropout(x, 0.5, training=self.training))
        
        en = (d.transpose(2, 2) @ alignment)

        return duration.squeeze(-1), en
    
    def F0Ntrain(self, x, s):
        x, _ = self.shared(x.transpose(2, 1))
        
        F0 = x.transpose(2, 1)
        for block in self.F0:
            F0 = block(F0, s)
        F0 = self.F0_proj(F0)

        N = x.transpose(2, 1)
        for block in self.N:
            N = block(N, s)
        N = self.N_proj(N)
        
        return F0.squeeze(1), N.squeeze(1)
    
    def length_to_mask(self, lengths):
        mask = torch.arange(lengths.max()).unsqueeze(0).expand(lengths.shape[0], -1).type_as(lengths)
        mask = torch.gt(mask+1, lengths.unsqueeze(1))
        return mask


class DurationEncoder(nn.Module):

    def __init__(self, sty_dim, d_model, nlayers, dropout=0.1):
        super().__init__()
        self.lstms = nn.ModuleList()
        for _ in range(nlayers):
            self.lstms.append(nn.LSTM(d_model + sty_dim, 
                                 d_model // 2, 
                                 num_layers=1, 
                                 batch_first=True, 
                                 bidirectional=True, 
                                 dropout=dropout))
            self.lstms.append(AdaLayerNorm(sty_dim, d_model))
        
        
        self.dropout = dropout
        self.d_model = d_model
        self.sty_dim = sty_dim

    def forward(self, x, style, text_lengths, m):
        masks = m.to(text_lengths.device)
        
        x = x.permute(2, 0, 1)
        s = style.expand(x.shape[0], x.shape[1], -1)
        x = torch.cat([x, s], axis=-1)
        x.masked_fill_(masks.unsqueeze(-1).transpose(0, 1), 0.0)
                
        x = x.transpose(0, 1)
        input_lengths = text_lengths.clone()
        x = x.transpose(2, 1)
        
        for block in self.lstms:
            if isinstance(block, AdaLayerNorm):
                r = block(x.transpose(2, 1), style)
                x = r.transpose(2, 1)
                x = torch.cat([x, s.permute(1, -1, 0)], axis=1)
                x.masked_fill_(masks.unsqueeze(-1).transpose(2, 1), 0.0)
            else:
                x = x.transpose(2, 1)
                x = nn.utils.rnn.pack_padded_sequence(
                    x, input_lengths, batch_first=True, enforce_sorted=False)
                block.flatten_parameters()
                x, _ = block(x)
                x, _ = nn.utils.rnn.pad_packed_sequence(
                    x, batch_first=True)
                x = F.dropout(x, p=self.dropout, training=self.training)
                x = x.transpose(2, 1)
                
                x_pad = torch.zeros([x.shape[0], x.shape[1], m.shape[-1]])

                x_pad[:, :, :x.shape[-1]] = x
                x = x_pad.to(x.device)
        return x.transpose(2, 1)
    
    def inference(self, x, style):
        x = self.embedding(x.transpose(2, 1)) * math.sqrt(self.d_model)
        style = style.expand(x.shape[0], x.shape[1], -1)
        x = torch.cat([x, style], axis=-1)
        src = self.pos_encoder(x)
        output = self.transformer_encoder(src).transpose(0, 1)
        return output
    
    def length_to_mask(self, lengths):
        mask = torch.arange(lengths.max()).unsqueeze(0).expand(lengths.shape[0], -1).type_as(lengths)
        mask = torch.gt(mask+1, lengths.unsqueeze(1))
        return mask
    

class StyleTTS2(nn.Module):
    def __init__(self, hf_path = '', config_path='', weights_path = '', device=torch.device('cpu')):
        super().__init__()
        self.noise = None
        self.device = device
        if hf_path:
            weights_path = hf_hub_download(repo_id=hf_path, filename="pytorch_model.bin")
            config_path = hf_hub_download(repo_id=hf_path, filename="config.yml")

        self.config = recursive_munch(yaml.safe_load(open(config_path)))
        self.weights = torch.load(weights_path, map_location='cpu', weights_only=True)

        self.to_mel = torchaudio.transforms.MelSpectrogram(n_mels=80, n_fft=2048, win_length=1200, hop_length=300)
        self.mean = self.config.model_params.diffusion.dist.mean
        self.std = self.config.model_params.diffusion.dist.std

        
        self.add_module('plbert', PLBert(AlbertConfig(**self.config.plbert_params)))
        self.add_module('plbert_encoder', nn.Linear(self.config.plbert_params.hidden_size, self.config.model_params.hidden_dim))
        
        params = self.config.model_params
        if params.decoder.type == "istftnet":
            from .Modules.istftnet import Decoder
            self.add_module('decoder', Decoder(dim_in=params.hidden_dim, style_dim=params.style_dim, dim_out=params.n_mels,
                    resblock_kernel_sizes = params.decoder.resblock_kernel_sizes,
                    upsample_rates = params.decoder.upsample_rates,
                    upsample_initial_channel=params.decoder.upsample_initial_channel,
                    resblock_dilation_sizes=params.decoder.resblock_dilation_sizes,
                    upsample_kernel_sizes=params.decoder.upsample_kernel_sizes, 
                    gen_istft_n_fft=params.decoder.gen_istft_n_fft, gen_istft_hop_size=params.decoder.gen_istft_hop_size))
        else:
            from .Modules.hifigan import Decoder
            self.add_module('decoder', Decoder(dim_in=params.hidden_dim, style_dim=params.style_dim, dim_out=params.n_mels,
                    resblock_kernel_sizes = params.decoder.resblock_kernel_sizes,
                    upsample_rates = params.decoder.upsample_rates,
                    upsample_initial_channel=params.decoder.upsample_initial_channel,
                    resblock_dilation_sizes=params.decoder.resblock_dilation_sizes,
                    upsample_kernel_sizes=params.decoder.upsample_kernel_sizes))
        self.add_module('text_encoder', TextEncoder(channels=params.hidden_dim, kernel_size=5, depth=params.n_layer, n_symbols=params.n_token))
    
        self.add_module('predictor', ProsodyPredictor(style_dim=params.style_dim, d_hid=params.hidden_dim, nlayers=params.n_layer, max_dur=params.max_dur, dropout=params.dropout))
    
        self.style_encoder = StyleEncoder(dim_in=params.dim_in, style_dim=params.style_dim, max_conv_dim=params.hidden_dim) # acoustic style encoder
        self.add_module('predictor_encoder', StyleEncoder(dim_in=params.dim_in, style_dim=params.style_dim, max_conv_dim=params.hidden_dim)) # prosodic style encoder
        if params.multispeaker:
            transformer = StyleTransformer1d(channels=params.style_dim*2, 
                                    context_embedding_features=self.config.plbert_params.hidden_size,
                                    context_features=params.style_dim*2, 
                                    **params.diffusion.transformer)
        else:
            transformer = Transformer1d(channels=params.style_dim*2, 
                                    context_embedding_features=self.config.plbert_params.hidden_size,
                                    **params.diffusion.transformer)
    
        diffusion = AudioDiffusionConditional(
            in_channels=1,
            embedding_max_length=self.config.plbert_params.max_position_embeddings,
            embedding_features=self.config.plbert_params.hidden_size,
            embedding_mask_proba=params.diffusion.embedding_mask_proba, # Conditional dropout of batch elements,
            channels=params.style_dim*2,
            context_features=params.style_dim*2,
        )
        
        
        
        diffusion.diffusion = KDiffusion(
            net=diffusion.unet,
            sigma_distribution=LogNormalDistribution(mean = params.diffusion.dist.mean, std = params.diffusion.dist.std),
            sigma_data=params.diffusion.dist.sigma_data, # a placeholder, will be changed dynamically when start training diffusion model
            dynamic_threshold=0.0 
        )
        diffusion.diffusion.net = transformer
        diffusion.unet = transformer
        self.add_module('diffusion', diffusion)

        sampler = DiffusionSampler(diffusion.diffusion,
                                       sampler=ADPM2Sampler(),
                                       sigma_schedule=KarrasSchedule(sigma_min=0.0001, sigma_max=3.0, rho=9.0),
                                       clamp=False)
        
        self.add_module('sampler', sampler)
        self.load_state_dict(self.weights)
        self.eval()


    def _load_state_dict(self, model, params):
        try:
            model.load_state_dict(params)
        except:
            from collections import OrderedDict
            state_dict = params
            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                name = k[7:] # remove `module.`
                new_state_dict[name] = v

            model.load_state_dict(new_state_dict, strict=False)
            model.to(self.device)
    
    
    def load_state_dict(self, params):
        self._load_state_dict(self.plbert, params['bert'])
        self._load_state_dict(self.plbert_encoder, params['bert_encoder'])
        self._load_state_dict(self.decoder, params['decoder'])
        self._load_state_dict(self.text_encoder, params['text_encoder'])
        self._load_state_dict(self.predictor, params['predictor'])
        self._load_state_dict(self.diffusion, params['diffusion'])
        self._load_state_dict(self.predictor_encoder, params['predictor_encoder'])
        self._load_state_dict(self.style_encoder, params['style_encoder'])
        #self._load_state_dict(self.sampler, params['sampler'])
        #self.register_buffer("style", params['style'])


    
    def preprocess(self, wave):
        wave_tensor = torch.from_numpy(wave).float()
        mel_tensor = self.to_mel(wave_tensor)
        mel_tensor = (torch.log(1e-5 + mel_tensor.unsqueeze(0)) - self.mean) / self.std
        return mel_tensor.unsqueeze(1)
    
    
    def compute_style(self, voice_audio):
        wave, sr = librosa.load(voice_audio, sr=24000)
        audio, _ = librosa.effects.trim(wave, top_db=30)
        if sr != 24000:
            audio = librosa.resample(audio, sr, 24000)
        mel_tensor = self.preprocess(audio).to(self.device)

        with torch.no_grad():
            ref_s = self.style_encoder(mel_tensor)
            ref_p = self.predictor_encoder(mel_tensor)

        return torch.cat([ref_s, ref_p], dim=1)


    
    def onnx_alignment(self, pred_dur, max_length):
        seq_length = pred_dur.sum().to(dtype=torch.int64)
        device = pred_dur.device
                
        clipped_dur = torch.clamp(pred_dur, 0, seq_length)
        
        zero_tensor = torch.zeros(1, device=device)
        prefix_sum = torch.cat([zero_tensor, torch.cumsum(clipped_dur[:-1], dim=0)])
        
        positions = torch.arange(seq_length, device=device).unsqueeze(0).expand(max_length, -1)
        
        start_positions = prefix_sum.unsqueeze(1)
        
        end_positions = torch.min(
            start_positions + clipped_dur.unsqueeze(1),
            start_positions.new_full(start_positions.shape, seq_length, dtype=torch.float)
        )
        
        mask = (positions >= start_positions) & (positions < end_positions)

        return mask.float()
    
    
    def forward(self, tokens, voice=None, speed = 1.0, alpha=0.0,  beta=0.0, embedding_scale=1.0, diffusion_steps=10, s_prev=torch.zeros(1,256)):
        
        tokens = tokens.to(self.device)
        tokens = torch.cat([torch.LongTensor([0]),tokens], axis=0)
        tokens = tokens.unsqueeze(0)
        with torch.no_grad():
            if self.noise is None or s_prev[0][0] == 0:
                self.noise = torch.randn(1,1,256).to(self.device)
            
            input_lengths = tokens.new_full((tokens.shape[0],), tokens.shape[1], dtype=torch.long)
            
            mask = torch.arange(input_lengths.max()).unsqueeze(0).expand(input_lengths.shape[0], -1).type_as(input_lengths)
            text_mask = torch.gt(mask+1, input_lengths.unsqueeze(1)).to(tokens.device)            
            t_en = self.text_encoder(tokens, input_lengths, text_mask)

            
            bert_dur = self.plbert(tokens, (~text_mask).int())
            d_en = self.plbert_encoder(bert_dur).transpose(2, 1)

            if voice is not None:
                s_pred = self.sampler(noise = self.noise,
                                embedding=bert_dur,
                                embedding_scale=embedding_scale,
                                features=voice, # reference from the same speaker as the embedding
                                num_steps=diffusion_steps).squeeze(1)
            else:
                s_pred = self.sampler(noise = self.noise,
                                      embedding=bert_dur[0].unsqueeze(0),
                                      num_steps=diffusion_steps,
                                      embedding_scale=embedding_scale).squeeze(0)
            
            
            is_not_empty = (s_prev.abs().sum() > 0).float()
            s_pred = is_not_empty * (alpha * s_prev + (1 - alpha) * s_pred) + (1 - is_not_empty) * s_pred
            
            s = s_pred[:, 128:]
            ref = s_pred[:, :128]
            
            if voice is not None:
                ref = alpha * ref + (1 - alpha)  * voice[:, :128]
                s = beta * s + (1 - beta)  * voice[:, 128:]

            
            d = self.predictor.text_encoder(d_en, s, input_lengths, text_mask)            
            x, _ = self.predictor.lstm(d)
            
            duration = self.predictor.duration_proj(x)
            duration = torch.sigmoid(duration).sum(axis=-1)/speed

            
            pred_dur = torch.round(duration.squeeze()).clamp(min=1)

                    
            if voice is not None:
                pred_dur[0] = 30


            if torch.onnx.is_in_onnx_export():
                pred_aln_trg = self.onnx_alignment(pred_dur, input_lengths[0])
            else:
                pred_aln_trg = torch.zeros(input_lengths, int(pred_dur.sum().data))
                c_frame = 0
                for i in range(pred_aln_trg.size(0)):
                    end_idx = min(c_frame + int(pred_dur[i]), pred_aln_trg.shape[1])
                    pred_aln_trg[i, c_frame:end_idx] = 1
                    c_frame += int(pred_dur[i])
                    if c_frame >= pred_aln_trg.shape[1]:
                        break

            # encode prosody
            en = (d.transpose(2, 1) @ pred_aln_trg.unsqueeze(0).to(self.device))

            
            F0_pred, N_pred = self.predictor.F0Ntrain(en, s)
            asr = (t_en @ pred_aln_trg.unsqueeze(0).to(self.device))

            
            out = self.decoder(asr, F0_pred, N_pred, ref.squeeze().unsqueeze(0))
            if voice is not None:
                out = out[:,:, 14500:]
            return out.squeeze(), s_pred
