import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.nn import Conv1d, ConvTranspose1d
from torch.nn.utils import weight_norm
from .utils import init_weights
from .decoder import  AdaINResBlock1, SourceModuleHnNSF, Decoder as BaseDecoder

import numpy as np 
from scipy.signal import get_window

LRELU_SLOPE = 0.1
            
class TorchSTFT(torch.nn.Module):
    def __init__(self, filter_length=800, hop_length=200, win_length=800, window='hann'):
        super().__init__()
        self.filter_length = filter_length
        self.hop_length = hop_length
        self.win_length = win_length
        self.window = torch.from_numpy(get_window(window, win_length, fftbins=True).astype(np.float32))
        self.istft_onnx = STFTONNX(model_type='istft_A', n_fft=filter_length, n_mels=80, hop_len=hop_length, max_frames=100096, window_type=torch.hann_window, pad_mode='reflect' ).eval()

    def transform(self, input_data):
        forward_transform = torch.stft(
            input_data,
            self.filter_length, self.hop_length, self.win_length, window=self.window.to(input_data.device),
            return_complex=False if torch.onnx.is_in_onnx_export() else True)

        if torch.onnx.is_in_onnx_export():
            real = forward_transform[..., 0]
            imag = forward_transform[..., 1]
            return torch.abs(real), torch.atan2(imag, real)
        else:
            return torch.abs(forward_transform), torch.angle(forward_transform)

    def inverse(self, magnitude, phase):
        if torch.onnx.is_in_onnx_export():
            inverse_transform = self.istft_onnx(magnitude, phase)
        else:
            inverse_transform = torch.istft(
                    magnitude * torch.exp(phase * 1j),
                    self.filter_length, self.hop_length, self.win_length, window=self.window.to(magnitude.device))
        return inverse_transform.unsqueeze(-2)  # unsqueeze to stay consistent with conv_transpose1d implementation

    def forward(self, input_data):
        self.magnitude, self.phase = self.transform(input_data)
        reconstruction = self.inverse(self.magnitude, self.phase)
        return reconstruction

class STFTONNX(torch.nn.Module):
    def __init__(self, model_type, n_fft, n_mels, hop_len, max_frames, window_type, pad_mode):
        super(STFTONNX, self).__init__()
        self.model_type = model_type
        self.n_fft = n_fft
        self.n_mels = n_mels
        self.hop_len = hop_len
        self.max_frames = max_frames
        self.window_type = window_type
        self.half_n_fft = self.n_fft // 2
        self.pad_mode = pad_mode
        window = {
            'bartlett': torch.bartlett_window,
            'blackman': torch.blackman_window,
            'hamming': torch.hamming_window,
            'hann': torch.hann_window,
            'kaiser': lambda x: torch.kaiser_window(x, periodic=True, beta=12.0)
        }.get(self.window_type, torch.hann_window)(self.n_fft).float()
        if self.model_type in ['stft_A', 'stft_B']:
            time_steps = torch.arange(self.n_fft).unsqueeze(0).float()
            frequencies = torch.arange(self.half_n_fft + 1).unsqueeze(1).float()
            omega = 2 * torch.pi * frequencies * time_steps / self.n_fft
            window = window.unsqueeze(0)
            self.register_buffer('cos_kernel', (torch.cos(omega) * window).unsqueeze(1))
            self.register_buffer('sin_kernel', (-torch.sin(omega) * window).unsqueeze(1))
            self.padding_zero = torch.zeros((1, 1, self.half_n_fft), dtype=torch.float32)

        elif self.model_type in ['istft_A', 'istft_B']:
            fourier_basis = torch.fft.fft(torch.eye(self.n_fft, dtype=torch.float32))
            fourier_basis = torch.vstack([
                torch.real(fourier_basis[:self.half_n_fft + 1, :]),
                torch.imag(fourier_basis[:self.half_n_fft + 1, :])
            ]).float()
            forward_basis = window * fourier_basis[:, None, :]
            inverse_basis = window * torch.linalg.pinv((fourier_basis * self.n_fft) / self.hop_len).T[:, None, :]
            n = self.n_fft + self.hop_len * (self.max_frames - 1)
            window_sum = torch.zeros(n, dtype=torch.float32)
            window_normalized = window / window.abs().max()
            total_pad = self.n_fft - window_normalized.shape[0]
            pad_left = total_pad // 2
            pad_right = total_pad - pad_left
            win_sq = torch.nn.functional.pad(window_normalized ** 2, (pad_left, pad_right), mode='constant', value=0)

            for i in range(self.max_frames):
                sample = i * self.hop_len
                window_sum[sample: min(n, sample + self.n_fft)] += win_sq[: max(0, min(self.n_fft, n - sample))]
            self.register_buffer("forward_basis", forward_basis)
            self.register_buffer("inverse_basis", inverse_basis)
            self.register_buffer("window_sum_inv", self.n_fft / (window_sum * self.hop_len))

    def forward(self, *args):
        if self.model_type == 'stft_A':
            return self.stft_A_forward(*args)
        if self.model_type == 'stft_B':
            return self.stft_B_forward(*args)
        elif self.model_type == 'istft_A':
            return self.istft_A_forward(*args)
        elif self.model_type== 'istft_B':
            return self.istft_B_forward(*args)

    def stft_A_forward(self, x):
        if self.pad_mode == 'reflect':
            x = torch.nn.functional.pad(x, (self.half_n_fft, self.half_n_fft), mode=self.pad_mode)
        else:
            x = torch.cat((self.padding_zero, x, self.padding_zero), dim=-1)
        real_part = torch.nn.functional.conv1d(x, self.cos_kernel, stride=self.hop_len)
        return real_part

    def stft_B_forward(self, x):
        if self.pad_mode == 'reflect':
            x = torch.nn.functional.pad(x, (self.half_n_fft, self.half_n_fft), mode=self.pad_mode)
        else:
            x = torch.cat((self.padding_zero, x, self.padding_zero), dim=-1)
        real_part = torch.nn.functional.conv1d(x, self.cos_kernel, stride=self.hop_len)
        image_part = torch.nn.functional.conv1d(x, self.sin_kernel, stride=self.hop_len)
        return real_part, image_part

    def istft_A_forward(self, magnitude, phase):
        inverse_transform = torch.nn.functional.conv_transpose1d(
            torch.cat((magnitude * torch.cos(phase), magnitude * torch.sin(phase)), dim=1),
            self.inverse_basis,
            stride=self.hop_len,
            padding=0,
        )
        output = inverse_transform[:, :, self.half_n_fft: -self.half_n_fft] * self.window_sum_inv[self.half_n_fft: inverse_transform.size(-1) - self.half_n_fft]
        return output

    def istft_B_forward(self, magnitude, real, imag):
        phase = torch.atan2(imag, real)
        inverse_transform = torch.nn.functional.conv_transpose1d(
            torch.cat((magnitude * torch.cos(phase), magnitude * torch.sin(phase)), dim=1),
            self.inverse_basis,
            stride=self.hop_len,
            padding=0,
        )
        output = inverse_transform[:, :, self.half_n_fft: -self.half_n_fft] * self.window_sum_inv[self.half_n_fft: inverse_transform.size(-1) - self.half_n_fft]
        return output


class Generator(torch.nn.Module):
    def __init__(self, style_dim, resblock_kernel_sizes, upsample_rates, upsample_initial_channel, resblock_dilation_sizes, upsample_kernel_sizes, gen_istft_n_fft, gen_istft_hop_size):
        super(Generator, self).__init__()

        self.num_kernels = len(resblock_kernel_sizes)
        self.num_upsamples = len(upsample_rates)
        resblock = AdaINResBlock1

        self.m_source = SourceModuleHnNSF(
                    sampling_rate=24000,
                    upsample_scale=np.prod(upsample_rates) * gen_istft_hop_size,
                    harmonic_num=8, voiced_threshod=10)
        self.f0_upsamp = torch.nn.Upsample(scale_factor=np.prod(upsample_rates) * gen_istft_hop_size)
        self.noise_convs = nn.ModuleList()
        self.noise_res = nn.ModuleList()
        
        self.ups = nn.ModuleList()
        for i, (u, k) in enumerate(zip(upsample_rates, upsample_kernel_sizes)):
            self.ups.append(weight_norm(
                ConvTranspose1d(upsample_initial_channel//(2**i), upsample_initial_channel//(2**(i+1)),
                                k, u, padding=(k-u)//2)))

        self.resblocks = nn.ModuleList()
        for i in range(len(self.ups)):
            ch = upsample_initial_channel//(2**(i+1))
            for j, (k, d) in enumerate(zip(resblock_kernel_sizes,resblock_dilation_sizes)):
                self.resblocks.append(resblock(ch, k, d, style_dim))
                
            c_cur = upsample_initial_channel // (2 ** (i + 1))
            
            if i + 1 < len(upsample_rates):  #
                stride_f0 = np.prod(upsample_rates[i + 1:])
                self.noise_convs.append(Conv1d(
                    gen_istft_n_fft + 2, c_cur, kernel_size=stride_f0 * 2, stride=stride_f0, padding=(stride_f0+1) // 2))
                self.noise_res.append(resblock(c_cur, 7, [1,3,5], style_dim))
            else:
                self.noise_convs.append(Conv1d(gen_istft_n_fft + 2, c_cur, kernel_size=1))
                self.noise_res.append(resblock(c_cur, 11, [1,3,5], style_dim))
                
                
        self.post_n_fft = gen_istft_n_fft
        self.conv_post = weight_norm(Conv1d(ch, self.post_n_fft + 2, 7, 1, padding=3))
        self.ups.apply(init_weights)
        self.conv_post.apply(init_weights)
        self.reflection_pad = torch.nn.ReflectionPad1d((1, 0))
        self.stft = TorchSTFT(filter_length=gen_istft_n_fft, hop_length=gen_istft_hop_size, win_length=gen_istft_n_fft)
        
        
    def forward(self, x, s, f0):
        with torch.no_grad():
            f0 = self.f0_upsamp(f0[:, None]).transpose(1, 2)  # bs,n,t

            har_source, noi_source, uv = self.m_source(f0)
            har_source = har_source.transpose(1, 2).squeeze(1)
            har_spec, har_phase = self.stft.transform(har_source)
            har = torch.cat([har_spec, har_phase], dim=1).to(dtype=f0.dtype)
        
        for i in range(self.num_upsamples):
            x = F.leaky_relu(x, LRELU_SLOPE)
            x_source = self.noise_convs[i](har)
            x_source = self.noise_res[i](x_source, s)

            x = self.ups[i](x)
            if i == self.num_upsamples - 1:
                x = self.reflection_pad(x)

            x = x + x_source
            xs = None
            for j in range(self.num_kernels):
                if xs is None:
                    xs = self.resblocks[i*self.num_kernels+j](x, s)
                else:
                    xs += self.resblocks[i*self.num_kernels+j](x, s)
            x = xs / self.num_kernels
        x = F.leaky_relu(x)
        x = self.conv_post(x)
        spec = torch.exp(x[:,:self.post_n_fft // 2 + 1, :]).float()
        phase = torch.sin(x[:, self.post_n_fft // 2 + 1:, :]).float()
        return self.stft.inverse(spec, phase)
    
    def fw_phase(self, x, s):
        for i in range(self.num_upsamples):
            x = F.leaky_relu(x, LRELU_SLOPE)
            x = self.ups[i](x)
            xs = None
            for j in range(self.num_kernels):
                if xs is None:
                    xs = self.resblocks[i*self.num_kernels+j](x, s)
                else:
                    xs += self.resblocks[i*self.num_kernels+j](x, s)
            x = xs / self.num_kernels
        x = F.leaky_relu(x)
        x = self.reflection_pad(x)
        x = self.conv_post(x)
        spec = torch.exp(x[:,:self.post_n_fft // 2 + 1, :])
        phase = torch.sin(x[:, self.post_n_fft // 2 + 1:, :])
        return spec, phase


class Decoder(BaseDecoder):
    def __init__(self, dim_in=512, style_dim=64,
                resblock_kernel_sizes = [3,7,11],
                upsample_rates = [10, 6],
                upsample_initial_channel=512,
                resblock_dilation_sizes=[[1,3,5], [1,3,5], [1,3,5]],
                upsample_kernel_sizes=[20, 12], 
                gen_istft_n_fft=20, gen_istft_hop_size=5):
        super().__init__(dim_in=dim_in, style_dim=style_dim)
        
        self.generator = Generator(style_dim, resblock_kernel_sizes, upsample_rates, 
                                   upsample_initial_channel, resblock_dilation_sizes, 
                                   upsample_kernel_sizes, gen_istft_n_fft, gen_istft_hop_size)