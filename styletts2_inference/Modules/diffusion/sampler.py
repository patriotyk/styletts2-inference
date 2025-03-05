import torch

from math import atan, cos, pi, sin, sqrt
from typing import Any, Callable, List, Optional, Tuple, Type
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torch import Tensor

from .utils import *



class Distribution:
    def __call__(self, num_samples: int, device: torch.device):
        raise NotImplementedError()


class LogNormalDistribution(Distribution):
    def __init__(self, mean: float, std: float):
        self.mean = mean
        self.std = std

    def __call__(
        self, num_samples: int, device: torch.device = torch.device("cpu")
    ) -> Tensor:
        normal = self.mean + self.std * torch.randn((num_samples,), device=device)
        return normal.exp()


""" Diffusion Classes """


def pad_dims(x: Tensor, ndim: int) -> Tensor:
    # Pads additional ndims to the right of the tensor
    return x.view(*x.shape, *((1,) * ndim))



def to_batch(
    batch_size: int,
    device: torch.device,
    x: Optional[float] = None,
    xs: Optional[Tensor] = None,
) -> Tensor:
    assert exists(x) ^ exists(xs), "Either x or xs must be provided"
    # If x provided use the same for all batch items
    if exists(x):
        xs = torch.full(size=(batch_size,), fill_value=x).to(device)
    assert exists(xs)
    return xs


class Diffusion(nn.Module):

    alias: str = ""

    """Base diffusion class"""

    def denoise_fn(
        self,
        x_noisy: Tensor,
        sigmas: Optional[Tensor] = None,
        sigma: Optional[float] = None,
        **kwargs,
    ) -> Tensor:
        raise NotImplementedError("Diffusion class missing denoise_fn")

    def forward(self, x: Tensor, noise: Tensor = None, **kwargs) -> Tensor:
        raise NotImplementedError("Diffusion class missing forward function")


class VDiffusion(Diffusion):

    alias = "v"

    def __init__(self, net: nn.Module, *, sigma_distribution: Distribution):
        super().__init__()
        self.net = net
        self.sigma_distribution = sigma_distribution

    def get_alpha_beta(self, sigmas: Tensor) -> Tuple[Tensor, Tensor]:
        angle = sigmas * pi / 2
        alpha = torch.cos(angle)
        beta = torch.sin(angle)
        return alpha, beta

    def denoise_fn(
        self,
        x_noisy: Tensor,
        sigmas: Optional[Tensor] = None,
        sigma: Optional[float] = None,
        **kwargs,
    ) -> Tensor:
        batch_size, device = x_noisy.shape[0], x_noisy.device
        sigmas = to_batch(x=sigma, xs=sigmas, batch_size=batch_size, device=device)
        return self.net(x_noisy, sigmas, **kwargs)

    def forward(self, x: Tensor, noise: Tensor = None, **kwargs) -> Tensor:
        batch_size, device = x.shape[0], x.device

        # Sample amount of noise to add for each batch element
        sigmas = self.sigma_distribution(num_samples=batch_size, device=device)
        sigmas_padded = rearrange(sigmas, "b -> b 1 1")

        # Get noise
        noise = default(noise, lambda: torch.randn_like(x))

        # Combine input and noise weighted by half-circle
        alpha, beta = self.get_alpha_beta(sigmas_padded)
        x_noisy = x * alpha + noise * beta
        x_target = noise * alpha - x * beta

        # Denoise and return loss
        x_denoised = self.denoise_fn(x_noisy, sigmas, **kwargs)
        return F.mse_loss(x_denoised, x_target)


class KDiffusion(Diffusion):
    """Elucidated Diffusion (Karras et al. 2022): https://arxiv.org/abs/2206.00364"""

    alias = "k"

    def __init__(
        self,
        net: nn.Module,
        *,
        sigma_distribution: Distribution,
        sigma_data: float,  # data distribution standard deviation
        dynamic_threshold: float = 0.0,
    ):
        super().__init__()
        self.add_module('net', net)
        self.sigma_data = sigma_data
        self.sigma_distribution = sigma_distribution
        self.dynamic_threshold = dynamic_threshold

    def get_scale_weights(self, sigmas: Tensor) -> Tuple[Tensor, ...]:
        sigma_data = self.sigma_data
        c_noise = torch.log(sigmas) * 0.25
        sigmas = rearrange(sigmas, "b -> b 1 1")
        c_skip = (sigma_data ** 2) / (sigmas ** 2 + sigma_data ** 2)
        c_out = sigmas * sigma_data * (sigma_data ** 2 + sigmas ** 2) ** -0.5
        c_in = (sigmas ** 2 + sigma_data ** 2) ** -0.5
        return c_skip, c_out, c_in, c_noise

    def denoise_fn(
        self,
        x_noisy: Tensor,
        sigmas: Optional[Tensor] = None,
        sigma: Optional[float] = None,
        **kwargs,
    ) -> Tensor:
        batch_size, device = x_noisy.shape[0], x_noisy.device
        sigmas = to_batch(x=sigma, xs=sigmas, batch_size=batch_size, device=device)

        # Predict network output and add skip connection
        c_skip, c_out, c_in, c_noise = self.get_scale_weights(sigmas)
        x_pred = self.net(c_in * x_noisy, c_noise, **kwargs)
        x_denoised = c_skip * x_noisy + c_out * x_pred

        return x_denoised

    def loss_weight(self, sigmas: Tensor) -> Tensor:
        # Computes weight depending on data distribution
        return (sigmas ** 2 + self.sigma_data ** 2) * (sigmas * self.sigma_data) ** -2

    def forward(self, x: Tensor, noise: Tensor = None, **kwargs) -> Tensor:
        batch_size, device = x.shape[0], x.device
        from einops import rearrange, reduce

        # Sample amount of noise to add for each batch element
        sigmas = self.sigma_distribution(num_samples=batch_size, device=device)
        sigmas_padded = rearrange(sigmas, "b -> b 1 1")

        # Add noise to input
        noise = default(noise, lambda: torch.randn_like(x))
        x_noisy = x + sigmas_padded * noise
        
        # Compute denoised values
        x_denoised = self.denoise_fn(x_noisy, sigmas=sigmas, **kwargs)

        # Compute weighted loss
        losses = F.mse_loss(x_denoised, x, reduction="none")
        losses = reduce(losses, "b ... -> b", "mean")
        losses = losses * self.loss_weight(sigmas)
        loss = losses.mean()
        return loss


class VKDiffusion(Diffusion):

    alias = "vk"

    def __init__(self, net: nn.Module, *, sigma_distribution: Distribution):
        super().__init__()
        self.net = net
        self.sigma_distribution = sigma_distribution

    def get_scale_weights(self, sigmas: Tensor) -> Tuple[Tensor, ...]:
        sigma_data = 1.0
        sigmas = rearrange(sigmas, "b -> b 1 1")
        c_skip = (sigma_data ** 2) / (sigmas ** 2 + sigma_data ** 2)
        c_out = -sigmas * sigma_data * (sigma_data ** 2 + sigmas ** 2) ** -0.5
        c_in = (sigmas ** 2 + sigma_data ** 2) ** -0.5
        return c_skip, c_out, c_in

    def sigma_to_t(self, sigmas: Tensor) -> Tensor:
        return sigmas.atan() / pi * 2

    def t_to_sigma(self, t: Tensor) -> Tensor:
        return (t * pi / 2).tan()

    def denoise_fn(
        self,
        x_noisy: Tensor,
        sigmas: Optional[Tensor] = None,
        sigma: Optional[float] = None,
        **kwargs,
    ) -> Tensor:
        batch_size, device = x_noisy.shape[0], x_noisy.device
        sigmas = to_batch(x=sigma, xs=sigmas, batch_size=batch_size, device=device)

        # Predict network output and add skip connection
        c_skip, c_out, c_in = self.get_scale_weights(sigmas)
        x_pred = self.net(c_in * x_noisy, self.sigma_to_t(sigmas), **kwargs)
        x_denoised = c_skip * x_noisy + c_out * x_pred
        return x_denoised

    def forward(self, x: Tensor, noise: Tensor = None, **kwargs) -> Tensor:
        batch_size, device = x.shape[0], x.device

        # Sample amount of noise to add for each batch element
        sigmas = self.sigma_distribution(num_samples=batch_size, device=device)
        sigmas_padded = rearrange(sigmas, "b -> b 1 1")

        # Add noise to input
        noise = default(noise, lambda: torch.randn_like(x))
        x_noisy = x + sigmas_padded * noise

        # Compute model output
        c_skip, c_out, c_in = self.get_scale_weights(sigmas)
        x_pred = self.net(c_in * x_noisy, self.sigma_to_t(sigmas), **kwargs)

        # Compute v-objective target
        v_target = (x - c_skip * x_noisy) / (c_out + 1e-7)

        # Compute loss
        loss = F.mse_loss(x_pred, v_target)
        return loss



class UniformDistribution(Distribution):
    def __call__(self, num_samples: int, device: torch.device = torch.device("cpu")):
        return torch.rand(num_samples, device=device)


class VKDistribution(Distribution):
    def __init__(
        self,
        min_value: float = 0.0,
        max_value: float = float("inf"),
        sigma_data: float = 1.0,
    ):
        self.min_value = min_value
        self.max_value = max_value
        self.sigma_data = sigma_data

    def __call__(
        self, num_samples: int, device: torch.device = torch.device("cpu")
    ) -> Tensor:
        sigma_data = self.sigma_data
        min_cdf = atan(self.min_value / sigma_data) * 2 / pi
        max_cdf = atan(self.max_value / sigma_data) * 2 / pi
        u = (max_cdf - min_cdf) * torch.randn((num_samples,), device=device) + min_cdf
        return torch.tan(u * pi / 2) * sigma_data


class KarrasSchedule(nn.Module):
    """https://arxiv.org/abs/2206.00364 equation 5"""

    def __init__(self, sigma_min: float, sigma_max: float, rho: float = 7.0):
        super().__init__()
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.rho = rho

    def forward(self, num_steps: int, device: Any) -> Tensor:
        rho_inv = 1.0 / self.rho
        steps = torch.arange(num_steps, device=device, dtype=torch.float32)
        sigmas = (
            self.sigma_max ** rho_inv
            + (steps / (num_steps - 1))
            * (self.sigma_min ** rho_inv - self.sigma_max ** rho_inv)
        ) ** self.rho
        sigmas = F.pad(sigmas, pad=(0, 1), value=0.0)
        return sigmas


class Sampler(nn.Module):

    diffusion_types: List[Type[Diffusion]] = []

    def forward(
        self, noise: Tensor, fn: Callable, sigmas: Tensor, num_steps: int
    ) -> Tensor:
        raise NotImplementedError()

    def inpaint(
        self,
        source: Tensor,
        mask: Tensor,
        fn: Callable,
        sigmas: Tensor,
        num_steps: int,
        num_resamples: int,
    ) -> Tensor:
        raise NotImplementedError("Inpainting not available with current sampler")



class ADPM2Sampler(Sampler):
    """https://www.desmos.com/calculator/jbxjlqd9mb"""

    diffusion_types = [KDiffusion, VKDiffusion]

    def __init__(self, rho: float = 1.0):
        super().__init__()
        self.rho = rho

    def get_sigmas(self, sigma: float, sigma_next: float) -> Tuple[float, float, float]:
        r = self.rho
        sigma_up = sqrt(sigma_next ** 2 * (sigma ** 2 - sigma_next ** 2) / sigma ** 2)
        sigma_down = sqrt(sigma_next ** 2 - sigma_up ** 2)
        sigma_mid = ((sigma ** (1 / r) + sigma_down ** (1 / r)) / 2) ** r
        return sigma_up, sigma_down, sigma_mid

    def step(self, x: Tensor, fn: Callable, sigma: float, sigma_next: float) -> Tensor:
        # Sigma steps
        sigma_up, sigma_down, sigma_mid = self.get_sigmas(sigma, sigma_next)
        # Derivative at sigma (∂x/∂sigma)
        d = (x - fn(x, sigma=sigma)) / sigma
        # Denoise to midpoint
        x_mid = x + d * (sigma_mid - sigma)
        # Derivative at sigma_mid (∂x_mid/∂sigma_mid)
        d_mid = (x_mid - fn(x_mid, sigma=sigma_mid)) / sigma_mid
        # Denoise to next
        x = x + d_mid * (sigma_down - sigma)
        # Add randomness
        x_next = x + torch.randn_like(x) * sigma_up
        return x_next

    def forward(
        self, noise: Tensor, fn: Callable, sigmas: Tensor, num_steps: int
    ) -> Tensor:
        x = sigmas[0] * noise
        # Denoise to sample
        for i in range(num_steps - 1):
            x = self.step(x, fn=fn, sigma=sigmas[i], sigma_next=sigmas[i + 1])  # type: ignore # noqa
        return x

    def inpaint(
        self,
        source: Tensor,
        mask: Tensor,
        fn: Callable,
        sigmas: Tensor,
        num_steps: int,
        num_resamples: int,
    ) -> Tensor:
        x = sigmas[0] * torch.randn_like(source)

        for i in range(num_steps - 1):
            # Noise source to current noise level
            source_noisy = source + sigmas[i] * torch.randn_like(source)
            for r in range(num_resamples):
                # Merge noisy source and current then denoise
                x = source_noisy * mask + x * ~mask
                x = self.step(x, fn=fn, sigma=sigmas[i], sigma_next=sigmas[i + 1])  # type: ignore # noqa
                # Renoise if not last resample step
                if r < num_resamples - 1:
                    sigma = sqrt(sigmas[i] ** 2 - sigmas[i + 1] ** 2)
                    x = x + sigma * torch.randn_like(x)

        return source * mask + x * ~mask


class DiffusionSampler(nn.Module):
    def __init__(
        self,
        diffusion: Diffusion,
        *,
        sampler: Sampler,
        sigma_schedule: KarrasSchedule,
        num_steps: Optional[int] = None,
        clamp: bool = True,
    ):
        super().__init__()
        self.add_module('diffusion', diffusion)
        self.add_module('sampler', sampler)
        self.sigma_schedule = sigma_schedule
        self.num_steps = num_steps
        self.clamp = clamp

        # Check sampler is compatible with diffusion type
        sampler_class = sampler.__class__.__name__
        diffusion_class = diffusion.__class__.__name__
        message = f"{sampler_class} incompatible with {diffusion_class}"
        assert diffusion.alias in [t.alias for t in sampler.diffusion_types], message

    def forward(
        self, noise: Tensor, num_steps: Optional[int] = None, **kwargs
    ) -> Tensor:
        device = noise.device
        num_steps = default(num_steps, self.num_steps)  # type: ignore
        assert exists(num_steps), "Parameter `num_steps` must be provided"
        # Compute sigmas using schedule
        sigmas = self.sigma_schedule(num_steps, device)
        # Append additional kwargs to denoise function (used e.g. for conditional unet)
        fn = lambda *a, **ka: self.diffusion.denoise_fn(*a, **{**ka, **kwargs})  # noqa
        # Sample using sampler
        x = self.sampler(noise, fn=fn, sigmas=sigmas, num_steps=num_steps)
        x = x.clamp(-1.0, 1.0) if self.clamp else x
        return x
