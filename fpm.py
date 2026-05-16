"""FlowMatchingModel wrapper (self-contained).

Wraps a neural network backbone with the transport-based flow matching framework.
Adapted from FPM.py with internal import paths adjusted.
"""

import torch
import torch.nn as nn

from transport import create_transport, Sampler


class FlowMatchingModel(nn.Module):
    def __init__(
        self,
        model: nn.Module,
        trace_num: int,
        time_steps: int,
        path_type: str = "Linear",
        prediction: str = "velocity",
        loss_weight: str = None,
        train_eps: float = None,
        sample_eps: float = None,
        sample_num: int = 5,
        device=None,
        sup_mode: str = "all",
        use_coherence: bool = False,
        sigma_obs: float = 1e-3,
        use_bayesian: bool = True,
        sampling_method: str = "ode",
        ode_sampling_method: str = "dopri5",
        ode_num_steps: int = 50,
        ode_atol: float = 1e-6,
        ode_rtol: float = 1e-3,
        sde_sampling_method: str = "Euler",
        sde_num_steps: int = 250,
        sde_diffusion_form: str = "sigma",
        sde_diffusion_norm: float = 1.0,
        sde_last_step: str = "Mean",
        sde_last_step_size: float = 0.04,
    ) -> None:
        super().__init__()
        self.model = model
        self.trace_num = trace_num
        self.time_steps = time_steps
        self.sample_num = sample_num
        self.device = device
        self.sup_mode = sup_mode
        self.use_coherence = use_coherence
        self.sigma_obs = sigma_obs
        self.use_bayesian = use_bayesian

        self.sampling_method = sampling_method
        self.ode_sampling_method = ode_sampling_method
        self.ode_num_steps = ode_num_steps
        self.ode_atol = ode_atol
        self.ode_rtol = ode_rtol
        self.sde_sampling_method = sde_sampling_method
        self.sde_num_steps = sde_num_steps
        self.sde_diffusion_form = sde_diffusion_form
        self.sde_diffusion_norm = sde_diffusion_norm
        self.sde_last_step = sde_last_step
        self.sde_last_step_size = sde_last_step_size

        self.transport = create_transport(
            path_type=path_type,
            prediction=prediction,
            loss_weight=loss_weight,
            train_eps=train_eps,
            sample_eps=sample_eps,
        )

        self.sampler = Sampler(self.transport)
        self.time_normalize = True

    def _normalize_time(self, t: torch.Tensor) -> torch.Tensor:
        if self.time_normalize:
            return t.float() * self.time_steps
        return t.float()

    def _model_wrapper(self, x: torch.Tensor, t: torch.Tensor, **model_kwargs):
        x_cond = model_kwargs.pop("x_cond", None)
        condL = model_kwargs.pop("condL", None)
        time_axis = model_kwargs.pop("time_axis", None)

        if x_cond is not None:
            model_in = torch.cat([x, x_cond], dim=1)
        else:
            model_in = x

        t_normalized = self._normalize_time(t)
        output = self.model(model_in, t_normalized, condL=condL, time_axis=time_axis)
        return output

    def forward(
        self,
        x: torch.Tensor,
        condL: tuple = None,
        x_cond: torch.Tensor = None,
        time: torch.Tensor = None,
    ) -> torch.Tensor:
        x_cond_captured = x_cond
        condL_captured = condL

        def wrapped_model(xt, t, **kwargs):
            if x_cond_captured is not None:
                if x_cond_captured.shape[0] != xt.shape[0]:
                    if x_cond_captured.shape[0] == 1:
                        x_cond_batched = x_cond_captured.expand(xt.shape[0], -1, -1, -1)
                    elif x_cond_captured.shape[0] > xt.shape[0]:
                        x_cond_batched = x_cond_captured[: xt.shape[0]]
                    else:
                        repeat_times = (xt.shape[0] + x_cond_captured.shape[0] - 1) // x_cond_captured.shape[0]
                        x_cond_batched = x_cond_captured.repeat(repeat_times, 1, 1, 1)[: xt.shape[0]]
                else:
                    x_cond_batched = x_cond_captured
                model_in = torch.cat([xt, x_cond_batched], dim=1)
            else:
                model_in = xt
            t_normalized = self._normalize_time(t)
            return self.model(model_in, t_normalized, condL=condL_captured, time_axis=kwargs.get("time_axis", None))

        model_kwargs = {}
        loss_dict = self.transport.training_losses(wrapped_model, x, model_kwargs)
        loss = loss_dict["loss"].mean()
        return loss

    @torch.inference_mode()
    def sample(
        self,
        condL: tuple = None,
        x_cond: torch.Tensor = None,
        x_mask: torch.Tensor = None,
        x_known: torch.Tensor = None,
        time_axis: torch.Tensor = None,
    ) -> torch.Tensor:
        x_cond_captured = x_cond
        condL_captured = condL
        time_axis_captured = time_axis

        def wrapped_model(xt, t, **kwargs):
            if x_cond_captured is not None:
                if x_cond_captured.shape[0] != xt.shape[0]:
                    if x_cond_captured.shape[0] == 1:
                        x_cond_batched = x_cond_captured.expand(xt.shape[0], -1, -1, -1)
                    elif x_cond_captured.shape[0] > xt.shape[0]:
                        x_cond_batched = x_cond_captured[: xt.shape[0]]
                    else:
                        repeat_times = (xt.shape[0] + x_cond_captured.shape[0] - 1) // x_cond_captured.shape[0]
                        x_cond_batched = x_cond_captured.repeat(repeat_times, 1, 1, 1)[: xt.shape[0]]
                else:
                    x_cond_batched = x_cond_captured
                model_in = torch.cat([xt, x_cond_batched], dim=1)
            else:
                model_in = xt
            t_normalized = self._normalize_time(t)
            return self.model(model_in, t_normalized, condL=condL_captured, time_axis=time_axis_captured)

        model_kwargs = {}
        b = self.sample_num
        shape = (b, 1, self.trace_num, self.time_steps)
        x0 = torch.randn(shape, device=self.device)

        if self.sampling_method == "ode":
            sample_fn = self.sampler.sample_ode(
                sampling_method=self.ode_sampling_method,
                num_steps=self.ode_num_steps,
                atol=self.ode_atol,
                rtol=self.ode_rtol,
                reverse=False,
            )
            samples = sample_fn(x0, wrapped_model, **model_kwargs)
            return samples[-1]
        elif self.sampling_method == "sde":
            sample_fn = self.sampler.sample_sde(
                sampling_method=self.sde_sampling_method,
                diffusion_form=self.sde_diffusion_form,
                diffusion_norm=self.sde_diffusion_norm,
                last_step=self.sde_last_step,
                last_step_size=self.sde_last_step_size,
                num_steps=self.sde_num_steps,
            )
            samples = sample_fn(x0, wrapped_model, **model_kwargs)
            return samples[-1]
        else:
            raise ValueError(f"Unknown sampling method: {self.sampling_method}")

    def parameters(self):
        return self.model.parameters()

    def train(self, mode: bool = True):
        self.model.train(mode)
        return self

    def eval(self):
        self.model.eval()
        return self
