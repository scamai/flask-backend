import sys
import copy
import weakref
import contextlib
from typing import Iterable, Optional

import timm
import torch
from torch import nn
from torch.nn import functional as F


class ExponentialMovingAverage:
    """
    Tracks the exponential moving average (EMA) of a set of parameters.

    Args:
        parameters (Iterable[torch.nn.Parameter]): Parameters to compute the EMA on,
            typically from `model.parameters()`. The EMA is computed on all parameters
            provided, regardless of whether `requires_grad` is `True`. This ensures
            consistent behavior even if the set of trainable parameters changes over time.

            To exclude parameters from the EMA, simply avoid passing them to the object.
            For example, to track only trainable parameters:

                ExponentialMovingAverage(
                    parameters=[p for p in model.parameters() if p.requires_grad],
                    decay=0.9
                )

        decay (float): The decay rate for the exponential average.

        use_num_updates (bool): If True, number of updates will be used in the EMA
            calculation, otherwise a fixed decay will be applied.
    """

    def __init__(
            self,
            parameters: Iterable[torch.nn.Parameter],
            decay: float,
            use_num_updates: bool = True,
            device=None
    ):
        if decay < 0.0 or decay > 1.0:
            raise ValueError('Decay must be between 0 and 1')
        self.decay = decay
        self.num_updates = 0 if use_num_updates else None
        parameters = list(parameters)
        self.shadow_params = [
            p.clone().detach().to(device)
            for p in parameters
        ]
        self.collected_params = None
        # By maintaining only a weakref to each parameter,
        # we maintain the old GC behaviour of ExponentialMovingAverage:
        # if the model goes out of scope but the ExponentialMovingAverage
        # is kept, no references to the model or its parameters will be
        # maintained, and the model will be cleaned up.
        self._params_refs = [weakref.ref(p) for p in parameters]

    def _get_parameters(
            self,
            parameters: Optional[Iterable[torch.nn.Parameter]]
    ) -> Iterable[torch.nn.Parameter]:
        if parameters is None:
            parameters = [p() for p in self._params_refs]
            if any(p is None for p in parameters):
                raise ValueError(
                    "(One of) the parameters with which this "
                    "ExponentialMovingAverage "
                    "was initialized no longer exists (was garbage collected);"
                    " please either provide `parameters` explicitly or keep "
                    "the model to which they belong from being garbage "
                    "collected."
                )
            return parameters
        else:
            parameters = list(parameters)
            if len(parameters) != len(self.shadow_params):
                raise ValueError(
                    "Number of parameters passed as argument is different "
                    "from number of shadow parameters maintained by this "
                    "ExponentialMovingAverage"
                )
            return parameters

    def update(
            self,
            parameters: Optional[Iterable[torch.nn.Parameter]] = None
    ) -> None:
        """
        Updates the maintained parameters to reflect the latest values.

        This method should be called after each parameter update, such as after
        `optimizer.step()`.

        Args:
            parameters (Optional[Iterable[torch.nn.Parameter]]): The set of
                parameters to update. If not provided (i.e., `None`), the parameters
                used during the initialization of this `ExponentialMovingAverage` object
                will be updated instead.
        """
        parameters = self._get_parameters(parameters)
        decay = self.decay
        if self.num_updates is not None:
            self.num_updates += 1
            decay = min(
                decay,
                (1 + self.num_updates) / (10 + self.num_updates)
            )
        one_minus_decay = 1.0 - decay
        with torch.no_grad():
            for s_param, param in zip(self.shadow_params, parameters):
                tmp = (s_param - param)
                # tmp will be a new tensor so we can do in-place
                tmp.mul_(one_minus_decay)
                s_param.sub_(tmp)

    def copy_to(
            self,
            parameters: Optional[Iterable[torch.nn.Parameter]] = None
    ) -> None:
        """
        Copies the current averaged parameters into the provided collection of parameters.

        Args:
            parameters (Optional[Iterable[torch.nn.Parameter]]): The parameters
                to be updated with the stored moving averages. If not provided (i.e., `None`),
                the parameters used during the initialization of this
                `ExponentialMovingAverage` object will be updated.
        """
        parameters = self._get_parameters(parameters)
        for s_param, param in zip(self.shadow_params, parameters):
            param.data.copy_(s_param.data)

    def store(
            self,
            parameters: Optional[Iterable[torch.nn.Parameter]] = None
    ) -> None:
        """
        Stores the current parameters for later restoration.

        Args:
            parameters (Optional[Iterable[torch.nn.Parameter]]): The parameters
                to temporarily store. If not provided (i.e., `None`), the parameters
                used during the initialization of this `ExponentialMovingAverage`
                object will be stored.
        """
        parameters = self._get_parameters(parameters)
        self.collected_params = [
            param.clone()
            for param in parameters
        ]

    def restore(
        self,
        parameters: Optional[Iterable[torch.nn.Parameter]] = None
    ) -> None:
        """
        Restores the parameters saved with the `store` method.

        This method is useful when you want to temporarily use EMA parameters for
        validation and then return to the original parameters.

        Args:
            parameters (Optional[Iterable[torch.nn.Parameter]]): The parameters to be
                restored. If None, the parameters used during initialization will be restored.
        """
        if self.collected_params is None:
            raise RuntimeError(
                "This ExponentialMovingAverage has no `store()`ed weights "
                "to `restore()`"
            )
        parameters = self._get_parameters(parameters)
        for c_param, param in zip(self.collected_params, parameters):
            param.data.copy_(c_param.data)

    @contextlib.contextmanager
    def average_parameters(
        self,
        parameters: Optional[Iterable[torch.nn.Parameter]] = None
    ):
        """
        Context manager to temporarily use the EMA values for parameters.

        This is equivalent to calling `store()`, `copy_to()`, and then `restore()`, but
        ensures that the original parameters are restored after the context ends.

        Args:
            parameters (Optional[Iterable[torch.nn.Parameter]]): The parameters to use
                the EMA values for. If None, the parameters used during initialization
                will be used.
        """
        parameters = self._get_parameters(parameters)
        self.store(parameters)
        self.copy_to(parameters)
        try:
            yield
        finally:
            self.restore(parameters)

    def to(self, device=None, dtype=None) -> None:
        """
        Moves the internal EMA buffers (parameters) to the specified device.

        Args:
            device: The device to move the parameters to (e.g., 'cpu', 'cuda').
            dtype: The desired data type for the parameters.
        """

        self.shadow_params = [
            p.to(device=device, dtype=dtype)
            if p.is_floating_point()
            else p.to(device=device)
            for p in self.shadow_params
        ]
        if self.collected_params is not None:
            self.collected_params = [
                p.to(device=device, dtype=dtype)
                if p.is_floating_point()
                else p.to(device=device)
                for p in self.collected_params
            ]
        return

    def state_dict(self) -> dict:
        r"""Returns the state of the ExponentialMovingAverage as a dict."""
        # Following PyTorch conventions, references to tensors are returned:
        # "returns a reference to the state and not its copy!" -
        # https://pytorch.org/tutorials/beginner/saving_loading_models.html#what-is-a-state-dict
        return {
            "decay": self.decay,
            "num_updates": self.num_updates,
            "shadow_params": self.shadow_params,
            "collected_params": self.collected_params
        }

    def load_state_dict(self, state_dict: dict, device=None) -> None:
        """
        Loads the state of the EMA from a given state dictionary.

        Args:
            state_dict (dict): The EMA state to load, typically obtained from a
                previous call to `state_dict()`.
            device: The device to move the loaded parameters to (optional).
        """

        # deepcopy, to be consistent with module API
        state_dict = copy.deepcopy(state_dict)
        self.decay = state_dict["decay"]
        if self.decay < 0.0 or self.decay > 1.0:
            raise ValueError('Decay must be between 0 and 1')
        self.num_updates = state_dict["num_updates"]
        assert self.num_updates is None or isinstance(self.num_updates, int), \
            "Invalid num_updates"

        self.shadow_params = state_dict["shadow_params"]
        assert isinstance(self.shadow_params, list), \
            "shadow_params must be a list"
        assert all(
            isinstance(p, torch.Tensor) for p in self.shadow_params
        ), "shadow_params must all be Tensors"

        self.collected_params = state_dict["collected_params"]
        if self.collected_params is not None:
            assert isinstance(self.collected_params, list), \
                "collected_params must be a list"
            assert all(
                isinstance(p, torch.Tensor) for p in self.collected_params
            ), "collected_params must all be Tensors"
            assert len(self.collected_params) == len(self.shadow_params), \
                "collected_params and shadow_params had different lengths"

        if len(self.shadow_params) == len(self._params_refs):
            # Consistent with torch.optim.Optimizer, cast things to consistent
            # device and dtype with the parameters
            params = [p() for p in self._params_refs]
            # If parameters have been garbage collected, just load the state
            # we were given without change.
            if not any(p is None for p in params):
                # ^ parameter references are still good
                for i, p in enumerate(params):
                    self.shadow_params[i] = self.shadow_params[i].to(
                        device=p.device, dtype=p.dtype
                    )
                    if self.collected_params is not None:
                        self.collected_params[i] = self.collected_params[i].to(
                            device=p.device, dtype=p.dtype
                        )
        else:
            raise ValueError(
                "Tried to `load_state_dict()` with the wrong number of "
                "parameters in the saved state."
            )


class MFF_Expert_Convnext(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        self.name = 'Effnet'
        self.baseline_extractor = timm.create_model('convnextv2_tiny.fcmae_ft_in1k', pretrained=pretrained,
                                                    num_classes=2)

    def forward(self, x):
        x = self.baseline_extractor.forward_features(x)
        feat = self.baseline_extractor.head.global_pool(x)[:, :, 0, 0]
        x = self.baseline_extractor.head(x)
        return x, feat


class MFF_Expert_EffB0(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        self.name = 'Effnet'
        self.baseline_extractor = timm.create_model('tf_efficientnet_b0_ns', pretrained=pretrained, num_classes=2)

    def forward(self, x):
        x = self.baseline_extractor.forward_features(x)
        x = self.baseline_extractor.global_pool(x)
        if self.baseline_extractor.drop_rate > 0.:
            x = F.dropout(x, p=self.baseline_extractor.drop_rate, training=self.baseline_extractor.training)
        feat = x
        x = self.baseline_extractor.classifier(x)
        return x, feat


class MFF_Expert_EffB4(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        self.name = 'Effnet'
        self.baseline_extractor = timm.create_model('tf_efficientnet_b4_ns', pretrained=pretrained, num_classes=2)

    def forward(self, x):
        x = self.baseline_extractor.forward_features(x)
        x = self.baseline_extractor.global_pool(x)
        if self.baseline_extractor.drop_rate > 0.:
            x = F.dropout(x, p=self.baseline_extractor.drop_rate, training=self.baseline_extractor.training)
        feat = x
        x = self.baseline_extractor.classifier(x)
        return x, feat


class MFF_Expert_EffB5(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        self.name = 'Effnet'
        self.baseline_extractor = timm.create_model('tf_efficientnet_b5_ns', pretrained=pretrained, num_classes=2)

    def forward(self, x):
        x = self.baseline_extractor.forward_features(x)
        x = self.baseline_extractor.global_pool(x)
        if self.baseline_extractor.drop_rate > 0.:
            x = F.dropout(x, p=self.baseline_extractor.drop_rate, training=self.baseline_extractor.training)
        feat = x
        x = self.baseline_extractor.classifier(x)
        return x, feat


class MFF_Expert_EffB6(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        self.name = 'Effnet'
        self.baseline_extractor = timm.create_model('tf_efficientnet_b6_ns', pretrained=pretrained, num_classes=2)

    def forward(self, x):
        x = self.baseline_extractor.forward_features(x)
        x = self.baseline_extractor.global_pool(x)
        if self.baseline_extractor.drop_rate > 0.:
            x = F.dropout(x, p=self.baseline_extractor.drop_rate, training=self.baseline_extractor.training)
        feat = x
        x = self.baseline_extractor.classifier(x)
        return x, feat


class MFF_MoE(nn.Module):
    def __init__(self, pretrained=True, device=None):
        super().__init__()
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        self.name = 'Effnet'
        self.experts = nn.ModuleList()
        self.ema_state = {}
        self.ema_list = []

        expert_details = [
            'MFF_Expert_Convnext',
            'MFF_Expert_Convnext',
            'MFF_Expert_EffB4',
            'MFF_Expert_EffB4',
            'MFF_Expert_EffB5',
            'MFF_Expert_EffB5',
            'MFF_Expert_EffB6',
        ]

        for idx, cls_name in enumerate(expert_details):
            expert = getattr(sys.modules[__name__], cls_name)(pretrained=pretrained)
            ema = ExponentialMovingAverage(expert.parameters(), decay=0.995, device=self.device)
            self.experts.append(expert)
            self.ema_list.append(ema)
            self.ema_state.update({idx: ema.state_dict()})

    # self.load(path='weights/')

    def load(self, path):
        # load network weights
        weights_file = torch.load(path + 'weight.pth', map_location=self.device)
        self.experts.load_state_dict(weights_file)

        # load ema weights
        self.ema_state = torch.load(path + 'ema.state', map_location=self.device)
        for idx, expert in enumerate(self.experts):
            self.ema_list[idx].load_state_dict(self.ema_state[idx])

    def save(self, path, idx=0):
        torch.save(self.experts.state_dict(), path + 'weight.pth')
        self.ema_state.update({idx: self.ema_list[idx].state_dict()})
        torch.save(self.ema_state, path + 'ema.state')

    def forward_expert(self, x, idx, isTrain=False):
        cur_net = self.experts[idx]
        cur_ema = self.ema_list[idx]
        if isTrain:
            x, feat = cur_net(x)
        else:
            with cur_ema.average_parameters():
                x, feat = cur_net(x)
                x = F.softmax(x, dim=1)[:, 1]
        return x, feat

    def forward(self, x):
        x = [self.forward_expert(x, idx)[0] for idx in range(len(self.experts))]
        x = torch.stack(x, dim=1)
        x = torch.mean(x, dim=1)
        return x
