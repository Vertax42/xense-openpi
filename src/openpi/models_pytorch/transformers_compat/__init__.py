"""Pi0-specific subclasses replacing the old ``transformers_replace`` patches.

Historically openpi shipped a set of HuggingFace transformers source files under
``src/openpi/models_pytorch/transformers_replace/`` that had to be ``cp``-ed on
top of the installed ``transformers==4.53.2`` package for PyTorch Pi0 to work.

Starting with ``transformers==5.3.0`` that invasive patching is gone: the Pi0
deltas (adaRMSNorm, gated residuals, dual-mode KV cache, unscaled image
projection) live here as small subclasses of the upstream classes.  Consumers
should import the ``Pi*`` variants below instead of the stock HF classes.
"""

from openpi.models_pytorch.transformers_compat.gemma import PiGemmaConfig
from openpi.models_pytorch.transformers_compat.gemma import PiGemmaDecoderLayer
from openpi.models_pytorch.transformers_compat.gemma import PiGemmaForCausalLM
from openpi.models_pytorch.transformers_compat.gemma import PiGemmaModel
from openpi.models_pytorch.transformers_compat.gemma import PiGemmaRMSNorm
from openpi.models_pytorch.transformers_compat.gemma import gated_residual
from openpi.models_pytorch.transformers_compat.paligemma import PiPaliGemmaForConditionalGeneration
from openpi.models_pytorch.transformers_compat.paligemma import PiPaliGemmaModel

__all__ = [
    "PiGemmaConfig",
    "PiGemmaDecoderLayer",
    "PiGemmaForCausalLM",
    "PiGemmaModel",
    "PiGemmaRMSNorm",
    "PiPaliGemmaForConditionalGeneration",
    "PiPaliGemmaModel",
    "gated_residual",
]
