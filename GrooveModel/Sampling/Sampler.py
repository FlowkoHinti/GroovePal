# top k/ top p
# temperature
import json
from abc import abstractmethod, ABC
from typing import Optional, Dict, Any




class DNATokenSampler(ABC):
    """
    Abstract base class for sampling DNA token sequences from generative models.

    Provides a consistent interface for sampling with temperature,
    top-k, top-p (nucleus), etc.
    """

    @abstractmethod
    def sample(
            self,
            dna_context: dict,
            temperature: float = 1.0,
            top_k: Optional[int] = None,
            top_p: Optional[float] = None,
            max_tokens: Optional[int] = None,
            **kwargs: dict[str, Any],
    ) -> dict:
        """
        Generate a DNA token sequence given a context.

        Args:
            dna_context (dict): List of input tokens (DNA-json).
            temperature (float, optional): Softmax temperature > 0.
            top_k (int, optional): Keep only the top-k most likely tokens.
            top_p (float, optional): Nucleus sampling cutoff (prob. mass).
            max_tokens (int, optional): Maximum number of tokens to generate.
            **kwargs: Extra model-specific parameters.

        Returns:
            List[str]: Generated sequence of DNA tokens.
        """
        pass