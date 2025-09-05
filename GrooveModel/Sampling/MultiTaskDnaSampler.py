import math
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union, Literal

import torch
from torch import nn
from dacite import from_dict, Config as DaciteConfig
from omegaconf import DictConfig, OmegaConf

from Configs import MAX_SEQUENCE_LENGTH
# --- these come from your codebase ---
from GrooveModel.Embedding.MultiTaskDnaEmbedding import MultiTaskDNAEmbeddingConfig
from GrooveModel.Models import ModelConfigxLstm, MultiTaskDNAxLSTM
from GrooveModel.Sampling.Sampler import DNATokenSampler, _load_cfg, _select_checkpoint_path, _categorical_sample
from GrooveModel.Tokenizer.MultiTaskDnaTokenizer import MultiTaskDnaTokenizer
from GrooveModel.Utils.BeatUnit import decode_beat_unit
from GrooveModel.Utils.DNAOffset import quantize_offset, decode_offset_ticks
from GrooveModel.Utils.DNAValue import decode_instrument
from GrooveModel.Utils.DNAVelocity import encode_velocity, normalize_velocity_tensor, quantize_velocity, decode_velocity
from GrooveModel.Utils.TimeSignatures import decode_time_signature

# -----------------------------
# Helper: default column order
# -----------------------------
TOKEN_COLS = {
    "instrument": 0,
    "velocity": 1,
    "beat_unit": 2,
    "offset": 3,
    "grid_factor": 4,
    "bpm": 5,
    "time_signature": 6,
}


# -----------------------------
# Multitask DNA Sampler
# -----------------------------

class MultitaskDNASampler(DNATokenSampler):
    """
    - Builds MultiTaskDNAxLSTM from a config (incl. tokenizer flags like absolute grid units).
    - Restores weights from checkpoint.
    - Tokenizes JSON context(s), runs autoregressive sampling with teacher forcing.
    - Returns DNA-shaped payload(s) for each input context and requested number of variations.
    """

    def __init__(
            self,
            cfg_or_path: Union[str, Path, DictConfig],
            checkpoint: Optional[Union[str, Path]] = None,
            device: Optional[torch.device] = None,
            use_best: bool = True,
    ):
        # Config + device
        self.cfg = _load_cfg(cfg_or_path)
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Build model
        self.embedding_config = from_dict(
            MultiTaskDNAEmbeddingConfig,
            OmegaConf.to_container(self.cfg.embedding, resolve=True),
            config=DaciteConfig(strict=True),
        )
        model_config = from_dict(
            ModelConfigxLstm,
            OmegaConf.to_container(self.cfg.model, resolve=True),
            config=DaciteConfig(strict=True),
        )
        self.model: MultiTaskDNAxLSTM = (MultiTaskDNAxLSTM(model_config, self.embedding_config)
                                         .to(self.device)
                                         .eval())

        # Checkpoint
        ckpt_path = _select_checkpoint_path(self.cfg, checkpoint, use_best)
        state = torch.load(str(ckpt_path), map_location="cpu")
        self.model.load_state_dict(state["model_state_dict"], strict=True)

        # Tokenizer (propagate tokenizer kwargs so flags like absolute_grid_units are honored)
        self.tok_kwargs: Dict[str, Any] = dict(getattr(self.cfg.dataset, "tokenizer", {}))
        self.absolute_grid_units: bool = bool(self.tok_kwargs.get("absolute_grid_units", False))
        self.tokenizer = MultiTaskDnaTokenizer()


    def _quantize_regression_values(self,
                                   reg_value: torch.Tensor,
                                   column: Literal["velocity","offset"],
                                   return_encoding: bool=False,
                                   ticks_per_gu: Optional[int] = None
                                   ) -> Union[int, float]:

        match column:
            case "velocity":
                step_id, step_val = quantize_velocity(reg_value.item(), return_as="both")
                if return_encoding:
                    return step_id
                else:
                    return step_val
            case "offset":
                encoded_id, step_val = quantize_offset(reg_value.item(),
                                                       return_as="both",
                                                       ticks_per_grid_unit=ticks_per_gu,
                                                       start_at_zero=True)
                if return_encoding:
                    return encoded_id
                else:
                    return step_val

    @torch.no_grad()
    def generate_autoregressive(self,
                          context_tokens: torch.Tensor,
                          beat_units: torch.Tensor,
                          dna_meta: dict[str, Any],
                          temperature: float = 1.0,
                          top_k: Optional[int] = None,
                          top_p: Optional[float] = None,
                          max_tokens: int = MAX_SEQUENCE_LENGTH // 2,
                          ):

        out_steps = torch.zeros([context_tokens.size(1) + max_tokens, context_tokens.size(2)], dtype=torch.long).to(self.device)

        state: dict[str, dict[str, tuple[torch.Tensor, ...]]] = None
        for t in range(context_tokens.size(1) + max_tokens):
            if t < context_tokens.size(1):
                context_tokens_t = context_tokens[:, t:t + 1, :]  # [1,1,F]
                beat_units_t = beat_units[:, t:t + 1]

                class_step, reg_step, state = self.model.step((context_tokens_t, beat_units_t), state=state)
            else:
                context_tokens_t = out_steps[t-1].unsqueeze(0).unsqueeze(1)  # [1,1,F]
                beat_units_t = torch.zeros((1,1)) # Placeholder as it is not used

                class_step, reg_step, state = self.model.step((context_tokens_t, beat_units_t))

            for col, i in TOKEN_COLS.items():
                if col in ('instrument', 'grid_factor', 'bpm', 'time_signature', 'beat_unit'):
                    if(col == 'beat_unit'):
                        pass
                    out_steps[t][i], probs = _categorical_sample(class_step[col], temperature, top_k, top_p)
                else:
                    out_steps[t][i] = self._quantize_regression_values(reg_step[col], column=col, return_encoding=True, ticks_per_gu=dna_meta["TicksPerGridUnit"])

        return out_steps


    def generate_dna_meta_base(self, dna_context: dict, variation: int) -> Dict[str, Any]:
        # TODO: check for invalid values
        dna_meta = {
            "DNA_ID": f"pred_{dna_context.get('DNA_ID', f'no_id')}_variation_{variation}",
            "DNASet": dna_context.get("DNASet", "predictions"),
            "DNAType": dna_context.get("DNAType", None),
            "AuthorData": dna_context.get("AuthorData", None),
            "StyleTags": dna_context.get("StyleTags", None),
            "OriginalMidiFileReference": dna_context.get("OriginalMidiFileReference", None),
            "AudioFileReference": dna_context.get("AudioFileReference", None),
            "IsTimeBased": dna_context.get("IsTimeBased", False),
            "Bpm": dna_context.get("Bpm", 120),
            "BpmRangeStart": dna_context.get("BpmRangeStart", None),
            "BpmRangeEnd": dna_context.get("BpmRangeEnd", None),
            "GridResolution": dna_context.get("GridResolution", None),
            "UnitLengthMs": dna_context.get("UnitLengthMs", None),
            "BasicKey": dna_context.get("BasicKey", None),
            "BasicScale": dna_context.get("BasicScale", None),
            "Numerator": dna_context.get("Numerator", 4),
            "Denominator": dna_context.get("Denominator", 4),
            "GridFactor": dna_context.get("GridFactor", 4),
            "IsTernary": dna_context.get("IsTernary", dna_context["GridFactor"] % 3 == 0),
            "TicksPerQuarterNote": dna_context.get("TicksPerQuarterNote", 480),
            "TicksPerGridUnit": dna_context.get("TicksPerGridUnit", 120),
            "Description": dna_context.get("Description", None),
            "FillStartTicks": dna_context.get("FillStartTicks", None),
        }
        return dna_meta

    @torch.no_grad()
    def sample(
            self,
            dna_context: dict,
            temperature: float = 1.0,
            top_k: Optional[int] = None,
            top_p: Optional[float] = None,
            max_tokens: Optional[int] = None,
            max_bars: Optional[int] = 4,
            num_variations: int = 1,
    ) -> List[Dict[str, Any]]:

        tokens, beat_positions = self.tokenizer.tokenize(dna_context, **self.tok_kwargs)  # (T,7), (T,)
        T = tokens.size(0)
        ctx_len = min(T, self.cfg.model.context_length)

        base_tokens_ctx = tokens[-ctx_len:].unsqueeze(0).to(self.device)  # (1, C, 7)
        base_beat_ctx = beat_positions[-ctx_len:].unsqueeze(0).to(self.device)  # (1, C)

        variations = []
        for variation in range(num_variations):

            # Meta defaults
            dna = self.generate_dna_meta_base(dna_context, variation)

            tokens = self.generate_autoregressive(
                base_tokens_ctx,
                base_beat_ctx,
                dna,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                max_tokens=max_tokens)

            # === Build DNAUnits from pred_tokens ===

            units_per_bar = dna["Numerator"] * dna["GridFactor"]

            dna_units = []
            if self.absolute_grid_units:
                dna_units, bar_count = self._build_dna_units_absolute(
                    tokens, units_per_bar, max_bars
                )
            else:
                dna_units, bar_count = self._build_dna_units_relative(
                    tokens, units_per_bar, dna["TicksPerGridUnit"], max_bars
                )

            dna["DNAUnits"] = dna_units
            dna["NumberOfBars"] = bar_count

            variations.append(dna)

        return variations


    # -----------------------------
    # DNA construction helpers
    # -----------------------------
    @staticmethod
    def _init_empty_unit() -> Dict[str, Any]:
        return {
            "Value": 0,  # leave for caller to compute via compose_value_fn if desired
            "OffsetTicksPerValuePart": {},  # you can later quantize offsets and fill this
            "VelocityPerValuePart": {},  # you can later quantize velocities and fill this
            "IsEmpty": True,

            'AvgOffsetTicks': 0.0,
            'AvgVelocity': 0.0,

            'ExcludeValue': None,
            'Wildcard': None,
            'TransposeInstruction': None,
            'ContinuingLastUnit': False
        }

    def _build_dna_units_absolute(
            self,
            pred_tokens: torch.Tensor,
            units_per_bar: int,
            max_bars: Optional[int] = None
    ) -> Tuple[List[Dict[str, Any]], int]:
        """
        For absolute grid-unit encoding, BeatUnit is the absolute index. We place
        token data directly at that position.
        """

        # Filter invalid tokens
        col = TOKEN_COLS["beat_unit"]
        diffs = torch.diff(pred_tokens[:, col], prepend=pred_tokens[:1, col])  # differences from prev
        mask = torch.abs(diffs) < 4 # Jumps more than x beat units
        pred_tokens = pred_tokens[mask]

        # Determine size
        max_pos = torch.max(pred_tokens[TOKEN_COLS["beat_unit"]])
        bar_count = math.ceil(max_pos / units_per_bar)

        dna_units: List[Dict[str, Any]] = [self._init_empty_unit() for _ in range(max_pos)]

        # TODO

        return dna_units, bar_count

    def _build_dna_units_relative(
            self,
            pred_tokens: torch.Tensor,
            units_per_bar: int,
            ticks_per_grid_unit: int,
            max_bars: Optional[int] = None
    ) -> Tuple[List[Dict[str, Any]], int]:
        """
        For relative grid-unit encoding, we detect bar starts (BeatUnit==1),
        slice bars, and place tokens within each bar.
        """

        # Filter invalid tokens
        mask = pred_tokens[:, TOKEN_COLS["beat_unit"]] < units_per_bar
        pred_tokens = pred_tokens[mask]

        # Clean output
        # ...

        # Find indices that start bars
        bar_starts: List[int] = []
        prev_were_one = False
        for t in range(pred_tokens.size(0)):
            if int(pred_tokens[t][TOKEN_COLS["beat_unit"]]) == 1 and not prev_were_one:
                bar_starts.append(t)
                prev_were_one = True
            elif prev_were_one and int(pred_tokens[t][TOKEN_COLS["beat_unit"]]) != 1:
                prev_were_one = False

        # Slice into bars
        bars: List[torch.Tensor] = []
        for i, s in enumerate(bar_starts):
            e = bar_starts[i + 1] if i + 1 < len(bar_starts) else len(pred_tokens)
            if e > s:
                bars.append(pred_tokens[s:e])

        bar_count = len(bars)
        dna_units: List[Dict[str, Any]] = []

        # Cut trailing bars
        if max_bars:
            bars = bars[:max_bars]

        for bar in bars:
            bar_dna = [self._init_empty_unit() for _ in range(units_per_bar)]
            for t in bar:
                instrument = t[TOKEN_COLS["instrument"]].item()
                instrument = decode_instrument(instrument)

                beat_unit = t[TOKEN_COLS["beat_unit"]].item()
                beat_unit = decode_beat_unit(beat_unit)

                offset = t[TOKEN_COLS["offset"]].item()
                offset = decode_offset_ticks(offset, ticks_per_grid_unit)

                velocity = t[TOKEN_COLS["velocity"]].item()
                velocity = decode_velocity(velocity)

                unit = bar_dna[beat_unit]

                unit["Value"] += instrument
                key = str(instrument)
                unit["OffsetTicksPerValuePart"][key] = offset
                unit["AvgOffsetTicks"] += offset
                unit["VelocityPerValuePart"][key] = velocity
                unit["AvgVelocity"] += velocity
                unit["IsEmpty"] = False

            dna_units.extend(bar_dna)

        # calculate averages
        for unit in dna_units:
            count = len(unit["VelocityPerValuePart"])
            unit["AvgOffsetTicks"] /= max(count, 1)
            unit["AvgVelocity"] /= max(count, 1)

        return dna_units, bar_count
