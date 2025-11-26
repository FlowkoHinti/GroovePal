import math
from doctest import UnexpectedException
from enum import IntEnum, auto
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union, Literal

import torch
from dacite import from_dict, Config as DaciteConfig
from omegaconf import DictConfig, OmegaConf

from Configs import MAX_SEQUENCE_LENGTH
from GrooveModel.Embedding.SequentialDnaEmbedding import SequentialDNAEmbeddingConfig
from GrooveModel.Models import ModelConfigxLstm, SequentialDNAxLSTM
from GrooveModel.Sampling.Sampler import (
    DNATokenSampler,
    _load_cfg,
    _select_checkpoint_path,
    _categorical_sample,
)
from GrooveModel.Tokenizer.MultiTaskDnaTokenizer import MultiTaskDnaTokenizer
from GrooveModel.Tokenizer.SequentialDnaTokenizer import SequentialDnaTokenizer
from GrooveModel.Utils.BeatUnit import decode_beat_unit
from GrooveModel.Utils.DNAOffset import quantize_offset, decode_offset_ticks, percent_step_to_offset
from GrooveModel.Utils.DNAValue import decode_instrument, InstrumentValues
from GrooveModel.Utils.DNAVelocity import quantize_velocity, decode_velocity


class Tokentype(IntEnum):
    Start = 0
    Instrument = auto()
    Velocity = auto()
    Offset = auto()
    Control = auto()

class SequentialDNASampler(DNATokenSampler):

    def __init__(
            self,
            cfg_or_path: Union[str, Path, DictConfig],
            checkpoint: Optional[Union[str, Path]] = None,
            device: Optional[torch.device] = None,
            use_best: bool = True,
    ):
        # Config + device
        self.cfg = _load_cfg(cfg_or_path, seq=True)
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Tokenizer (propagate tokenizer kwargs so flags like absolute_grid_units are honored)
        self.tok_kwargs: Dict[str, Any] = dict(getattr(self.cfg.dataset, "tokenizer", {}))
        self.absolute_grid_units: bool = bool(self.tok_kwargs.get("absolute_grid_units", False))
        self.tokenizer = SequentialDnaTokenizer()

        # Build model
        self.embedding_config = from_dict(
            SequentialDNAEmbeddingConfig,
            OmegaConf.to_container(self.cfg.embedding, resolve=True),
            config=DaciteConfig(strict=True),
        )
        vocab_size = len(SequentialDnaTokenizer.vocab)
        self.embedding_config.vocab_size = vocab_size

        model_config = from_dict(
            ModelConfigxLstm,
            OmegaConf.to_container(self.cfg.model, resolve=True),
            config=DaciteConfig(strict=True),
        )
        self.model: SequentialDNAxLSTM = (
            SequentialDNAxLSTM(model_config, self.embedding_config, self.absolute_grid_units).to(
                self.device).eval()
        )

        # Checkpoint
        ckpt_path = _select_checkpoint_path(self.cfg, checkpoint, use_best)
        state = torch.load(str(ckpt_path), map_location="cpu")
        self.model.load_state_dict(state["model_state_dict"], strict=True)

    @torch.no_grad()
    def generate_autoregressive(
            self,
            context_tokens: torch.Tensor,  # should be [1, C] if these are IDs
            beat_units: torch.Tensor,  # shape: [1, C]
            units_per_bar: int,
            temperature: float = 1.0,
            top_k: Optional[int] = None,
            top_p: Optional[float] = None,
            max_tokens: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Autoregressively generate new tokens conditioned on the context.
        Returns a tensor of shape [C + max_tokens] (long).
        """

        # Get the Rest Id for correct Rest handling
        v = self.tokenizer.vocab
        rest_id = v.instrument_id_from_value(InstrumentValues.Rest)

        ctx_len = int(context_tokens.size(1))
        new_tokens = int(max_tokens if max_tokens is not None else (MAX_SEQUENCE_LENGTH // 2))

        # Allocate token output buffer
        out_steps = torch.zeros((ctx_len + new_tokens), dtype=torch.long, device=self.device)

        # Beat-unit buffer needs to be longer than new_tokens because we
        # always write "the next" unit after using the current one.
        out_beat_units = torch.zeros(new_tokens + 2, dtype=torch.long, device=self.device)
        out_beat_units[0] = beat_units[0, -1]  # store last context beat unit as starting point

        state: Optional[Dict[str, Dict[str, Tuple[torch.Tensor, ...]]]] = None

        for t in range(ctx_len + new_tokens):
            if t < ctx_len - 1:
                # --- Teacher forcing branch (use provided context) ---
                context_tokens_t = context_tokens[:, t: t + 1]  # [1,1]
                beat_units_t = beat_units[:, t: t + 1]  # [1,1]
                class_step, state = self.model.step((context_tokens_t, beat_units_t), state=state, inference=True)

                # Copy the ground-truth token ID into the buffer
                out_steps[t] = context_tokens[0, t]
            else:
                # --- Generation branch (use previous outputs) ---
                bu_id = (t - ctx_len) + 1

                # Previous generated token -> shape [1,1]
                prev = out_steps[t - 1].view(1, 1)

                # Current beat unit -> also [1,1]
                beat_units_t = out_beat_units[bu_id].view(1, 1)

                # Step the model forward
                class_step, state = self.model.step((prev, beat_units_t), state=state)

                # Sample from distribution
                sample, _ = _categorical_sample(class_step, temperature, top_k, top_p)
                out_steps[t] = int(sample)

                # Compute next beat unit
                if sample == v.ID_SEP or sample == rest_id:
                    next_bu = beat_units_t.item() + 1
                else:
                    next_bu = beat_units_t.item()

                # Wrap around if relative grid is used
                if not self.absolute_grid_units:
                    next_bu = next_bu % units_per_bar

                # Only write the next slot if it’s inside bounds
                if bu_id + 1 < out_beat_units.numel():
                    out_beat_units[bu_id + 1] = next_bu

                # Stop early on EOS
                if sample == self.tokenizer.vocab.ID_EOS:
                    break

        return out_steps

    def generate_dna_meta_base(self, dna_context: Dict[str, Any], variation: int) -> Dict[str, Any]:
        """
        Build a robust metadata dict with safe fallbacks (avoids KeyErrors).
        """
        dna_id = dna_context.get("DNA_ID") or "no_id"
        bpm = int(dna_context.get("Bpm", 120))
        grid_factor = int(dna_context.get("GridFactor", 4))

        unit_length_ms = 60000 / (bpm * grid_factor)

        # Map GridFactor to "music resolution"
        match grid_factor:
            case 4:
                music_resolution = 0
            case 2:
                music_resolution = 1
            case 3:
                music_resolution = 3
            case 6:
                music_resolution = 4
            case _:
                music_resolution = None

        is_ternary_default = (grid_factor % 3 == 0)

        dna_meta: Dict[str, Any] = {
            "DNA_ID": f"{dna_id}_variation_{variation}",
            "DNASet": dna_context.get("DNASet", "predictions"),
            "DNAType": dna_context.get("DNAType", 0),  # DnaType.Drumbeats = 0
            "AuthorData": dna_context.get("AuthorData", "groovepal"),
            "StyleTags": dna_context.get("StyleTags"),
            "OriginalMidiFileReference": dna_context.get("OriginalMidiFileReference"),
            "AudioFileReference": dna_context.get("AudioFileReference"),
            "IsTimeBased": dna_context.get("IsTimeBased", False),
            "Bpm": bpm,
            "BpmRangeStart": dna_context.get("BpmRangeStart", bpm * 0.8),
            "BpmRangeEnd": dna_context.get("BpmRangeEnd", bpm * 1.2),
            "GridResolution": dna_context.get("GridResolution", music_resolution),
            "UnitLengthMs": dna_context.get("UnitLengthMs", unit_length_ms),
            "BasicKey": dna_context.get("BasicKey"),
            "BasicScale": dna_context.get("BasicScale"),
            "Numerator": int(dna_context.get("Numerator", 4)),
            "Denominator": int(dna_context.get("Denominator", 4)),
            "GridFactor": grid_factor,
            "IsTernary": dna_context.get("IsTernary", is_ternary_default),
            "TicksPerQuarterNote": int(dna_context.get("TicksPerQuarterNote", 480)),
            "TicksPerGridUnit": int(dna_context.get("TicksPerGridUnit", 120)),
            "Description": dna_context.get("Description"),
            "FillStartTicks": dna_context.get("FillStartTicks"),
            "MusicEvents": [],
        }
        return dna_meta

    @torch.no_grad()
    def sample(
            self,
            dna_context: Dict[str, Any],
            temperature: float = 1.0,
            top_k: Optional[int] = None,
            top_p: Optional[float] = None,
            max_tokens: Optional[int] = None,
            max_bars: Optional[int] = 4,
            num_variations: int = 1,
    ) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
        """
        Tokenize the context, generate continuations, and build DNAUnits payload(s).

        Returns:
            base_dna: The original (preprocessed) DNA
            variations: List of generated DNAs (each with generated DNAUnits only)
        """
        # --- Tokenize the input context
        tokens_ctx, beat_positions = self.tokenizer.tokenize(dna_context, **self.tok_kwargs)  # (T,), (T,)
        T = int(tokens_ctx.size(0))
        ctx_len = min(T, int(self.cfg.model.context_length))

        base_tokens_ctx = tokens_ctx[-ctx_len:].unsqueeze(0).to(self.device)  # (1, C)
        base_beat_ctx = beat_positions[-ctx_len:].unsqueeze(0).to(self.device)  # (1, C)

        # --- Build the base/original DNA
        base_dna = dna_context
        units_per_bar = int(base_dna["Numerator"]) * int(base_dna["GridFactor"])
        ticks_per_grid_unit = int(base_dna["TicksPerGridUnit"])

        # Build from original (non-generated) tokens
        base_units, base_bar_count = self._build_dna_units(
            tokens_ctx, units_per_bar, ticks_per_grid_unit
        )
        base_dna["DNAUnits"] = base_units
        base_dna["NumberOfBars"] = base_bar_count

        # --- Generate variations
        variations: List[Dict[str, Any]] = []
        for variation in range(num_variations):
            dna = self.generate_dna_meta_base(dna_context, variation)
            units_per_bar = int(dna["Numerator"]) * int(dna["GridFactor"])
            ticks_per_grid_unit = int(dna["TicksPerGridUnit"])

            # Generate continuation
            pred_tokens = self.generate_autoregressive(
                base_tokens_ctx,
                base_beat_ctx,
                units_per_bar,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                max_tokens=max_tokens,
            )

            # Slice off context portion (keep only generated tokens)
            ctx_len = int(base_tokens_ctx.size(1))
            gen_tokens = pred_tokens[ctx_len:]

            # --- Build DNAUnits only from generated tokens
            dna_units, bar_count = self._build_dna_units(
                gen_tokens, units_per_bar, ticks_per_grid_unit
            )

            dna["DNAUnits"] = dna_units
            dna["NumberOfBars"] = bar_count
            variations.append(dna)

        return base_dna, variations

    # -----------------------------
    # DNA construction helpers
    # -----------------------------
    @staticmethod
    def _init_empty_unit() -> Dict[str, Any]:
        return {
            "Value": 0,  # the caller may compute via compose_value_fn if desired
            "OffsetTicksPerValuePart": {},
            "VelocityPerValuePart": {},
            "AvgOffsetTicks": 0,
            "AvgVelocity": 0.0,
            "ExcludeValue": None,
            "Wildcard": False,
            "TransposeInstruction": 0,
            "IsEmpty": True,
            "ContinuingLastUnit": False,
        }

    @staticmethod
    def _finalize_units_average(units: List[Dict[str, Any]]) -> None:
        """Compute AvgVelocity / AvgOffsetTicks once per unit."""
        for unit in units:
            velocities = list(unit["VelocityPerValuePart"].values())
            offsets = list(unit["OffsetTicksPerValuePart"].values())
            n = len(velocities)
            unit["AvgVelocity"] = (sum(velocities) / n) if n else 0.0
            unit["AvgOffsetTicks"] = int(round(sum(offsets) / n)) if n else 0

    def _build_dna_units(self, pred_tokens: torch.Tensor, units_per_bar: int, ticks_per_gridunit: int) -> Tuple[List[Dict[str, Any]], int]:

        vocab = self.tokenizer.vocab

        units: List[Dict[str, Any]] = [self._init_empty_unit()]
        unit_id: int = 0
        last_token: Tokentype = Tokentype.Start
        key: str = ""
        initial_run = True


        for token_id in pred_tokens:
            if token_id in (vocab.ID_EOS, vocab['PAD']):
                break

            token = vocab.token(token_id)
            if initial_run:
                initial_run = False
                if token_id == vocab.ID_SEP:
                    # Skip leading SEPs
                    continue

            # Unit handling
            if token_id == vocab.ID_SEP:
                unit_id += 1
                units.append(self._init_empty_unit())
                key = ""
                last_token = Tokentype.Control

            elif token in vocab.INSTRUMENTS:
                if last_token in (Tokentype.Start, Tokentype.Control, Tokentype.Offset):
                    instrument = token
                    value = InstrumentValues[instrument]

                    if value == InstrumentValues.Rest:
                        # Treat REST as a full empty step: advance to a new unit.
                        unit_id += 1
                        units.append(self._init_empty_unit())
                        key = ""
                        last_token = Tokentype.Control
                        continue

                    units[unit_id]["Value"] += value
                    units[unit_id]["IsEmpty"] = False
                    key = str(value)
                    last_token = Tokentype.Instrument

            elif token in vocab.VELOCITIES:
                if last_token in (Tokentype.Instrument, Tokentype.Offset) and key != "":
                    velocity = int(token.split("_")[1])
                    velocity = decode_velocity(velocity, include_padding=False)
                    units[unit_id]["VelocityPerValuePart"][key] = velocity
                    last_token = Tokentype.Velocity

            elif token in vocab.OFFSETS:
                if last_token in (Tokentype.Instrument, Tokentype.Velocity) and key != "":
                    offset_step = int(token.split("_")[2])
                    offset = percent_step_to_offset(offset_step, ticks_per_gridunit)
                    units[unit_id]["OffsetTicksPerValuePart"][key] = offset
                    last_token = Tokentype.Offset

            # NO HANDLING FOR NOW -> WE TAKE THE ORIGINAL METADATA
            elif token in vocab.BPM:
                continue
            elif token in vocab.GRID_FACTORS:
                continue
            elif token in vocab.TIME_SIGNATURES:
                continue
            elif token in vocab.SPECIAL_TOKENS:
                continue
            else:
                raise ValueError(token)

        self._finalize_units_average(units)
        bar_count = math.ceil(len(units) / units_per_bar)

        return units, bar_count
