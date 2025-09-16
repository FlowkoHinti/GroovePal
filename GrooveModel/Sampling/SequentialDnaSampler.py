import math
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
from GrooveModel.Utils.DNAOffset import quantize_offset, decode_offset_ticks
from GrooveModel.Utils.DNAValue import decode_instrument
from GrooveModel.Utils.DNAVelocity import quantize_velocity, decode_velocity


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
            context_tokens: torch.Tensor,  # shape: [1, C, F]
            beat_units: torch.Tensor,  # shape: [1, C]
            units_per_bar: int,
            absolute_grid_units: bool,
            temperature: float = 1.0,
            top_k: Optional[int] = None,
            top_p: Optional[float] = None,
            max_tokens: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Autoregressively generate new tokens conditioned on the context.
        Returns a tensor of shape [C + max_tokens] (long).
        """
        ctx_len = int(context_tokens.size(1))
        new_tokens = int(max_tokens if max_tokens is not None else (MAX_SEQUENCE_LENGTH // 2))

        out_steps = torch.zeros((ctx_len + new_tokens), dtype=torch.long, device=self.device)
        out_beat_units = torch.zeros(new_tokens, dtype=torch.long, device=self.device)
        out_beat_units[0] = beat_units[:,-1]

        state: Optional[Dict[str, Dict[str, Tuple[torch.Tensor, ...]]]] = None
        for t in range(ctx_len + new_tokens):
            if t < ctx_len - 1:
                # Teacher forcing on the provided context
                context_tokens_t = context_tokens[:, t: t + 1]  # [1,1,F]
                beat_units_t = beat_units[:, t: t + 1]  # [1,1]
                class_step, state = self.model.step((context_tokens_t, beat_units_t), state=state)
                # also copy the ground truth token into the buffer so we can reference it if needed
                out_steps[t] = context_tokens[0, t]  # keep context tokens intact
            else:
                # Use previous generated token
                bu_id = (t - ctx_len) + 1
                prev = out_steps[t - 1].unsqueeze(0).unsqueeze(1)  # [1,1]
                beat_units_t = out_beat_units[bu_id]
                class_step, state = self.model.step((prev, beat_units_t), state=state)

                # Sample / quantize each column
                sample, _ = _categorical_sample(class_step, temperature, top_k, top_p)
                out_steps[t] = int(sample)

                # Setup next beat unit -> when token = SEP
                next_bu = beat_units_t.item() + 1 if sample == self.tokenizer.vocab.ID_SEP else beat_units_t.item()
                if absolute_grid_units:
                    out_beat_units[bu_id + 1] = next_bu
                else:
                    out_beat_units[bu_id + 1] = next_bu % units_per_bar

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
            "DNA_ID": f"pred_{dna_id}_variation_{variation}",
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
    ) -> List[Dict[str, Any]]:
        """
        Tokenize the context, generate continuations, and build DNAUnits payload(s).
        """
        tokens_ctx, beat_positions = self.tokenizer.tokenize(dna_context, **self.tok_kwargs)  # (T,7), (T,)
        T = int(tokens_ctx.size(0))
        ctx_len = min(T, int(self.cfg.model.context_length))

        base_tokens_ctx = tokens_ctx[-ctx_len:].unsqueeze(0).to(self.device)  # (1, C)
        base_beat_ctx = beat_positions[-ctx_len:].unsqueeze(0).to(self.device)  # (1, C)

        variations: List[Dict[str, Any]] = []
        for variation in range(num_variations):
            dna = self.generate_dna_meta_base(dna_context, variation)
            units_per_bar = int(dna["Numerator"]) * int(dna["GridFactor"])

            pred_tokens = self.generate_autoregressive(
                base_tokens_ctx,
                base_beat_ctx,
                dna,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                max_tokens=max_tokens,
            )

            # === Build DNAUnits from predictions ===

            dna_units, bar_count = self._build_dna_units()


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

    def _decode_token_components(
            self,
            token: torch.Tensor,
            ticks_per_grid_unit: int,
            absolute: bool,
    ) -> Optional[Tuple[int, int, int, float]]:
        """
        Decode a single token row into (instrument, beat_unit, offset_ticks, velocity).
        Returns None for rests / empty instruments.
        """
        instrument_id = int(token[TOKEN_COLS["instrument"]].item())
        instrument = decode_instrument(instrument_id)
        if instrument == 0:
            return None

        beat_unit_raw = int(token[TOKEN_COLS["beat_unit"]].item())
        beat_unit = int(decode_beat_unit(beat_unit_raw, absolute=absolute))

        offset_enc = int(token[TOKEN_COLS["offset"]].item())
        offset = int(decode_offset_ticks(offset_enc, ticks_per_grid_unit, start_at_zero=True))

        velocity_enc = int(token[TOKEN_COLS["velocity"]].item())
        velocity = float(decode_velocity(velocity_enc))

        return instrument, beat_unit, offset, velocity

    @staticmethod
    def _apply_event_to_unit(
            unit: Dict[str, Any],
            instrument: int,
            offset_ticks: int,
            velocity: float,
    ) -> None:
        """Mutate a unit dict with a single (instrument, offset, velocity) event."""
        unit["Value"] |= instrument
        key = str(instrument)
        unit["OffsetTicksPerValuePart"][key] = offset_ticks
        unit["VelocityPerValuePart"][key] = velocity
        unit["IsEmpty"] = False

    @staticmethod
    def _finalize_units_average(units: List[Dict[str, Any]]) -> None:
        """Compute AvgVelocity / AvgOffsetTicks once per unit."""
        for unit in units:
            velocities = list(unit["VelocityPerValuePart"].values())
            offsets = list(unit["OffsetTicksPerValuePart"].values())
            n = len(velocities)
            unit["AvgVelocity"] = (sum(velocities) / n) if n else 0.0
            unit["AvgOffsetTicks"] = int(round(sum(offsets) / n)) if n else 0

    def _build_dna_units_absolute(
            self,
            pred_tokens: torch.Tensor,  # [N, F]
            units_per_bar: int,
            ticks_per_grid_unit: int,
    ) -> Tuple[List[Dict[str, Any]], int]:
        """
        For absolute grid-unit encoding, BeatUnit is the absolute index (0-based).
        We place token data directly at that position.
        """

        max_pos = int(torch.max(pred_tokens[:, TOKEN_COLS["beat_unit"]]).item())
        size = max_pos + 1  # include last position
        bar_count = math.ceil(size / units_per_bar)

        dna_units: List[Dict[str, Any]] = [self._init_empty_unit() for _ in range(size)]

        for token in pred_tokens:
            decoded = self._decode_token_components(token, ticks_per_grid_unit, absolute=True)
            if decoded is None:
                continue
            instrument, beat_unit, offset, velocity = decoded
            if 0 <= beat_unit < size:
                self._apply_event_to_unit(dna_units[beat_unit], instrument, offset, velocity)

        self._finalize_units_average(dna_units)
        return dna_units, bar_count

    def _build_dna_units_relative(
            self,
            pred_tokens: torch.Tensor,  # [N, F]
            units_per_bar: int,
            ticks_per_grid_unit: int,
    ) -> Tuple[List[Dict[str, Any]], int]:
        """
        For relative grid-unit encoding, we detect bar starts (BeatUnit==1),
        slice bars, and place tokens within each bar.
        """


        # Detect non-repeating bar starts at BeatUnit == 1
        bu_col = TOKEN_COLS["beat_unit"]
        bar_starts: List[int] = []
        prev_is_one = False
        for idx in range(pred_tokens.size(0)):
            is_one = int(pred_tokens[idx, bu_col].item()) == 1
            if is_one and not prev_is_one:
                bar_starts.append(idx)
            prev_is_one = is_one

        # Slice into bars
        bars: List[torch.Tensor] = []
        for i, s in enumerate(bar_starts):
            e = bar_starts[i + 1] if i + 1 < len(bar_starts) else pred_tokens.size(0)
            if e > s:
                bars.append(pred_tokens[s:e])

        if max_bars:
            bars = bars[:max_bars]

        bar_count = len(bars)
        dna_units: List[Dict[str, Any]] = []

        for bar in bars:
            bar_units = [self._init_empty_unit() for _ in range(units_per_bar)]
            for token in bar:
                decoded = self._decode_token_components(token, ticks_per_grid_unit, absolute=False)
                if decoded is None:
                    continue
                instrument, beat_unit, offset, velocity = decoded
                if 0 <= beat_unit < units_per_bar:
                    self._apply_event_to_unit(bar_units[beat_unit], instrument, offset, velocity)
            dna_units.extend(bar_units)

        self._finalize_units_average(dna_units)
        return dna_units, bar_count
