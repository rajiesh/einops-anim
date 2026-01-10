"""
Einops repeat operation animation using Manim CE.

Demonstrates how einops.repeat duplicates tensor dimensions.

Examples:
    repeat(x, 'h w -> h w c', c=3)  # add new axis with repetition
    repeat(x, 'h w -> (repeat h) w', repeat=2)  # repeat along existing axis
    repeat(x, 'h w -> h w 1')  # add singleton dimension
"""

from __future__ import annotations
from typing import Dict, List, Tuple
import numpy as np
from manim import *
from einops_manim_lib import *


# =========================
# USER INPUT (edit only these)
# =========================
TENSOR_NAME = "x"
PATTERN = "h w -> h w c"  # Input pattern -> Output pattern
REPEAT_PARAMS = {"c": 3}  # Parameters for new/repeated axes

# =========================
# SETTINGS
# =========================
RNG_SEED = 42
AXIS_SIZE_CHOICES = (2, 3)


class RepeatAnim(Scene):
    def construct(self):
        # Parse pattern
        if "->" not in PATTERN:
            raise ValueError("Pattern must contain '->'")

        input_spec, output_spec = PATTERN.split("->")
        input_axes = parse_axes_spec(input_spec)
        output_axes = self._parse_output_axes(output_spec)

        # Assign sizes for input axes
        axis_sizes = assign_minimal_axis_sizes(input_axes, choices=AXIS_SIZE_CHOICES, seed=RNG_SEED)

        # Add repeat parameters
        for ax, size in REPEAT_PARAMS.items():
            axis_sizes[ax] = size

        # Create input tensor
        input_shape = tuple(axis_sizes[ax] for ax in input_axes)
        rng = np.random.RandomState(RNG_SEED)
        input_tensor = rng.randint(0, 10, size=input_shape, dtype=int)

        # Compute output using einops
        try:
            from einops import repeat
            output_tensor = repeat(input_tensor, PATTERN, **REPEAT_PARAMS)
        except Exception as e:
            # Fallback: compute manually
            output_tensor = self._manual_repeat(input_tensor, input_axes, output_axes,
                                               axis_sizes, REPEAT_PARAMS)

        # Header
        params_str = ", ".join(f"{k}={v}" for k, v in REPEAT_PARAMS.items())
        title = safe_text(
            f'einops.repeat({TENSOR_NAME}, "{PATTERN}", {params_str})',
            font_size=24
        ).to_edge(UP)
        shape_info = safe_text(
            f"shape: {input_shape} -> {output_tensor.shape}",
            font_size=20
        ).next_to(title, DOWN, buff=0.12)
        repeat_info = safe_text(
            f"repeating: {params_str}",
            font_size=18
        ).next_to(shape_info, DOWN, buff=0.10)

        self.play(Write(title), run_time=1.0)
        self.play(FadeIn(shape_info, shift=0.1 * DOWN), run_time=0.8)
        self.play(FadeIn(repeat_info, shift=0.1 * DOWN), run_time=0.8)
        self.wait(1.0)

        # Display input tensor
        input_display, input_type, input_grid = display_tensor(
            input_tensor, input_axes, "input", font_size_label=20
        )
        input_display.scale(0.75).next_to(repeat_info, DOWN, buff=0.4).to_edge(LEFT)

        self.play(FadeIn(input_display, shift=0.15 * DOWN), run_time=1.2)
        self.wait(1.0)

        # Display output tensor
        output_display_axes = self._flatten_output_axes(output_axes)
        output_display, output_type, output_grid = display_tensor(
            output_tensor, output_display_axes, "output", font_size_label=20
        )
        output_display.scale(0.7).to_edge(RIGHT).align_to(input_display, UP)

        self.play(FadeIn(output_display, shift=0.15 * DOWN), run_time=1.2)
        self.wait(1.0)

        # Animate repetition: show how input elements are copied to output
        self._animate_repetition(input_tensor, output_tensor, input_grid, output_grid,
                                input_axes, output_display_axes,
                                input_type, output_type, axis_sizes)

        # Done
        done = safe_text("Repeat complete!", font_size=18).to_edge(DOWN)
        self.play(FadeIn(done, shift=0.1 * UP), run_time=0.8)
        self.wait(2.0)

    def _parse_output_axes(self, output_spec: str) -> List[str]:
        """Parse output axes, preserving parentheses info."""
        tokens = parse_axes_spec(output_spec)
        return tokens

    def _flatten_output_axes(self, output_axes: List[str]) -> List[str]:
        """Flatten axes for display."""
        flattened = []
        for ax in output_axes:
            if "(" in ax or ")" in ax:
                # Remove parentheses
                clean = ax.replace("(", "").replace(")", "")
                for a in clean.split():
                    if a:
                        flattened.append(a)
            else:
                flattened.append(ax)
        return flattened

    def _manual_repeat(self, tensor: np.ndarray, input_axes: List[str],
                      output_axes: List[str], axis_sizes: Dict[str, int],
                      repeat_params: Dict[str, int]) -> np.ndarray:
        """Manual repeat implementation."""
        # Find new axes and compute output shape
        output_flat = self._flatten_output_axes(output_axes)
        new_axes = [ax for ax in output_flat if ax not in input_axes]

        # For simple case: add new axes at the end
        output_shape = list(tensor.shape)
        for ax in new_axes:
            if ax in repeat_params:
                output_shape.append(repeat_params[ax])

        # Repeat the tensor along new axes
        result = tensor.copy()
        for ax in new_axes:
            if ax in repeat_params:
                result = np.expand_dims(result, axis=-1)
                result = np.repeat(result, repeat_params[ax], axis=-1)

        return result.reshape(output_shape)

    def _animate_repetition(self, input_tensor: np.ndarray, output_tensor: np.ndarray,
                           input_grid: VGroup, output_grid: VGroup,
                           input_axes: List[str], output_axes: List[str],
                           input_type: str, output_type: str, axis_sizes: Dict[str, int]):
        """Animate how input elements are repeated to output."""
        # Work area
        work_anchor = safe_text(".", font_size=18).move_to(ORIGIN)
        work_anchor.set_opacity(0)
        self.add(work_anchor)

        # Sample elements to show
        num_input_samples = min(4, input_tensor.size)
        input_indices_list = list(np.ndindex(input_tensor.shape))

        for sample_idx in range(num_input_samples):
            # Get input element
            in_idx = input_indices_list[(sample_idx * len(input_indices_list)) // num_input_samples]
            value = input_tensor[in_idx]

            # Highlight input cell
            input_hl = self._highlight_by_type(input_grid, in_idx, input_type, YELLOW)
            if input_hl:
                self.play(Create(input_hl), run_time=0.5)

            # Show input index
            in_idx_str = ", ".join(f"{ax}={in_idx[i]}" for i, ax in enumerate(input_axes))
            header = safe_text(f"input[{in_idx_str}] = {value}", font_size=20).move_to(ORIGIN)
            self.play(FadeIn(header, shift=0.1 * UP), run_time=0.6)

            # Find all output positions that contain this value
            # Due to repetition, this value appears in multiple output locations
            output_positions = np.where(output_tensor == value)
            num_copies = len(output_positions[0])

            info_text = safe_text(
                f"repeated to {min(num_copies, 3)} output positions",
                font_size=16
            ).next_to(header, DOWN, buff=0.15)
            self.play(FadeIn(info_text, shift=0.05 * UP), run_time=0.6)
            self.wait(1.0)

            # Show a few output positions
            output_highlights = VGroup()
            for copy_idx in range(min(3, num_copies)):
                out_idx = tuple(pos[copy_idx] for pos in output_positions)
                output_hl = self._highlight_by_type(output_grid, out_idx, output_type, GREEN)
                if output_hl:
                    output_highlights.add(output_hl)

            if len(output_highlights) > 0:
                self.play(Create(output_highlights), run_time=0.5)
                self.wait(1.5)

            # Cleanup
            self.play(
                FadeOut(header),
                FadeOut(info_text),
                FadeOut(input_hl) if input_hl else Wait(0),
                FadeOut(output_highlights) if len(output_highlights) > 0 else Wait(0),
                run_time=0.6
            )
            self.wait(0.3)

    def _highlight_by_type(self, grid: VGroup, indices: Tuple, display_type: str, color):
        """Highlight a cell based on display type."""
        try:
            if display_type == "4d" and len(indices) == 4:
                i0, i1, i2, i3 = indices
                return highlight_cell_4d(grid, i0, i1, i2, i3, color=color)
            elif display_type == "3d" and len(indices) == 3:
                d, r, c = indices
                return highlight_cell_3d(grid, d, r, c, color=color)
            elif display_type == "2d" and len(indices) >= 1:
                if len(indices) == 1:
                    return highlight_cell(grid, 0, indices[0], color=color)
                elif len(indices) >= 2:
                    return highlight_cell(grid, indices[0], indices[1], color=color)
        except (IndexError, KeyError):
            return None
        return None
