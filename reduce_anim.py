"""
Einops reduce operation animation using Manim CE.

Demonstrates how einops.reduce aggregates tensor dimensions using reduction operations.

Examples:
    reduce(x, 'h w c -> h w', 'mean')  # average over channels
    reduce(x, 'b h w c -> b c', 'max')  # max pool over spatial dims
    reduce(x, 'h w -> h', 'sum')  # sum over width
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
PATTERN = "h w c -> h w"  # Input pattern -> Output pattern
REDUCTION = "mean"  # Reduction operation: 'mean', 'sum', 'max', 'min', 'prod'

# =========================
# SETTINGS
# =========================
RNG_SEED = 42
AXIS_SIZE_CHOICES = (2, 3)


class ReduceAnim(Scene):
    def construct(self):
        # Parse pattern
        if "->" not in PATTERN:
            raise ValueError("Pattern must contain '->'")

        input_spec, output_spec = PATTERN.split("->")
        input_axes = parse_axes_spec(input_spec)
        output_axes = parse_axes_spec(output_spec)

        # Determine reduction axes
        reduction_axes = [ax for ax in input_axes if ax not in output_axes]

        # Assign sizes
        all_axes = input_axes.copy()
        axis_sizes = assign_minimal_axis_sizes(all_axes, choices=AXIS_SIZE_CHOICES, seed=RNG_SEED)

        # Create input tensor
        input_shape = tuple(axis_sizes[ax] for ax in input_axes)
        rng = np.random.RandomState(RNG_SEED)
        input_tensor = rng.randint(0, 10, size=input_shape, dtype=int)

        # Compute output using einops
        try:
            from einops import reduce as einops_reduce
            output_tensor = einops_reduce(input_tensor, PATTERN, REDUCTION)
        except Exception as e:
            # Fallback: compute manually
            output_tensor = self._manual_reduce(input_tensor, input_axes, output_axes,
                                               reduction_axes, axis_sizes, REDUCTION)

        # Header
        title = safe_text(
            f'einops.reduce({TENSOR_NAME}, "{PATTERN}", "{REDUCTION}")',
            font_size=26
        ).to_edge(UP)
        shape_info = safe_text(
            f"shape: {input_shape} -> {output_tensor.shape}",
            font_size=20
        ).next_to(title, DOWN, buff=0.12)
        reduce_info = safe_text(
            f"reducing over: {', '.join(reduction_axes) if reduction_axes else 'none'}",
            font_size=18
        ).next_to(shape_info, DOWN, buff=0.10)

        self.play(Write(title), run_time=1.0)
        self.play(FadeIn(shape_info, shift=0.1 * DOWN), run_time=0.8)
        self.play(FadeIn(reduce_info, shift=0.1 * DOWN), run_time=0.8)
        self.wait(1.0)

        # Display input tensor
        input_display, input_type, input_grid = display_tensor(
            input_tensor, input_axes, "input", font_size_label=20
        )
        input_display.scale(0.75).next_to(reduce_info, DOWN, buff=0.4).to_edge(LEFT)

        self.play(FadeIn(input_display, shift=0.15 * DOWN), run_time=1.2)
        self.wait(1.0)

        # Display output tensor (with placeholders)
        output_shape = tuple(axis_sizes[ax] for ax in output_axes)
        output_placeholder = np.zeros(output_shape, dtype=int)
        output_display, output_type, output_grid = display_tensor(
            output_placeholder, output_axes, "output", font_size_label=20
        )
        output_display.scale(0.75).to_edge(RIGHT).align_to(input_display, UP)

        self.play(FadeIn(output_display, shift=0.15 * DOWN), run_time=1.2)
        self.wait(1.0)

        # Animate reduction computation
        self._animate_reduction(input_tensor, output_tensor, input_grid, output_grid,
                               input_axes, output_axes, reduction_axes,
                               input_type, output_type, axis_sizes)

        # Done
        done = safe_text(f"Reduce ({REDUCTION}) complete!", font_size=18).to_edge(DOWN)
        self.play(FadeIn(done, shift=0.1 * UP), run_time=0.8)
        self.wait(2.0)

    def _manual_reduce(self, tensor: np.ndarray, input_axes: List[str],
                      output_axes: List[str], reduction_axes: List[str],
                      axis_sizes: Dict[str, int], reduction: str) -> np.ndarray:
        """Manual reduce implementation."""
        # Find axes to reduce
        reduce_axis_indices = [input_axes.index(ax) for ax in reduction_axes]

        if reduction == "mean":
            return np.mean(tensor, axis=tuple(reduce_axis_indices))
        elif reduction == "sum":
            return np.sum(tensor, axis=tuple(reduce_axis_indices))
        elif reduction == "max":
            return np.max(tensor, axis=tuple(reduce_axis_indices))
        elif reduction == "min":
            return np.min(tensor, axis=tuple(reduce_axis_indices))
        elif reduction == "prod":
            return np.prod(tensor, axis=tuple(reduce_axis_indices))
        else:
            raise ValueError(f"Unknown reduction: {reduction}")

    def _animate_reduction(self, input_tensor: np.ndarray, output_tensor: np.ndarray,
                          input_grid: VGroup, output_grid: VGroup,
                          input_axes: List[str], output_axes: List[str],
                          reduction_axes: List[str],
                          input_type: str, output_type: str, axis_sizes: Dict[str, int]):
        """Animate the reduction computation."""
        # Work area
        work_anchor = safe_text(".", font_size=18).move_to(ORIGIN)
        work_anchor.set_opacity(0)
        self.add(work_anchor)

        # Get text references for output cells
        output_text_refs = self._get_text_refs(output_grid, output_type, output_tensor.shape)

        # Iterate through output cells
        output_indices_list = list(np.ndindex(output_tensor.shape))
        num_to_show = min(6, len(output_indices_list))

        for sample_idx in range(num_to_show):
            out_idx = output_indices_list[(sample_idx * len(output_indices_list)) // num_to_show]

            # Build output index mapping
            out_idx_dict = {ax: out_idx[i] for i, ax in enumerate(output_axes)}

            # Highlight output cell
            output_hl = self._highlight_by_type(output_grid, out_idx, output_type, GREEN)
            if output_hl:
                self.play(Create(output_hl), run_time=0.6)

            # Show computation header
            out_idx_str = ", ".join(f"{ax}={val}" for ax, val in out_idx_dict.items())
            header = safe_text(f"compute output[{out_idx_str}]", font_size=20).move_to(ORIGIN)
            self.play(FadeIn(header, shift=0.1 * UP), run_time=0.6)

            # Collect values to reduce
            values_to_reduce = []
            highlights = VGroup()

            # Iterate over all combinations of reduction axes
            reduction_sizes = [axis_sizes[ax] for ax in reduction_axes]
            if reduction_sizes:
                for red_combo in list(np.ndindex(tuple(reduction_sizes)))[:5]:  # Show first 5
                    # Build full input index
                    input_idx = []
                    red_idx = 0
                    for ax in input_axes:
                        if ax in out_idx_dict:
                            input_idx.append(out_idx_dict[ax])
                        else:
                            input_idx.append(red_combo[red_idx])
                            red_idx += 1

                    val = input_tensor[tuple(input_idx)]
                    values_to_reduce.append(val)

                    # Highlight input cell
                    input_hl = self._highlight_by_type(input_grid, tuple(input_idx), input_type, YELLOW)
                    if input_hl:
                        highlights.add(input_hl)
            else:
                # No reduction, direct copy
                input_idx = [out_idx_dict[ax] for ax in input_axes]
                val = input_tensor[tuple(input_idx)]
                values_to_reduce.append(val)

            if len(highlights) > 0:
                self.play(Create(highlights), run_time=0.5)

            # Show reduction computation
            if len(values_to_reduce) > 0:
                values_str = ", ".join(str(v) for v in values_to_reduce[:5])
                if len(values_to_reduce) > 5:
                    values_str += ", ..."

                result_val = output_tensor[out_idx]

                comp_text = safe_text(
                    f"{REDUCTION}([{values_str}]) = {result_val:.2f}" if isinstance(result_val, (float, np.floating))
                    else f"{REDUCTION}([{values_str}]) = {result_val}",
                    font_size=16
                ).next_to(header, DOWN, buff=0.15)

                self.play(FadeIn(comp_text, shift=0.05 * UP), run_time=0.8)
                self.wait(1.5)

                # Update output cell
                self._update_output_cell(output_text_refs, out_idx, output_type, result_val)

                # Cleanup
                self.play(
                    FadeOut(comp_text),
                    FadeOut(highlights) if len(highlights) > 0 else Wait(0),
                    FadeOut(header),
                    FadeOut(output_hl) if output_hl else Wait(0),
                    run_time=0.6
                )
                self.wait(0.5)

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
                else:
                    return highlight_cell(grid, indices[0], indices[1], color=color)
        except (IndexError, KeyError):
            return None
        return None

    def _get_text_refs(self, grid: VGroup, display_type: str, shape: Tuple) -> dict:
        """Get references to text elements in the grid."""
        refs = {}
        if display_type == "2d":
            for r in range(shape[0] if len(shape) > 1 else 1):
                for c in range(shape[-1]):
                    refs[(r, c)] = grid[r][c][1]
        # Add support for 3d/4d if needed
        return refs

    def _update_output_cell(self, text_refs: dict, indices: Tuple, display_type: str, value):
        """Update the value in an output cell."""
        if display_type == "2d":
            if len(indices) == 1:
                key = (0, indices[0])
            else:
                key = (indices[0], indices[1])

            if key in text_refs:
                new_txt = Text(
                    f"{value:.1f}" if isinstance(value, (float, np.floating)) else str(int(value)),
                    font_size=18
                )
                new_txt.move_to(text_refs[key].get_center())
                new_txt.set_opacity(1.0)
                self.play(Transform(text_refs[key], new_txt), run_time=0.7)
