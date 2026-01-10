"""
Generalized einops-style einsum animation using ManimCE.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple, Sequence, Optional

import numpy as np
from manim import *


# =========================
# USER INPUT (edit only these)
# =========================
TENSOR_NAMES = ["a", "b"]  # used only for labels (must match number of inputs in the spec)
EINSPEC = "h c w, h c w -> w"


# =========================
# STYLE CONTROLS
# =========================
RNG_SEED = 7

# Minimal sizes used for axes (chosen deterministically per axis name)
AXIS_SIZE_CHOICES = (2, 3)

# Animation settings - removed budgeting, display everything
ANIMATE_ALL_OUTPUT_ENTRIES = True  # Show all output cells
SHOW_ALL_REDUCTION_TERMS = True    # Show all reduction terms


# =========================
# Parsing + compilation
# =========================
def _split_axes(s: str) -> List[str]:
    s = s.strip()
    if not s:
        return []
    return [tok for tok in s.split() if tok.strip()]


@dataclass(frozen=True)
class EinsumSpec:
    inputs: List[List[str]]   # per operand axis names
    output: List[str]         # output axis names


def parse_einops_einsum_spec(spec: str) -> EinsumSpec:
    if "->" not in spec:
        raise ValueError('Einsum spec must contain "->" (einops.einsum style).')
    lhs, rhs = spec.split("->", 1)
    lhs_parts = [p.strip() for p in lhs.split(",")]
    inputs = [_split_axes(p) for p in lhs_parts]
    output = _split_axes(rhs)

    if any(len(inp) == 0 for inp in inputs):
        raise ValueError("Each input must have at least one axis name.")
    return EinsumSpec(inputs=inputs, output=output)


def build_numpy_einsum_subscripts(spec: EinsumSpec) -> Tuple[str, Dict[str, str]]:
    letters = list("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ")
    named_axes: List[str] = []

    for inp in spec.inputs:
        for ax in inp:
            if ax not in named_axes:
                named_axes.append(ax)
    for ax in spec.output:
        if ax not in named_axes:
            named_axes.append(ax)

    if len(named_axes) > len(letters):
        raise ValueError(f"Too many unique axes ({len(named_axes)}); not enough einsum letters.")

    mapping = {ax: letters[i] for i, ax in enumerate(named_axes)}
    in_subs = ["".join(mapping[ax] for ax in inp) for inp in spec.inputs]
    out_sub = "".join(mapping[ax] for ax in spec.output)
    subscripts = ",".join(in_subs) + "->" + out_sub
    return subscripts, mapping


def assign_minimal_axis_sizes(all_axes: Sequence[str], choices=(2, 3), seed=7) -> Dict[str, int]:
    """
    Deterministic “minimal” size assignment: each axis gets either 2 or 3.
    Stable per axis name given the seed.
    """
    rng = np.random.RandomState(seed)
    sizes: Dict[str, int] = {}
    for ax in sorted(set(all_axes)):
        sizes[ax] = int(rng.choice(list(choices)))
    return sizes


def make_random_int_tensors(spec: EinsumSpec, axis_sizes: Dict[str, int], rng: np.random.RandomState) -> List[np.ndarray]:
    tensors = []
    for inp_axes in spec.inputs:
        shape = tuple(axis_sizes[a] for a in inp_axes)
        t = rng.randint(0, 6, size=shape, dtype=int)
        tensors.append(t)
    return tensors


# =========================
# Visualization helpers
# =========================
def number_cell(value: int, cell_size=0.55, font_size=22) -> VGroup:
    sq = Square(side_length=cell_size)
    txt = Text(str(int(value)), font_size=font_size)
    txt.move_to(sq.get_center())
    return VGroup(sq, txt)


def grid_from_2d(values_2d: np.ndarray, cell_size=0.55, font_size=22) -> VGroup:
    rows, cols = values_2d.shape
    g = VGroup()
    for r in range(rows):
        row = VGroup()
        for c in range(cols):
            row.add(number_cell(int(values_2d[r, c]), cell_size=cell_size, font_size=font_size))
        row.arrange(RIGHT, buff=0.07)
        g.add(row)
    g.arrange(DOWN, buff=0.07)
    return g


def grid_from_3d(tensor: np.ndarray, axes: List[str], cell_size=0.45, font_size=18) -> VGroup:
    """
    Display a 3D tensor as stacked 2D grids showing all dimensions.
    Returns display_group containing labeled grids.
    Structure: VGroup of [labeled_grid_0, labeled_grid_1, ...]
    Each labeled_grid is: VGroup(label, grid)
    Each grid is: VGroup of rows, each row is VGroup of cells
    """
    if len(axes) < 3:
        # Fall back to 2D display
        if tensor.ndim == 2:
            return grid_from_2d(tensor, cell_size, font_size)
        elif tensor.ndim == 1:
            return grid_from_2d(tensor.reshape(1, -1), cell_size, font_size)

    # For 3D: show as stacked grids (depth, rows, cols)
    # Display as: first axis creates separate grids, middle axis = rows, last axis = cols
    depth, rows, cols = tensor.shape

    all_grids = VGroup()
    for d in range(depth):
        slice_2d = tensor[d, :, :]
        grid = grid_from_2d(slice_2d, cell_size=cell_size, font_size=font_size)

        # Add a label for this slice
        label = Text(f"{axes[0]}={d}", font_size=14)
        labeled_grid = VGroup(label, grid).arrange(DOWN, buff=0.08)
        all_grids.add(labeled_grid)

    # Arrange grids horizontally with spacing
    all_grids.arrange(RIGHT, buff=0.3)

    return all_grids


def grid_from_4d(tensor: np.ndarray, axes: List[str], cell_size=0.35, font_size=14) -> VGroup:
    """
    Display a 4D tensor as nested stacked grids showing all dimensions.
    For (batch, h, c, w): shows batch rows, each row has c grids of (h x w)
    Structure: batch rows vertically, within each row: c grids horizontally
    """
    if len(axes) < 4:
        # Fall back to 3D display
        return grid_from_3d(tensor, axes, cell_size, font_size)

    # For 4D: (dim0, dim1, dim2, dim3)
    # Display as: dim0 (rows of groups), dim2 (grids per group), each grid is (dim1 x dim3)
    dim0, dim1, dim2, dim3 = tensor.shape

    all_batch_rows = VGroup()
    for i0 in range(dim0):
        # Create a row for this batch element
        batch_row_grids = VGroup()

        for i2 in range(dim2):
            # Each grid shows slice [i0, :, i2, :]
            slice_2d = tensor[i0, :, i2, :]
            grid = grid_from_2d(slice_2d, cell_size=cell_size, font_size=font_size)

            # Label this grid
            label = Text(f"{axes[2]}={i2}", font_size=12)
            labeled_grid = VGroup(label, grid).arrange(DOWN, buff=0.05)
            batch_row_grids.add(labeled_grid)

        # Arrange grids horizontally
        batch_row_grids.arrange(RIGHT, buff=0.2)

        # Add batch label
        batch_label = Text(f"{axes[0]}={i0}", font_size=14, color=BLUE)
        labeled_batch_row = VGroup(batch_label, batch_row_grids).arrange(RIGHT, buff=0.15)
        all_batch_rows.add(labeled_batch_row)

    # Arrange batch rows vertically
    all_batch_rows.arrange(DOWN, buff=0.25, aligned_edge=LEFT)

    return all_batch_rows


def choose_display_axes(axes: List[str]) -> Tuple[Optional[str], Optional[str]]:
    if len(axes) == 0:
        return None, None
    if len(axes) == 1:
        return None, axes[0]  # display as 1xN
    return axes[0], axes[-1]  # rows=first, cols=last


def slice_to_2d(tensor: np.ndarray, axes: List[str], fixed_index: Dict[str, int]) -> Tuple[np.ndarray, Optional[str], Optional[str]]:
    row_ax, col_ax = choose_display_axes(axes)
    if row_ax is None and col_ax is None:
        return np.array([[int(tensor)]]), None, None

    # For tensors with 3+ dimensions, we want to keep all dimensions and reshape appropriately
    # Only slice out dimensions that are in fixed_index AND are not display axes
    indexer = []
    kept_axes = []
    for ax in axes:
        if ax == row_ax or ax == col_ax:
            indexer.append(slice(None))
            kept_axes.append(ax)
        elif ax in fixed_index:
            # Only fix this axis if it's explicitly in fixed_index
            indexer.append(int(fixed_index[ax]))
        else:
            # Keep all other axes - don't arbitrarily slice them
            indexer.append(slice(None))
            kept_axes.append(ax)

    view = tensor[tuple(indexer)]

    if row_ax is None and col_ax is not None:
        return np.array(view, dtype=int).reshape(1, -1), None, col_ax

    view2d = np.array(view, dtype=int)
    # Properly handle multi-dimensional tensors
    if view2d.ndim == 1:
        view2d = view2d.reshape(1, -1) if col_ax else view2d.reshape(-1, 1)
    elif view2d.ndim > 2:
        # For 3D+, reshape by keeping first dim and flattening the rest into columns
        # This ensures a (3,2,3) becomes (3,6) not (3,3)
        view2d = view2d.reshape(view2d.shape[0], -1)
    return view2d, row_ax, col_ax


def highlight_cell(grid: VGroup, r: int, c: int, color=YELLOW) -> SurroundingRectangle:
    return SurroundingRectangle(grid[r][c][0], color=color, buff=0.04)


def highlight_cell_3d(grid_3d: VGroup, d: int, r: int, c: int, color=YELLOW) -> SurroundingRectangle:
    """
    Highlight a cell in a 3D grid display.
    grid_3d structure: VGroup of labeled_grids
    labeled_grid[d] = VGroup(label, grid)
    grid = VGroup of rows
    """
    labeled_grid = grid_3d[d]
    grid = labeled_grid[1]  # Skip the label, get the grid
    return SurroundingRectangle(grid[r][c][0], color=color, buff=0.04)


def highlight_cell_4d(grid_4d: VGroup, i0: int, i1: int, i2: int, i3: int, color=YELLOW) -> SurroundingRectangle:
    """
    Highlight a cell in a 4D grid display.
    grid_4d structure: VGroup of batch_rows
    batch_row[i0] = VGroup(batch_label, batch_row_grids)
    batch_row_grids = VGroup of labeled_grids
    labeled_grid[i2] = VGroup(label, grid)
    grid[i1][i3] = the cell we want
    """
    batch_row = grid_4d[i0]
    batch_row_grids = batch_row[1]  # Skip batch label
    labeled_grid = batch_row_grids[i2]
    grid = labeled_grid[1]  # Skip the grid label
    return SurroundingRectangle(grid[i1][i3][0], color=color, buff=0.04)


def safe_text(s: str, font_size: int = 20) -> Text:
    return Text(s, font_size=font_size)


# =========================
# Scene
# =========================
class EinsumBudgeted(Scene):
    def construct(self):
        # Parse spec
        spec = parse_einops_einsum_spec(EINSPEC)
        if len(TENSOR_NAMES) != len(spec.inputs):
            raise ValueError(
                f"TENSOR_NAMES has {len(TENSOR_NAMES)} names but spec has {len(spec.inputs)} inputs."
            )

        # Collect all axes and assign minimal sizes (2 or 3)
        all_axes = []
        for inp in spec.inputs:
            all_axes.extend(inp)
        all_axes.extend(spec.output)

        axis_sizes = assign_minimal_axis_sizes(all_axes, choices=AXIS_SIZE_CHOICES, seed=RNG_SEED)

        # Create tensors + compute output
        rng = np.random.RandomState(RNG_SEED)
        tensors = make_random_int_tensors(spec, axis_sizes, rng)
        np_subs, _ = build_numpy_einsum_subscripts(spec)
        out = np.einsum(np_subs, *tensors)
        out_arr = np.array(out)

        # Reduction axes
        all_in_axes: List[str] = []
        for axes in spec.inputs:
            for ax in axes:
                if ax not in all_in_axes:
                    all_in_axes.append(ax)
        reduce_axes = [ax for ax in all_in_axes if ax not in spec.output]

        # Header
        code_line = safe_text(
            f'einops.einsum({", ".join(TENSOR_NAMES)}, "{EINSPEC}")',
            font_size=26
        ).to_edge(UP)

        subs_line = safe_text(f"compiled to einops.einsum('{np_subs}', ...)", font_size=20).next_to(
            code_line, DOWN, buff=0.12
        )
        axes_line = safe_text(
            "axis sizes: " + ", ".join([f"{k}={v}" for k, v in sorted(axis_sizes.items())]),
            font_size=18
        ).next_to(subs_line, DOWN, buff=0.10)

        self.play(Write(code_line), run_time=1.0)
        self.play(FadeIn(subs_line, shift=0.1 * DOWN), run_time=0.8)
        self.play(FadeIn(axes_line, shift=0.1 * DOWN), run_time=0.8)
        self.wait(1.0)

        # Display operands
        fixed_index_default = {ax: 0 for ax in axis_sizes.keys()}

        operand_groups = VGroup()
        operand_displays = []  # (op_axes, grid_structure, display_info, tensor)

        for name, t, axes in zip(TENSOR_NAMES, tensors, spec.inputs):
            lbl1 = safe_text(f"{name} shape={t.shape}", font_size=20)
            lbl2 = safe_text("axes: " + " ".join(axes), font_size=16)

            # Check tensor dimensionality
            if t.ndim == 4 and len(axes) == 4:
                # Use 4D display
                grid = grid_from_4d(t, axes, cell_size=0.3, font_size=12)
                disp_text = f"{axes[0]} (batches), {axes[2]} (channels), grids=({axes[1]}×{axes[3]})"
                lbl3 = safe_text(f"display: {disp_text}", font_size=14)
                g = VGroup(lbl1, lbl2, lbl3, grid).arrange(DOWN, buff=0.12)
                operand_groups.add(g)
                # Store info for highlighting: (axes, grid_structure, is_4d, tensor)
                operand_displays.append((axes, grid, "4d", t))
            elif t.ndim == 3 and len(axes) == 3:
                # Use 3D display
                grid = grid_from_3d(t, axes, cell_size=0.4, font_size=16)
                disp_text = f"{axes[0]} (stacked), rows={axes[1]}, cols={axes[2]}"
                lbl3 = safe_text(f"display: {disp_text}", font_size=14)
                g = VGroup(lbl1, lbl2, lbl3, grid).arrange(DOWN, buff=0.12)
                operand_groups.add(g)
                # Store info for highlighting: (axes, grid_structure, is_3d, tensor)
                operand_displays.append((axes, grid, "3d", t))
            else:
                # Use 2D display (original behavior)
                view2d, row_ax, col_ax = slice_to_2d(t, axes, fixed_index_default)
                grid = grid_from_2d(view2d, cell_size=0.5, font_size=18)

                disp = []
                if row_ax is not None:
                    disp.append(f"rows={row_ax}")
                if col_ax is not None:
                    disp.append(f"cols={col_ax}")
                lbl3 = safe_text("display: " + (", ".join(disp) if disp else "scalar"), font_size=14)

                g = VGroup(lbl1, lbl2, lbl3, grid).arrange(DOWN, buff=0.12)
                operand_groups.add(g)
                # Store info: (axes, grid, is_2d, tensor, row_ax, col_ax, fixed_index)
                operand_displays.append((axes, grid, "2d", t, row_ax, col_ax, fixed_index_default.copy()))

        # Arrange operands vertically on the left to avoid horizontal overlap
        operand_groups.arrange(DOWN, buff=0.4, aligned_edge=LEFT).scale(0.75)
        operand_groups.next_to(axes_line, DOWN, buff=0.35).to_edge(LEFT)

        self.play(FadeIn(operand_groups, shift=0.15 * DOWN), run_time=1.2)
        self.wait(1.0)

        # Display output tensor (support both 2D and 3D)
        out_axes = spec.output
        out_is_3d = out_arr.ndim == 3 and len(out_axes) == 3
        out_view2d: Optional[np.ndarray] = None  # Initialize for 2D case
        out_text_refs_3d: List[List[List[Text]]] = []  # Initialize for 3D case
        out_text_refs_2d: List[List[Text]] = []  # Initialize for 2D case

        if out_is_3d:
            # 3D output - create placeholder grids
            depth, rows, cols = out_arr.shape
            out_grid = VGroup()
            # out_text_refs_3d already initialized above

            for d in range(depth):
                slice_grid = VGroup()
                slice_text_refs: List[List[Text]] = []

                for r in range(rows):
                    row = VGroup()
                    row_texts = []
                    for c in range(cols):
                        cell = number_cell(0, cell_size=0.4, font_size=16)
                        cell[1].set_opacity(0.25)
                        row.add(cell)
                        row_texts.append(cell[1])
                    row.arrange(RIGHT, buff=0.07)
                    slice_grid.add(row)
                    slice_text_refs.append(row_texts)
                slice_grid.arrange(DOWN, buff=0.07)

                # Add label for this slice
                label = Text(f"{out_axes[0]}={d}", font_size=14)
                labeled_grid = VGroup(label, slice_grid).arrange(DOWN, buff=0.08)
                out_grid.add(labeled_grid)
                out_text_refs_3d.append(slice_text_refs)

            out_grid.arrange(RIGHT, buff=0.3)

            out_lbl1 = safe_text(f"out shape={out_arr.shape}", font_size=20)
            out_lbl2 = safe_text("axes: " + " ".join(out_axes), font_size=16)
            disp_text = f"{out_axes[0]} (stacked), rows={out_axes[1]}, cols={out_axes[2]}"
            out_lbl3 = safe_text(f"display: {disp_text}", font_size=14)
            out_group = VGroup(out_lbl1, out_lbl2, out_lbl3, out_grid).arrange(DOWN, buff=0.12).scale(0.7)
        else:
            # 2D output (original behavior)
            out_view2d, _, _ = slice_to_2d(out_arr, out_axes, fixed_index_default)

            out_grid = VGroup()
            # out_text_refs_2d already initialized above
            rows, cols = out_view2d.shape
            for r in range(rows):
                row = VGroup()
                row_texts = []
                for c in range(cols):
                    cell = number_cell(0, cell_size=0.5, font_size=18)
                    cell[1].set_opacity(0.25)
                    row.add(cell)
                    row_texts.append(cell[1])
                row.arrange(RIGHT, buff=0.07)
                out_grid.add(row)
                out_text_refs_2d.append(row_texts)
            out_grid.arrange(DOWN, buff=0.07)

            out_lbl1 = safe_text(f"out shape={out_arr.shape}", font_size=20)
            out_lbl2 = safe_text("axes: " + (" ".join(out_axes) if out_axes else "(scalar)"), font_size=16)
            out_group = VGroup(out_lbl1, out_lbl2, out_grid).arrange(DOWN, buff=0.12).scale(0.85)

        # Position output on the right side, aligned with the top of operands
        out_group.to_edge(RIGHT).align_to(operand_groups, UP)

        self.play(FadeIn(out_group, shift=0.15 * DOWN), run_time=1.2)
        self.wait(1.0)

        reduce_line = safe_text(
            "reduced axes: " + (" ".join(reduce_axes) if reduce_axes else "(none)"),
            font_size=18
        ).next_to(out_group, DOWN, buff=0.25).align_to(out_group, LEFT)
        self.play(FadeIn(reduce_line), run_time=0.8)
        self.wait(1.0)

        # Build reduction combinations
        reduce_sizes = [axis_sizes[ax] for ax in reduce_axes]
        if reduce_axes:
            all_reduce_combos = list(np.ndindex(*reduce_sizes))
        else:
            all_reduce_combos = [()]

        # Helper functions for indexing
        def operand_index_tuple(op_axes: List[str], out_idx: Dict[str, int], red_combo: Tuple[int, ...]) -> Tuple[int, ...]:
            red_map = {ax: val for ax, val in zip(reduce_axes, red_combo)}
            idx = []
            for ax in op_axes:
                if ax in out_idx:
                    idx.append(int(out_idx[ax]))
                elif ax in red_map:
                    idx.append(int(red_map[ax]))
                else:
                    idx.append(0)
            return tuple(idx)

        # Animate all output cells and all reduction terms (no budgeting)
        # Generate all output cell coordinates based on whether output is 2D or 3D
        if out_is_3d:
            depth, rows, cols = out_arr.shape
            # For 3D: iterate through (d, r, c)
            candidate_cells = [(d, r, c) for d in range(depth) for r in range(rows) for c in range(cols)]
        else:
            assert out_view2d is not None, "out_view2d should be initialized for 2D output"
            rows, cols = out_view2d.shape
            # For 2D: iterate through (r, c) - store as (None, r, c) for unified handling
            candidate_cells = [(None, r, c) for r in range(rows) for c in range(cols)]

        # Animate all output cells (no budgeting)
        if ANIMATE_ALL_OUTPUT_ENTRIES:
            target_cells = candidate_cells
        else:
            target_cells = candidate_cells[:6]  # Fallback limit

        # Show all reduction terms (no budgeting)
        if SHOW_ALL_REDUCTION_TERMS:
            terms_to_show = len(all_reduce_combos)
        else:
            terms_to_show = min(len(all_reduce_combos), 10)  # Fallback limit

        # Work area for arithmetic - position in the center of the screen
        work_anchor = safe_text(".", font_size=18).move_to(ORIGIN)
        work_anchor.set_opacity(0)  # Make it invisible
        self.add(work_anchor)

        # Animate each chosen output cell
        for cell_coords in target_cells:
            # Unpack coordinates based on 2D or 3D
            if out_is_3d:
                # For 3D: cell_coords is (d, r, c) where all are ints
                d_val, r_val, c_val = cell_coords  # type: int, int, int
                # Map (d, r, c) to axis indices
                out_idx = {out_axes[0]: d_val, out_axes[1]: r_val, out_axes[2]: c_val}
                true_val = int(out_arr[d_val, r_val, c_val])
                # Highlight output cell in 3D display
                out_hl = highlight_cell_3d(out_grid, d_val, r_val, c_val, color=GREEN)
            else:
                # For 2D: cell_coords is (None, r, c)
                _, r_val, c_val = cell_coords
                d_val = None  # Not used for 2D
                # Map (r, c) to axis indices - need to figure out which axes map to rows/cols
                out_idx = {}
                if len(out_axes) >= 2:
                    out_idx[out_axes[0]] = r_val  # First axis = rows
                    out_idx[out_axes[-1]] = c_val  # Last axis = cols
                elif len(out_axes) == 1:
                    out_idx[out_axes[0]] = c_val  # Single axis = cols
                assert out_view2d is not None
                true_val = int(out_view2d[r_val, c_val])
                # Highlight output cell in 2D display
                out_hl = highlight_cell(out_grid, r_val, c_val, color=GREEN)

            self.play(Create(out_hl), run_time=0.6)

            # Show which output entry we're computing (named indices)
            idx_parts = [f"{ax}={out_idx.get(ax, 0)}" for ax in out_axes]
            header = safe_text("compute out[" + (", ".join(idx_parts) if idx_parts else "scalar") + "]", font_size=20)
            header.move_to(ORIGIN)

            self.play(FadeIn(header, shift=0.1 * UP), run_time=0.6)

            # Reduction term loop (show first N)
            combos = all_reduce_combos[:terms_to_show]
            running_sum = 0

            sum_line = safe_text("sum = 0", font_size=20).next_to(header, DOWN, buff=0.1).align_to(header, LEFT)
            self.play(FadeIn(sum_line, shift=0.25 * UP), run_time=0.6)

            for red_combo in combos:
                prod = 1
                operand_values = []
                operand_descriptions = []  # Store tensor[indices] descriptions
                highlights = VGroup()

                for op_idx_in_list, op_display in enumerate(operand_displays):
                    op_axes = op_display[0]
                    op_grid = op_display[1]
                    display_type = op_display[2]
                    t = op_display[3]

                    op_idx = operand_index_tuple(op_axes, out_idx, red_combo)
                    val = int(np.array(t)[op_idx])
                    operand_values.append(val)

                    # Build description like "a[0,1,2]"
                    tensor_name = TENSOR_NAMES[op_idx_in_list]
                    indices_str = ",".join(str(i) for i in op_idx)
                    operand_descriptions.append(f"{tensor_name}[{indices_str}]")

                    prod *= val

                    # highlight displayed cell based on display type
                    if display_type == "4d":
                        # 4D display: op_idx is (i0, i1, i2, i3) matching tensor shape
                        if len(op_idx) == 4:
                            i0, i1, i2, i3 = op_idx
                            if (0 <= i0 < t.shape[0] and 0 <= i1 < t.shape[1] and
                                0 <= i2 < t.shape[2] and 0 <= i3 < t.shape[3]):
                                highlights.add(highlight_cell_4d(op_grid, i0, i1, i2, i3, color=YELLOW))
                    elif display_type == "3d":
                        # 3D display: op_idx is (d, r, c) matching tensor shape
                        if len(op_idx) == 3:
                            d, rr, cc = op_idx
                            if 0 <= d < t.shape[0] and 0 <= rr < t.shape[1] and 0 <= cc < t.shape[2]:
                                highlights.add(highlight_cell_3d(op_grid, d, rr, cc, color=YELLOW))
                    elif display_type == "2d":
                        # 2D display (original behavior)
                        op_row_ax = op_display[4]
                        op_col_ax = op_display[5]
                        op_fixed = op_display[6]

                        if op_row_ax is None and op_col_ax is None:
                            continue

                        rr = 0
                        cc = 0
                        if op_row_ax is not None:
                            rr = op_idx[op_axes.index(op_row_ax)]
                        if op_col_ax is not None:
                            cc = op_idx[op_axes.index(op_col_ax)]

                        displayed = slice_to_2d(np.array(t), op_axes, op_fixed)[0]
                        if 0 <= rr < displayed.shape[0] and 0 <= cc < displayed.shape[1]:
                            highlights.add(highlight_cell(op_grid, rr, cc, color=YELLOW))

                running_sum += prod

                if len(highlights) > 0:
                    self.play(Create(highlights), run_time=0.5)

                # term text (no LaTeX) - show tensor names with indices and values
                red_part = ""
                if reduce_axes:
                    red_kv = ", ".join([f"{ax}={v}" for ax, v in zip(reduce_axes, red_combo)])
                    red_part = f"   ({red_kv})"

                # Build term with both tensor[indices]=value and the product
                term_parts = [f"{desc}={val}" for desc, val in zip(operand_descriptions, operand_values)]
                term_str = " * ".join(term_parts) + f" = {prod}" + red_part

                term_line = safe_text(
                    "term: " + term_str,
                    font_size=16
                ).next_to(sum_line, DOWN, buff=0.08).align_to(sum_line, LEFT)

                new_sum_line = safe_text(f"sum = {running_sum}", font_size=20).move_to(sum_line)

                self.play(FadeIn(term_line, shift=0.05 * UP), Transform(sum_line, new_sum_line), run_time=0.8)
                self.wait(3)
                self.play(FadeOut(term_line), run_time=0.8)

                if len(highlights) > 0:
                    self.play(FadeOut(highlights), run_time=0.4)

            # Show true value and write it into the output cell
            true_line = safe_text(f"true value = {true_val}", font_size=18).next_to(sum_line, DOWN, buff=0.10).align_to(sum_line, LEFT)
            self.play(FadeIn(true_line, shift=0.05 * UP), run_time=0.6)
            self.wait(0.5)

            new_txt = Text(str(true_val), font_size=16 if out_is_3d else 18)
            if out_is_3d:
                # Type narrowing: in 3D branch, d_val is guaranteed to be int
                assert d_val is not None, "d_val must be set in 3D mode"
                assert len(out_text_refs_3d) > 0, "out_text_refs_3d must be populated in 3D mode"
                new_txt.move_to(out_text_refs_3d[d_val][r_val][c_val].get_center())
                new_txt.set_opacity(1.0)
                self.play(Transform(out_text_refs_3d[d_val][r_val][c_val], new_txt), run_time=0.7)
                # Indicate the cell in 3D grid
                labeled_grid = out_grid[d_val]
                grid = labeled_grid[1]
                self.play(Indicate(grid[r_val][c_val][0], scale_factor=1.06), run_time=0.7)
            else:
                # Type narrowing: in 2D branch, out_text_refs_2d is guaranteed to be populated
                assert len(out_text_refs_2d) > 0, "out_text_refs_2d must be populated in 2D mode"
                new_txt.move_to(out_text_refs_2d[r_val][c_val].get_center())
                new_txt.set_opacity(1.0)
                self.play(Transform(out_text_refs_2d[r_val][c_val], new_txt), run_time=0.7)
                self.play(Indicate(out_grid[r_val][c_val][0], scale_factor=1.06), run_time=0.7)

            self.play(FadeOut(true_line), FadeOut(sum_line), FadeOut(header), FadeOut(out_hl), run_time=0.6)
            self.wait(0.5)

        # Since we animate all cells now, this section is no longer needed
        # (All cells are already filled in the loop above)

        done = safe_text("Done (axis sizes were auto-assigned to 2 or 3).", font_size=18).to_edge(DOWN)
        self.play(FadeIn(done, shift=0.1 * UP), run_time=0.8)
        self.wait(2.0)
