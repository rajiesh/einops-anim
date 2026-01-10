"""
Common library for einops animations using Manim Community Edition.

This module provides shared utilities for visualizing einops operations:
- Tensor display functions (2D, 3D, 4D grids)
- Cell highlighting functions
- Text and layout helpers
"""

from __future__ import annotations
from typing import List, Optional, Tuple
import numpy as np
from manim import *


# =========================
# Visualization helpers
# =========================

def safe_text(s: str, font_size: int = 20) -> Text:
    """Create a Text object (no LaTeX to avoid errors)."""
    return Text(s, font_size=font_size)


def number_cell(value: int, cell_size=0.55, font_size=22) -> VGroup:
    """Create a cell with a number inside a square."""
    sq = Square(side_length=cell_size)
    txt = Text(str(int(value)), font_size=font_size)
    txt.move_to(sq.get_center())
    return VGroup(sq, txt)


def grid_from_2d(values_2d: np.ndarray, cell_size=0.55, font_size=22) -> VGroup:
    """
    Display a 2D array as a grid of cells.

    Returns:
        VGroup of rows, each row is a VGroup of cells.
        Structure: VGroup[row][col][0=square, 1=text]
    """
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

    Args:
        tensor: 3D numpy array (depth, rows, cols)
        axes: List of axis names [depth_name, row_name, col_name]

    Returns:
        VGroup of labeled grids.
        Structure: VGroup[d] = VGroup(label, grid)
        where grid is VGroup[row][col][0=square, 1=text]
    """
    if len(axes) < 3:
        # Fall back to 2D display
        if tensor.ndim == 2:
            return grid_from_2d(tensor, cell_size, font_size)
        elif tensor.ndim == 1:
            return grid_from_2d(tensor.reshape(1, -1), cell_size, font_size)

    # For 3D: show as stacked grids (depth, rows, cols)
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

    Args:
        tensor: 4D numpy array (dim0, dim1, dim2, dim3)
        axes: List of axis names [dim0_name, dim1_name, dim2_name, dim3_name]

    For (batch, h, c, w): shows batch rows, each row has c grids of (h x w)
    Structure: batch rows vertically, within each row: c grids horizontally

    Returns:
        VGroup of batch rows.
        Structure: VGroup[batch][1=batch_row_grids][channel][1=grid][row][col][0=square, 1=text]
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


# =========================
# Highlighting functions
# =========================

def highlight_cell(grid: VGroup, r: int, c: int, color=YELLOW) -> SurroundingRectangle:
    """Highlight a cell in a 2D grid display."""
    return SurroundingRectangle(grid[r][c][0], color=color, buff=0.04)


def highlight_cell_3d(grid_3d: VGroup, d: int, r: int, c: int, color=YELLOW) -> SurroundingRectangle:
    """
    Highlight a cell in a 3D grid display.

    Args:
        grid_3d: VGroup structure from grid_from_3d
        d: depth index
        r: row index
        c: column index
    """
    labeled_grid = grid_3d[d]
    grid = labeled_grid[1]  # Skip the label, get the grid
    return SurroundingRectangle(grid[r][c][0], color=color, buff=0.04)


def highlight_cell_4d(grid_4d: VGroup, i0: int, i1: int, i2: int, i3: int, color=YELLOW) -> SurroundingRectangle:
    """
    Highlight a cell in a 4D grid display.

    Args:
        grid_4d: VGroup structure from grid_from_4d
        i0: batch index
        i1: height/row index
        i2: channel index
        i3: width/col index
    """
    batch_row = grid_4d[i0]
    batch_row_grids = batch_row[1]  # Skip batch label
    labeled_grid = batch_row_grids[i2]
    grid = labeled_grid[1]  # Skip the grid label
    return SurroundingRectangle(grid[i1][i3][0], color=color, buff=0.04)


# =========================
# Tensor display helper
# =========================

def display_tensor(tensor: np.ndarray, axes: List[str], name: str, font_size_label=20) -> Tuple[VGroup, str, VGroup]:
    """
    Display a tensor with appropriate dimensionality (2D, 3D, or 4D).

    Args:
        tensor: numpy array to display
        axes: List of axis names
        name: Tensor name for labeling
        font_size_label: Font size for labels

    Returns:
        Tuple of (display_group, display_type, grid):
        - display_group: Complete VGroup with labels and grid
        - display_type: "2d", "3d", or "4d"
        - grid: The grid VGroup for highlighting
    """
    lbl1 = safe_text(f"{name} shape={tensor.shape}", font_size=font_size_label)
    lbl2 = safe_text("axes: " + " ".join(axes), font_size=font_size_label - 4)

    if tensor.ndim == 4 and len(axes) == 4:
        # Use 4D display
        grid = grid_from_4d(tensor, axes, cell_size=0.3, font_size=12)
        disp_text = f"{axes[0]} (batches), {axes[2]} (channels), grids=({axes[1]}×{axes[3]})"
        lbl3 = safe_text(f"display: {disp_text}", font_size=14)
        display_group = VGroup(lbl1, lbl2, lbl3, grid).arrange(DOWN, buff=0.12)
        display_type = "4d"
    elif tensor.ndim == 3 and len(axes) == 3:
        # Use 3D display
        grid = grid_from_3d(tensor, axes, cell_size=0.4, font_size=16)
        disp_text = f"{axes[0]} (stacked), rows={axes[1]}, cols={axes[2]}"
        lbl3 = safe_text(f"display: {disp_text}", font_size=14)
        display_group = VGroup(lbl1, lbl2, lbl3, grid).arrange(DOWN, buff=0.12)
        display_type = "3d"
    else:
        # Use 2D display
        if tensor.ndim == 1:
            tensor_2d = tensor.reshape(1, -1)
        elif tensor.ndim == 2:
            tensor_2d = tensor
        else:
            # Flatten higher dimensions to 2D
            tensor_2d = tensor.reshape(tensor.shape[0], -1)

        grid = grid_from_2d(tensor_2d, cell_size=0.5, font_size=18)

        disp_parts = []
        if tensor.ndim >= 2 and len(axes) >= 2:
            disp_parts.append(f"rows={axes[0]}")
            disp_parts.append(f"cols={axes[-1]}")
        elif tensor.ndim == 1 and len(axes) >= 1:
            disp_parts.append(f"cols={axes[0]}")

        lbl3 = safe_text("display: " + (", ".join(disp_parts) if disp_parts else "scalar"), font_size=14)
        display_group = VGroup(lbl1, lbl2, lbl3, grid).arrange(DOWN, buff=0.12)
        display_type = "2d"

    return display_group, display_type, grid


# =========================
# Axis parsing helpers
# =========================

def parse_axes_spec(spec: str) -> List[str]:
    """Parse a space-separated axis specification into a list of axis names."""
    spec = spec.strip()
    if not spec:
        return []
    return [tok.strip() for tok in spec.split() if tok.strip()]


def assign_minimal_axis_sizes(all_axes: List[str], choices=(2, 3), seed=7) -> dict:
    """
    Deterministic "minimal" size assignment: each axis gets either 2 or 3.
    Stable per axis name given the seed.

    Args:
        all_axes: List of axis names
        choices: Tuple of possible sizes
        seed: Random seed for deterministic generation

    Returns:
        Dictionary mapping axis names to sizes
    """
    rng = np.random.RandomState(seed)
    sizes = {}
    for ax in sorted(set(all_axes)):
        sizes[ax] = int(rng.choice(list(choices)))
    return sizes
