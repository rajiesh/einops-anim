# Einops Animation Suite

A collection of Manim Community Edition animations for visualizing einops operations.

## Files

- **`einops_manim_lib.py`** - Common library with shared visualization functions
- **`einsum_generalised.py`** - Animation for `einops.einsum()` operation
- **`reduce_anim.py`** - Animation for `einops.reduce()` operation
- **`repeat_anim.py`** - Animation for `einops.repeat()` operation

## Installation

```bash
pip install manim numpy einops
```

## Usage

### Einsum Animation

Edit the configuration in `einsum_generalised.py`:

```python
TENSOR_NAMES = ["a", "b"]
EINSPEC = "h c w, h c w -> w"
```

Run:
```bash
docker run --rm -it -v "./einops-anim:/manim" manimcommunity/manim manim -qm einsum_generalised.py EinsumBudgeted
```

### Reduce Animation

Edit the configuration in `reduce_anim.py`:

```python
TENSOR_NAME = "x"
PATTERN = "h w c -> h w"  # Reduce over channels
REDUCTION = "mean"  # Options: 'mean', 'sum', 'max', 'min', 'prod'
```

Examples:
- `"h w c -> h w"` with `REDUCTION = "mean"` - Average pooling over channels
- `"b h w c -> b c"` with `REDUCTION = "max"` - Max pooling over spatial dimensions
- `"h w -> h"` with `REDUCTION = "sum"` - Sum over width

Run:
```bash
docker run --rm -it -v "./einops-anim:/manim" manimcommunity/manim manim -qm reduce_anim.py ReduceAnim
```

### Repeat Animation

Edit the configuration in `repeat_anim.py`:

```python
TENSOR_NAME = "x"
PATTERN = "h w -> h w c"  # Add new axis
REPEAT_PARAMS = {"c": 3}  # Repeat 3 times along new axis
```

Examples:
- `"h w -> h w c"` with `REPEAT_PARAMS = {"c": 3}` - Add new axis with repetition
- `"h w -> (repeat h) w"` with `REPEAT_PARAMS = {"repeat": 2}` - Repeat along existing axis
- `"h w -> h w 1"` - Add singleton dimension

Run:
```bash
docker run --rm -it -v "./einops-anim:/manim" manimcommunity/manim manim -qm repeat_anim.py RepeatAnim
```

## Features

### Common Features (All Animations)

- **Up to 4D tensors** - Visualizes 2D, 3D, and 4D tensors with appropriate layouts
- **Automatic sizing** - Axes are automatically assigned sizes (2 or 3) for compact display
- **Slow-paced animations** - Increased run times for better comprehension
- **Centered computation display** - Arithmetic shown in center of screen between tensors
- **Element tracking** - Shows how individual elements transform between input and output

### Einsum Specific Features

- Shows all reduction terms (no budgeting)
- Displays tensor[indices]=value format for clarity
- Highlights corresponding cells in input tensors during multiplication
- Step-by-step accumulation of sum

### Reduce Specific Features

- Highlights all input elements being reduced
- Shows reduction formula (mean, sum, max, etc.)
- Updates output cells with computed values

### Repeat Specific Features

- Shows which input elements are copied to multiple output locations
- Highlights repeated values across output tensor

## Animation Settings

All animations support these common settings:

```python
RNG_SEED = 42  # Random seed for reproducible tensors
AXIS_SIZE_CHOICES = (2, 3)  # Possible sizes for auto-assigned axes
```

## Output Formats

Manim supports various quality settings:

- `-ql` - Low quality (480p) - fast rendering
- `-qm` - Medium quality (720p)
- `-qh` - High quality (1080p)
- `-qk` - 4K quality (2160p)

Add `-p` flag to preview after rendering.

Example for high quality:
```bash
manim -pqh einsum_generalised.py EinsumBudgeted
```

## Docker Usage

You can also run the animations using Docker:

```bash
docker run --rm -it -v "/Users/rajieshnarayanan/workspace/einops-animation:/manim" manimcommunity/manim manim -qm einsum_generalised.py EinsumBudgeted
```

## Library Functions

The `einops_manim_lib.py` provides reusable functions:

### Display Functions
- `grid_from_2d()` - Display 2D array as grid
- `grid_from_3d()` - Display 3D tensor as stacked grids
- `grid_from_4d()` - Display 4D tensor as nested grids
- `display_tensor()` - Automatic display based on dimensionality

### Highlighting Functions
- `highlight_cell()` - Highlight 2D grid cell
- `highlight_cell_3d()` - Highlight 3D grid cell
- `highlight_cell_4d()` - Highlight 4D grid cell

### Helper Functions
- `safe_text()` - Create Text objects (no LaTeX)
- `parse_axes_spec()` - Parse axis specifications
- `assign_minimal_axis_sizes()` - Auto-assign axis sizes

## Extending

To create animations for other operations:

1. Import the library: `from einops_manim_lib import *`
2. Use `display_tensor()` to show tensors
3. Use highlighting functions to show element correspondences
4. Follow the layout pattern: header at top, input left, output right, computation center

## Troubleshooting

**Error: "Missing $ inserted"**
- The code avoids LaTeX by using `Text()` instead of `Tex()` or `MathTex()`

**Tensors too small to see**
- Adjust `AXIS_SIZE_CHOICES` to use larger values like `(3, 4)`
- Modify `cell_size` and `font_size` parameters in display functions

**Animation too fast**
- Increase `run_time` parameters in `self.play()` calls
- Increase `self.wait()` durations

**Type errors with indices**
- Check that tensor dimensions match the pattern specification
- Verify `AXIS_PARAMS` or `REPEAT_PARAMS` are correctly set

## License

This code is provided as-is for educational purposes.
