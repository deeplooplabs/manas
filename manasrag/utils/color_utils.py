"""Color utilities for graph visualization.

This module provides color schemes and color generation functions
for visualizing knowledge graphs, communities, and entities.
"""

import colorsys
from typing import Optional


# Entity type color mapping
DEFAULT_TYPE_COLORS = {
    "ORGANIZATION": "#FF6B6B",      # Red
    "PERSON": "#4ECDC4",            # Teal
    "LOCATION": "#45B7D1",          # Blue
    "PRODUCT": "#FFA07A",           # Light Salmon
    "EVENT": "#98D8C8",             # Mint
    "CONCEPT": "#F7DC6F",           # Yellow
    "TECHNICAL_TERM": "#BB8FCE",    # Purple
    "UNKNOWN": "#95A5A6",           # Gray
}


def get_type_color(entity_type: str, custom_colors: Optional[dict[str, str]] = None) -> str:
    """Get color for a given entity type.

    Args:
        entity_type: The type of entity (e.g., "PERSON", "ORGANIZATION")
        custom_colors: Optional custom color mapping

    Returns:
        Hex color string
    """
    colors = custom_colors or DEFAULT_TYPE_COLORS
    return colors.get(entity_type.upper(), colors["UNKNOWN"])


def generate_community_colors(n_communities: int) -> dict[str, str]:
    """Generate distinct colors for N communities using HSV color space.

    Args:
        n_communities: Number of communities to generate colors for

    Returns:
        Dictionary mapping community IDs to hex colors
    """
    colors = {}
    for i in range(n_communities):
        # Distribute hues evenly around the color wheel
        hue = i / n_communities
        saturation = 0.7
        value = 0.9

        # Convert HSV to RGB
        rgb = colorsys.hsv_to_rgb(hue, saturation, value)

        # Convert RGB to hex
        hex_color = "#%02x%02x%02x" % (
            int(rgb[0] * 255),
            int(rgb[1] * 255),
            int(rgb[2] * 255)
        )
        colors[str(i)] = hex_color

    return colors


def create_colormap(
    n: int,
    saturation: float = 0.7,
    value: float = 0.9,
    start_hue: float = 0.0
) -> list[str]:
    """Create a list of N distinct colors.

    Args:
        n: Number of colors to generate
        saturation: Color saturation (0-1)
        value: Color brightness (0-1)
        start_hue: Starting hue position (0-1)

    Returns:
        List of hex color strings
    """
    colors = []
    for i in range(n):
        hue = (start_hue + i / n) % 1.0
        rgb = colorsys.hsv_to_rgb(hue, saturation, value)
        hex_color = "#%02x%02x%02x" % (
            int(rgb[0] * 255),
            int(rgb[1] * 255),
            int(rgb[2] * 255)
        )
        colors.append(hex_color)

    return colors


def get_gradient_color(value: float, min_val: float, max_val: float) -> str:
    """Generate a color from a gradient based on value.

    Creates a blue-to-red gradient where blue represents low values
    and red represents high values.

    Args:
        value: The value to map to a color
        min_val: Minimum value in the range
        max_val: Maximum value in the range

    Returns:
        Hex color string
    """
    # Normalize value to 0-1 range
    if max_val == min_val:
        normalized = 0.5
    else:
        normalized = (value - min_val) / (max_val - min_val)
    normalized = max(0.0, min(1.0, normalized))

    # Create gradient from blue (0.66) to red (0.0)
    hue = 0.66 * (1 - normalized)

    rgb = colorsys.hsv_to_rgb(hue, 0.8, 0.9)
    return "#%02x%02x%02x" % (
        int(rgb[0] * 255),
        int(rgb[1] * 255),
        int(rgb[2] * 255)
    )


def adjust_brightness(hex_color: str, factor: float) -> str:
    """Adjust the brightness of a hex color.

    Args:
        hex_color: Hex color string (e.g., "#FF6B6B")
        factor: Brightness adjustment factor (>1 brightens, <1 darkens)

    Returns:
        Adjusted hex color string
    """
    # Remove '#' and convert to RGB
    hex_color = hex_color.lstrip("#")
    r, g, b = int(hex_color[0:2], 16), int(hex_color[2:4], 16), int(hex_color[4:6], 16)

    # Adjust brightness
    r = min(255, int(r * factor))
    g = min(255, int(g * factor))
    b = min(255, int(b * factor))

    return "#%02x%02x%02x" % (r, g, b)


def get_contrasting_text_color(hex_color: str) -> str:
    """Get black or white text color based on background brightness.

    Args:
        hex_color: Background color in hex format

    Returns:
        "#000000" for dark text or "#FFFFFF" for light text
    """
    # Remove '#' and convert to RGB
    hex_color = hex_color.lstrip("#")
    r, g, b = int(hex_color[0:2], 16), int(hex_color[2:4], 16), int(hex_color[4:6], 16)

    # Calculate luminance
    luminance = (0.299 * r + 0.587 * g + 0.114 * b) / 255

    # Return black for light backgrounds, white for dark backgrounds
    return "#000000" if luminance > 0.5 else "#FFFFFF"
