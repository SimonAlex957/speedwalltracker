import re

# ---- Constants ----
SPACING = 125  # mm
ROWS_PER_PANEL = 10
COLS_PER_PANEL = 11

PANEL_HEIGHT = (ROWS_PER_PANEL - 1)  * SPACING + (2 * 187.5)
PANEL_WIDTH = (COLS_PER_PANEL + 1) * SPACING    # 1375 mm

# Column mapping (no J, K in standard speed wall layout)
COLUMN_INDEX = {
    "A": 1,
    "B": 2,
    "C": 3,
    "D": 4,
    "E": 5,
    "F": 6,
    "G": 7,
    "H": 8,
    "I": 9,
    "L": 10,
    "M": 11
}


def parse_panel(panel_name):
    """
    Extract side (SN or DX) and vertical index number.
    Example: DX2 -> ('DX', 2)
    """
    match = re.match(r"(SN|DX)\s?(\d+)", panel_name.upper())
    if not match:
        raise ValueError("Invalid panel name")

    side = match.group(1)
    number = int(match.group(2))
    return side, number


def parse_position(position):
    """
    Extract column letter and row number.
    Accepts 'F4' or '4F'
    """
    position = position.upper().strip()

    match = re.match(r"([A-Z])(\d+)", position)
    if match:
        col = match.group(1)
        row = int(match.group(2))
        return col, row

    match = re.match(r"(\d+)([A-Z])", position)
    if match:
        row = int(match.group(1))
        col = match.group(2)
        return col, row

    raise ValueError("Invalid hold position format")


def get_hold_coordinates(panel_name, hold_position):
    side, panel_number = parse_panel(panel_name)
    col_letter, row_number = parse_position(hold_position)

    # Local coordinates inside panel
    x_local = (COLUMN_INDEX[col_letter]) * SPACING
    y_local = (row_number - 1) * SPACING + 187.5 

    # Horizontal offset (DX panels are right column)
    if side == "SN":
        x_offset = 0
    else:  # DX
        x_offset = PANEL_WIDTH

    # Vertical offset (stacked panels)
    y_offset = (panel_number - 1) * PANEL_HEIGHT

    # Absolute coordinates
    x_absolute = x_offset + x_local
    y_absolute = y_offset + y_local

    return x_absolute, y_absolute




foot_holds = [
    ["DX1", "F4"],
    ["DX1", "A10"],
    ["SN2", "G3"],
    ["DX3", "C6"],
    ["DX5", "E1"],
    ["SN5", "H1"],
    ["SN5", "M6"],
    ["SN5", "E7"],
    ["DX7", "B10"],
    ["DX8", "E1"],
    ["DX8", "A5"],
]

big_holds = [
    ["DX2", "F1"],
    ["DX2", "G3"],
    ["DX2", "A9"],
    ["SN3", "G4"],
    ["SN3", "M10"],
    ["DX4", "B2"],
    ["SN4", "M8"],
    ["DX5", "C3"],
    ["DX5", "E9"],
    ["SN6", "H2"],
    ["SN6", "L7"],
    ["SN6", "F9"],
    ["SN7", "M4"],   # converted from 4M
    ["SN7", "G9"],   # converted from 9G
    ["SN8", "L1"],   # converted from 1L
    ["SN8", "I3"],   # converted from 3I
    ["SN8", "C8"],
    ["SN9", "M10"],
    ["DX9", "A2"],
    ["DX9", "E7"],
]

foot_coords = []
big_coords = []
for panel, position in big_holds:
    coords = get_hold_coordinates(panel, position)
    big_coords.append(coords)

for panel, position in foot_holds:
    coords = get_hold_coordinates(panel, position)
    foot_coords.append(coords)   
print(big_coords)
print(foot_coords)

import matplotlib.pyplot as plt

x_coords = [coord[0] for coord in big_coords]
y_coords = [coord[1] for coord in big_coords]

plt.figure(figsize=(10, 12))
plt.scatter(x_coords, y_coords, color='blue', s=50)
plt.xlabel('X (mm)')
plt.ylabel('Y (mm)')
plt.title('Speed Wall Big Hold Positions')
plt.grid(True, alpha=0.3)
plt.axis('equal')
plt.savefig('/home/server/codeServer/fyp/speedwalltracker/big_holds_plot.png', dpi=150, bbox_inches='tight')
plt.close()

x_coords = [coord[0] for coord in foot_coords]
y_coords = [coord[1] for coord in foot_coords]

plt.figure(figsize=(10, 12))
plt.scatter(x_coords, y_coords, color='red', s=50)
plt.xlabel('X (mm)')
plt.ylabel('Y (mm)')
plt.title('Speed Wall Foot Hold Positions')
plt.grid(True, alpha=0.3)
plt.axis('equal')
plt.savefig('/home/server/codeServer/fyp/speedwalltracker/foot_holds_plot.png', dpi=150, bbox_inches='tight')
plt.close()