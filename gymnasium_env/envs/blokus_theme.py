__all__ = ["BlokusTheme"]

class BlokusTheme:
    default_black = (0, 0, 0)
    def __init__(self, render_scale = None, cell_colors=None, background_color=None, grid_color=None):
        self._background_color = background_color or (255, 255, 255) # White 

        if not cell_colors:
            self._expander_colors = [
                self.background_color, # Default
                (242, 204, 199), # Light Red
                (218, 230, 253), # Light Blue
                (205, 239, 205), # Light Green
                (230, 198, 133), # Light Yellow
            ]
            self._cell_colors = [
                self.background_color, # Default
                (192, 0, 0),     # Red
                (66, 132, 243),     # Blue
                (26, 127, 55),      # Green
                (189, 127, 0)   # Yellow
            ]
        else: 
            self._expander_colors = [
                tuple(min(255, c + 50) for c in cell_colors[0]), # Default
                tuple(min(255, c + 50) for c in cell_colors[1]) if len(cell_colors) > 1 else None,
                tuple(min(255, c + 50) for c in cell_colors[2]) if len(cell_colors) > 2 else None,
                tuple(min(255, c + 50) for c in cell_colors[3]) if len(cell_colors) > 3 else None,
                tuple(min(255, c + 50) for c in cell_colors[4]) if len(cell_colors) > 4 else None,
            ]
            self._cell_colors = cell_colors

        self._locked_colors = self._expander_colors
        self._grid_color = grid_color or (200, 200, 200)  # Gray
        self.render_scale = render_scale or 10.0
        self._normalise_colors()

    def _normalise_colors(self):
        self._cell_colors_norm = [
            tuple(c / 255.0 for c in color) if color else None
            for color in self._cell_colors
        ]
        self._expander_colors_norm = [
            tuple(c / 255.0 for c in color) if color else None
            for color in self._expander_colors
        ]
        self._locked_colors_norm = [
            tuple(c / 255.0 for c in color) if color else None
            for color in self._locked_colors
        ]
        self._background_color_norm = tuple(c / 255.0 for c in self._background_color)
        self._grid_color_norm = tuple(c / 255.0 for c in self._grid_color)

    def cell_color(self, player_id):
        """Default player_id: 0 = white, 1 = red, 2 = blue, 3 = green, 4 = yellow"""
        if 0 <= player_id < len(self._cell_colors):
            return self._cell_colors[player_id]
        raise ValueError(f"Invalid player ID: {player_id}. Must be between 0 and {len(self._cell_colors) - 1}.")
    
    def expander_color(self, player_id):
        """Default player_id: 0 = white, 1 = red, 2 = blue, 3 = green, 4 = yellow"""
        if 0 <= player_id < len(self._expander_colors):
            return self._expander_colors[player_id]
        raise ValueError(f"Invalid player ID: {player_id}. Must be between 1 and {len(self._expander_colors) - 1}.")

    def locked_color(self, player_id):
        """Default player_id: 0 = white, 1 = red, 2 = blue, 3 = green, 4 = yellow"""
        if 0 <= player_id < len(self._locked_colors):
            return self._locked_colors[player_id]
        raise ValueError(f"Invalid player ID: {player_id}. Must be between 1 and {len(self._locked_colors) - 1}.")

    @property
    def background_color(self):
        return self._background_color
    
    @property
    def grid_color(self):
        return self._grid_color
    
    ## NORMALISED COLORS 
    def cell_color_norm(self, player_id):
        if 0 <= player_id < len(self._cell_colors_norm):
            return self._cell_colors_norm[player_id]
        raise ValueError(f"Invalid player ID: {player_id}. Must be between 0 and {len(self._cell_colors_norm) - 1}.")
    
    def expander_color_norm(self, player_id):
        if 0 <= player_id < len(self._expander_colors_norm):
            return self._expander_colors_norm[player_id]
        raise ValueError(f"Invalid player ID: {player_id}. Must be between 1 and {len(self._expander_colors_norm) - 1}.")
    
    def locked_color_norm(self, player_id):
        if 0 <= player_id < len(self._locked_colors_norm):
            return self._locked_colors_norm[player_id]
        raise ValueError(f"Invalid player ID: {player_id}. Must be between 1 and {len(self._locked_colors_norm) - 1}.")
    
    @property
    def background_color_norm(self):
        return self._background_color_norm
    
    @property
    def grid_color_norm(self):
        return self._grid_color_norm