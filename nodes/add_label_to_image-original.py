from math import ceil
from typing import Tuple

from torch import Tensor
from torchvision.transforms.v2.functional import to_pil_image, to_image  # type: ignore
from PIL import Image, ImageDraw
from PIL.ImageFont import FreeTypeFont

from ..font_manager import FontCollection


class AddLabelToImage:
    """
    Extends an image vertically to add a label.
    """

    fonts = FontCollection()

    @classmethod
    def INPUT_TYPES(cls):
        font_names = list(cls.fonts.keys())
        return {
            "required": {
                "image": ("IMAGE",),
                "font": (font_names, {"default": cls.fonts.default_font_name}),
                "label": ("STRING", {"multiline": True}),
                "position": (["top", "bottom"],),
                "text_size": ("INT", {"default": 48, "min": 4}),
                "padding": ("INT", {"default": 24}),
                "line_spacing": ("INT", {"default": 5}),
                "text_color": ("STRING", {"default": "#fff"}),
                "background_color": ("STRING", {"default": "#000"}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "execute"

    CATEGORY = "image/transform"

    def execute(
        self,
        image: Tensor,
        font: str,
        label: str,
        text_size: int,
        padding: int,
        line_spacing: int,
        position: str,
        text_color: str,
        background_color: str,
    ):
        """
        Extends an image vertically to add a label.

        Args:
            image (Tensor): The input image as a tensor with shape [1, H, W, C].
            font (str): The font name to be used for the label.
            label (str): The text of the label.
            text_size (int): The size of the label text in pixels.
            padding (int): Padding around the label text in pixels.
            line_spacing (int): Spacing between lines of the label.
            position (str): Position of the label, either 'top' or 'bottom'.
            text_color (str): Color of the label text as a hex reference.
            background_color (str): Background color of the label area as a hex reference.

        Returns:
            Tensor: The image with the label added, as a tensor with shape [1, H, W, C].

        Raises:
            ValueError: If an invalid position is provided.
        """

        original_image = to_pil_image(image.squeeze(0).permute(2, 0, 1))
        width, height = original_image.size
        font_obj = self.fonts[font].font_variant(size=text_size)

        _, label_height, text_size = self.calculate_label_dimensions(
            font_obj, label, text_size, line_spacing, padding, width
        )

        sized_font = font_obj.font_variant(size=text_size)
        label_image = self.draw_label(
            sized_font, label, width, label_height, line_spacing, text_color, background_color
        )

        combined_image = Image.new("RGB", (width, height + label_height), (0, 0, 0))
        if position == "top":
            combined_image.paste(original_image, (0, label_height))
            combined_image.paste(label_image, (0, 0))
        elif position == "bottom":
            combined_image.paste(label_image, (0, height))
            combined_image.paste(original_image, (0, 0))
        else:
            raise ValueError(f"Unknown position: {position}")

        return (to_image(combined_image) / 255.0).permute(1, 2, 0)[None, None, ...]

    def calculate_label_dimensions(
        self, font: FreeTypeFont, label: str, text_size: int, line_spacing: int, padding: int, max_width: float
    ) -> Tuple[int, int, int]:
        """
        Calculate the dimensions needed to draw a label within an image.

        This will reduce the font size where necessary to make the text fit.

        Args:
            font (FreeTypeFont): The Pillow font to use.
            label (str): The text to calculate dimensions for.
            text_size (int): Starting font size for the label.
            line_spacing (int): Spacing between lines of text.
            padding (int): Padding around the text.
            max_width (float): Maximum allowed width for the text box.

        Returns:
            tuple[int, int, int]: The calculated width, height, and final font size.
        """

        temp_image = Image.new("RGB", (1, 1))
        while True:
            sized_font = font.font_variant(size=text_size)
            x1, y1, x2, y2 = ImageDraw.Draw(temp_image).multiline_textbbox(
                xy=(0, 0),
                text=label,
                font=sized_font,
                spacing=line_spacing,
                align="center",
            )
            ascent, descent = sized_font.getmetrics()
            width = ceil(x2 - x1 + padding * 2)
            height = ceil(y2 - y1 + padding * 2 + (ascent + descent) / 2)
            if width <= max_width:
                break
            text_size -= 1
            if text_size <= 8:
                break

        return width, height, text_size

    def draw_label(
        self,
        font: FreeTypeFont,
        label: str,
        width: int,
        height: int,
        line_spacing: int,
        text_color: str,
        background_color: str,
    ) -> Image.Image:
        """
        Draws an image containing a label.

        Args:
            font (FreeTypeFont): The Pillow font to use for text rendering.
            label (str): The text to use as the label.
            width (int): Width of the image in pixels.
            height (int): Height of the image in pixels.
            line_spacing (int): Spacing between lines of text.
            text_color (str): Color of the text as a hex reference.
            background_color (str): Background color of the image as a hex reference.

        Returns:
            Image: An image object with the label drawn on it.
        """

        image = Image.new("RGB", (width, height), background_color)
        draw = ImageDraw.Draw(image)
        draw.multiline_text(
            xy=(width / 2, height / 2),
            text=label,
            fill=text_color,
            font=font,
            anchor="mm",
            spacing=line_spacing,
            align="center",
        )
        return image
