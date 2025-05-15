from math import ceil
from typing import Tuple, List

import torch
from torch import Tensor
from torchvision.transforms.v2.functional import to_pil_image, to_image
from PIL import Image, ImageDraw, ImageFont # 確保 ImageFont 導入
from PIL.ImageFont import FreeTypeFont

from ..font_manager import FontCollection


class AddLabelToImage:
    """
    Draws text directly onto a batch of images.
    Labels are provided as a multi-line string, each line for a corresponding image.
    The image dimensions are NOT changed by this node itself.
    It is assumed that the input 'image' batch contains images of THE SAME dimensions
    (e.g., ensured by an upstream 'Image List to Image Batch' node or similar).
    """

    fonts = FontCollection()

    def _parse_color_with_alpha(self, color_hex: str, default_alpha: int = 255) -> Tuple[int, int, int, int]:
        color_hex = color_hex.lstrip('#')
        if len(color_hex) == 6:
            r, g, b = tuple(int(color_hex[i:i+2], 16) for i in (0, 2, 4))
            return r, g, b, default_alpha
        elif len(color_hex) == 8:
            r, g, b, a = tuple(int(color_hex[i:i+2], 16) for i in (0, 2, 4, 6))
            return r, g, b, a
        else:
            print(f"[AddLabelToImage WARNING] Invalid color format: '{color_hex}'. Using black opaque as fallback.")
            return 0, 0, 0, 255

    @classmethod
    def INPUT_TYPES(cls):
        font_names = list(cls.fonts.keys())
        default_font_for_ui = ""
        if hasattr(cls.fonts, 'default_font_name') and cls.fonts.default_font_name and cls.fonts.default_font_name in font_names:
            default_font_for_ui = cls.fonts.default_font_name
        elif font_names:
            default_font_for_ui = font_names[0]
        else:
            font_names.append("No fonts available")
            default_font_for_ui = "No fonts available"

        return {
            "required": {
                "image": ("IMAGE",),
                "label_text": ("STRING", {"multiline": True, "default": "Label 1\nLabel 2"}),
                "font_name": (font_names, {"default": default_font_for_ui}),
                "text_position": (["bottom_center", "top_center", "bottom_left", "bottom_right", "top_left", "top_right", "center_center"], {"default": "bottom_center"}),
                "font_size": ("INT", {"default": 48, "min": 4, "max": 1024, "step": 1}),
                "margin": ("INT", {"default": 24, "min": 0, "max": 256, "step": 1}),
                "line_spacing": ("INT", {"default": 5, "min": 0, "max": 128, "step": 1}),
                "text_color_hex": ("STRING", {"default": "#ffffff"}),
                "background_color_hex": ("STRING", {"default": "#00000080"}), # 參數名與方法簽名匹配
                "background_padding": ("INT", {"default": 5, "min": 0, "max": 50, "step": 1}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "execute_draw_on_batch"
    CATEGORY = "image/transform"

    def execute_draw_on_batch(
        self,
        image: Tensor,
        label_text: str,
        font_name: str,
        text_position: str,
        font_size: int,
        margin: int,
        line_spacing: int,
        text_color_hex: str,
        background_color_hex: str, # 方法參數名與 INPUT_TYPES 鍵名匹配
        background_padding: int,
    ):
        print(f"\n[AddLabelToImage EXECUTE_DRAW_ON_BATCH START]")
        if not isinstance(image, torch.Tensor) or image.ndim != 4:
            print(f"[AddLabelToImage ERROR] Input image is not a valid batch tensor. Shape: {image.shape if isinstance(image, torch.Tensor) else type(image)}")
            bs = image.shape[0] if isinstance(image, torch.Tensor) and image.ndim == 4 else 1
            return (torch.zeros((bs, 64, 64, 3), dtype=image.dtype if isinstance(image, torch.Tensor) else torch.float32, device=image.device if isinstance(image, torch.Tensor) else 'cpu'),)

        print(f"  Input image batch shape: {image.shape}")

        label_lines = [line.strip() for line in label_text.strip().split('\n') if line.strip()]
        if not label_lines:
            print("[AddLabelToImage INFO] No labels provided. Returning original image batch.")
            return (image,)

        num_provided_labels = len(label_lines)
        batch_size = image.shape[0]
        processed_pil_images_chw: List[Tensor] = []

        try:
            base_font_object = self.fonts[font_name]
        except KeyError:
            available_fonts = list(self.fonts.keys())
            print(f"[AddLabelToImage ERROR] Font '{font_name}' not found. Available: {available_fonts}. Returning original image batch.")
            return (image,)

        parsed_text_color = self._parse_color_with_alpha(text_color_hex, 255)
        # 使用正確的參數名 background_color_hex
        parsed_bg_color = self._parse_color_with_alpha(background_color_hex, 128) 
        print(f"  Parsed background_color_hex '{background_color_hex}' to RGBA: {parsed_bg_color}")

        for i in range(batch_size):
            current_image_tensor_hwc = image[i]
            print(f"    Loop {i}: current_image_tensor_hwc shape (from image[{i}]): {current_image_tensor_hwc.shape}") # 新增日誌
            
            pil_image_to_draw_on = to_pil_image(current_image_tensor_hwc.permute(2, 0, 1)).convert("RGBA")
            img_width, img_height = pil_image_to_draw_on.size
            
            current_label_text = label_lines[i % num_provided_labels] if num_provided_labels > 0 else ""
            print(f"    Processing image {i} (Size: {img_width}x{img_height}, Mode: {pil_image_to_draw_on.mode}) with label: '{current_label_text}'")

            draw = ImageDraw.Draw(pil_image_to_draw_on)
            
            actual_text_width = 0
            actual_text_height = 0
            sized_font = None

            if current_label_text:
                current_font_size_iter = font_size
                min_font_size = 8
                text_bbox = (0,0,0,0) 

                while True:
                    sized_font = base_font_object.font_variant(size=current_font_size_iter)
                    try:
                        text_bbox = draw.multiline_textbbox((0,0), current_label_text, font=sized_font, spacing=line_spacing, align="center")
                    except TypeError: 
                        text_bbox = draw.multiline_textbbox((0,0), current_label_text, font=sized_font, spacing=line_spacing)
                    actual_text_width = text_bbox[2] - text_bbox[0]
                    if actual_text_width <= (img_width - margin * 2) or not current_label_text: break
                    if current_font_size_iter <= min_font_size:
                        try:
                            text_bbox = draw.multiline_textbbox((0,0), current_label_text, font=sized_font, spacing=line_spacing, align="center")
                        except TypeError:
                            text_bbox = draw.multiline_textbbox((0,0), current_label_text, font=sized_font, spacing=line_spacing)
                        actual_text_width = text_bbox[2] - text_bbox[0]
                        break
                    current_font_size_iter -= 1
                actual_text_height = text_bbox[3] - text_bbox[1]

                text_draw_x, text_draw_y = 0.0, 0.0
                final_anchor = "lt"
                if text_position == "bottom_center": text_draw_x, text_draw_y, final_anchor = img_width / 2, float(img_height - margin), "ms"
                elif text_position == "top_center": text_draw_x, text_draw_y, final_anchor = img_width / 2, float(margin), "mt"
                elif text_position == "bottom_left": text_draw_x, text_draw_y, final_anchor = float(margin), float(img_height - margin), "ls"
                elif text_position == "bottom_right": text_draw_x, text_draw_y, final_anchor = float(img_width - margin), float(img_height - margin), "rs"
                elif text_position == "top_left": text_draw_x, text_draw_y, final_anchor = float(margin), float(margin), "lt"
                elif text_position == "top_right": text_draw_x, text_draw_y, final_anchor = float(img_width - margin), float(margin), "rt"
                elif text_position == "center_center": text_draw_x, text_draw_y, final_anchor = img_width / 2, img_height / 2, "mm"

                # 使用正確的參數名 background_color_hex
                if background_color_hex.lower() != "none" and parsed_bg_color[3] > 0 and sized_font:
                    try:
                        final_text_pixel_bbox = draw.multiline_textbbox((text_draw_x, text_draw_y), current_label_text, font=sized_font, spacing=line_spacing, align="center", anchor=final_anchor)
                    except TypeError: 
                        temp_text_bbox_for_fallback = draw.multiline_textbbox((0,0), current_label_text, font=sized_font, spacing=line_spacing, align="center")
                        fb_actual_text_width = temp_text_bbox_for_fallback[2] - temp_text_bbox_for_fallback[0]
                        fb_actual_text_height = temp_text_bbox_for_fallback[3] - temp_text_bbox_for_fallback[1]
                        fb_x1, fb_y1 = text_draw_x, text_draw_y
                        if final_anchor[0] == 'm': fb_x1 -= fb_actual_text_width / 2
                        elif final_anchor[0] == 'r': fb_x1 -= fb_actual_text_width
                        if final_anchor[1] == 'm': fb_y1 -= fb_actual_text_height / 2
                        elif final_anchor[1] == 't': pass 
                        elif final_anchor[1] == 's' or final_anchor[1] == 'd': fb_y1 -= fb_actual_text_height
                        final_text_pixel_bbox = (fb_x1, fb_y1, fb_x1 + fb_actual_text_width, fb_y1 + fb_actual_text_height)
                        print(f"      Fallback bbox calculation for background: {final_text_pixel_bbox}")

                    bg_x1 = final_text_pixel_bbox[0] - background_padding # 使用 background_padding
                    bg_y1 = final_text_pixel_bbox[1] - background_padding # 使用 background_padding
                    bg_x2 = final_text_pixel_bbox[2] + background_padding # 使用 background_padding
                    bg_y2 = final_text_pixel_bbox[3] + background_padding # 使用 background_padding
                    
                    bg_x1,bg_y1,bg_x2,bg_y2 = max(0.0,bg_x1),max(0.0,bg_y1),min(float(img_width),bg_x2),min(float(img_height),bg_y2)
                    
                    if bg_x1 < bg_x2 and bg_y1 < bg_y2:
                        draw.rectangle([bg_x1, bg_y1, bg_x2, bg_y2], fill=parsed_bg_color)

                if sized_font:
                    draw.multiline_text(xy=(text_draw_x, text_draw_y), text=current_label_text, fill=parsed_text_color, font=sized_font, anchor=final_anchor, spacing=line_spacing, align="center")
            
            final_pil_image_rgb = pil_image_to_draw_on.convert("RGB")
            output_tensor_chw = to_image(final_pil_image_rgb) / 255.0
            processed_pil_images_chw.append(output_tensor_chw)
        
        if not processed_pil_images_chw:
            print("[AddLabelToImage WARNING] No images were processed after loop. Returning original batch or dummy.")
            return (image if image.numel() > 0 else torch.zeros((batch_size if batch_size > 0 else 1, 64, 64, 3), dtype=image.dtype if isinstance(image, torch.Tensor) else torch.float32, device=image.device if isinstance(image, torch.Tensor) else 'cpu'),)

        try:
            stacked_images_bchw = torch.stack(processed_pil_images_chw, dim=0)
            final_output_tensor_bhwc = stacked_images_bchw.permute(0, 2, 3, 1)
            print(f"  Batch processed successfully. Output shape: {final_output_tensor_bhwc.shape}")
            print(f"[AddLabelToImage EXECUTE_DRAW_ON_BATCH END]")
            return (final_output_tensor_bhwc,)
        except RuntimeError as e:
            print(f"[AddLabelToImage ERROR] Failed to stack processed images: {e}")
            print(f"  This means the processed images (after adding label) in 'processed_pil_images_chw' have different H or W dimensions.")
            for idx, t_info in enumerate(processed_pil_images_chw):
                print(f"    Processed image {idx} shape (CHW): {t_info.shape}")
            return (image,)

    # 移除 __init__ (如果之前有)
    # 移除 calculate_label_dimensions 和 draw_label (如果之前有，並且其邏輯已被整合)