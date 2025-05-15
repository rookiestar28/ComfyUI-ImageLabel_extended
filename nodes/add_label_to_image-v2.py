# (導入和類定義頂部保持不變)
from math import ceil
from typing import Tuple, List

import torch
from torch import Tensor
from torchvision.transforms.v2.functional import to_pil_image, to_image
from PIL import Image, ImageDraw, ImageFont # 確保 ImageFont 被導入以備 PIL 内部使用
from PIL.ImageFont import FreeTypeFont

from ..font_manager import FontCollection

class AddLabelToImage:
    fonts = FontCollection()

    def __init__(self):
        self.collected_images_list: List[Tensor] = []
        self.cached_expected_batch_size: int = 0
        self.cached_labels_string: str = ""
        self.parsed_label_lines: List[str] = []
        print(f"[AddLabelToImage INFO] New instance or __init__ called.")

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
                "label": ("STRING", {"multiline": True, "default": "Label 1"}),
                "font": (font_names, {"default": default_font_for_ui}),
                # ---- 修改 INPUT_TYPES ----
                "text_position": (["bottom_center", "top_center", "bottom_left", "bottom_right", "top_left", "top_right", "center_center"], {"default": "bottom_center"}),
                "text_size": ("INT", {"default": 48, "min": 4, "max": 1024, "step": 1}),
                "margin": ("INT", {"default": 24, "min": 0, "max": 256, "step": 1}), # 改名為 margin，表示離邊緣的距離
                "line_spacing": ("INT", {"default": 5, "min": 0, "max": 128, "step": 1}),
                "text_color": ("STRING", {"default": "#ffffff"}),
                "text_background_color": ("STRING", {"default": "#00000080"}), # 文字背景顏色，可以帶 alpha (例如 #RRGGBBAA)
                "text_background_padding": ("INT", {"default": 5, "min": 0, "max": 50, "step": 1}), # 文字背景的額外padding
                "expected_batch_size": ("INT", {"default": 1, "min": 1, "max": 256, "step": 1}),
            },
            # "optional": { # 如果將背景設為可選
            #     "text_background_color": ("STRING", {"default": "None"}), # 允許 "None" 或透明色
            # }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "collect_and_draw_on_image" # 新函數名
    CATEGORY = "image/transform"

    def _parse_color_with_alpha(self, color_hex: str, default_alpha: int = 255) -> Tuple[int, int, int, int]:
        """將 #RRGGBB 或 #RRGGBBAA 格式的十六進位顏色轉換為 RGBA 元組。"""
        color_hex = color_hex.lstrip('#')
        if len(color_hex) == 6: # RGB
            r, g, b = tuple(int(color_hex[i:i+2], 16) for i in (0, 2, 4))
            return r, g, b, default_alpha
        elif len(color_hex) == 8: # RGBA
            r, g, b, a = tuple(int(color_hex[i:i+2], 16) for i in (0, 2, 4, 6))
            return r, g, b, a
        else: # 默認黑色完全不透明或根據情況拋出錯誤
            print(f"[AddLabelToImage WARNING] Invalid color format: '{color_hex}'. Using black.")
            return 0, 0, 0, default_alpha


    def collect_and_draw_on_image(
        self,
        image: Tensor, # 逐個到達的圖片
        label: str,
        font: str,
        text_position: str, # 新的文字位置參數
        text_size: int,
        margin: int,      # 新的邊距參數
        line_spacing: int,
        text_color: str,
        text_background_color: str, # 新的文字背景顏色參數
        text_background_padding: int,
        expected_batch_size: int,
    ):
        print(f"\n[AddLabelToImage EXECUTE START]")
        print(f"  Input image shape: {image.shape if isinstance(image, torch.Tensor) else type(image)}")
        print(f"  Expected batch size (input): {expected_batch_size}")
        # ... (其他日誌) ...

        # --- 重置收集器邏輯 (與上一版本相同) ---
        if self.cached_expected_batch_size != expected_batch_size or \
           self.cached_labels_string != label:
            print(f"  Parameters changed. Resetting accumulator.")
            self.collected_images_list = []
            self.cached_expected_batch_size = expected_batch_size
            self.cached_labels_string = label
            self.parsed_label_lines = [line.strip() for line in label.strip().split('\n') if line.strip()]
            print(f"  New batch initialized. Expecting: {self.cached_expected_batch_size} images. Parsed {len(self.parsed_label_lines)} labels.")

        if not isinstance(image, torch.Tensor) or image.ndim != 4 or image.shape[0] != 1:
            # ... (錯誤處理和返回 dummy_tensor 的邏輯不變) ...
            print(f"[AddLabelToImage WARNING] Unexpected image format. Shape: {image.shape if isinstance(image, torch.Tensor) else type(image)}. Returning dummy.")
            dummy_tensor = torch.zeros((1, 64, 64, 3), dtype=torch.float32, device='cpu')
            if isinstance(image, torch.Tensor) and image.device is not None: dummy_tensor = dummy_tensor.to(image.device)
            return (dummy_tensor,)


        single_image_hwc = image[0] # image shape is [1, H, W, C]
        self.collected_images_list.append(single_image_hwc)
        print(f"  Image appended. Collected: {len(self.collected_images_list)}/{self.cached_expected_batch_size}")

        if len(self.collected_images_list) >= self.cached_expected_batch_size:
            print(f"  Batch of {len(self.collected_images_list)} collected. Processing...")
            
            images_to_process_hwc = self.collected_images_list[:self.cached_expected_batch_size]
            # 注意：由於我們現在直接在原圖尺寸上操作，不再需要 stack 原始圖片來做尺寸統一
            # image_batch_to_process = torch.stack(images_to_process_hwc, dim=0) # 這行不再需要，因為我們逐個處理

            print(f"  Clearing collected_images_list as a full batch will be processed now.")
            self.collected_images_list = [] # 清空，為下一個批次做準備
            
            num_provided_labels = len(self.parsed_label_lines)
            processed_images_tensors_chw: List[Tensor] = [] # 存儲處理後的圖片 (CHW格式)

            if not self.parsed_label_lines:
                print("[AddLabelToImage WARNING] No label lines parsed. Outputting original images (as a batch).")
                # 如果沒有標籤，我們需要將收集到的 images_to_process_hwc (列表) stack 起來返回
                # 這裡仍然需要檢查尺寸一致性，或者返回一個錯誤/佔位符
                # 為了簡化，假設此時原始圖片尺寸一致 (因為用戶已在源頭固定)
                try:
                    original_batch_bhwc = torch.stack(images_to_process_hwc, dim=0)
                    return (original_batch_bhwc,)
                except RuntimeError as e: # 尺寸不一致
                    print(f"[AddLabelToImage ERROR] Collected original images have different sizes, cannot stack for passthrough: {e}")
                    dummy_batch = torch.zeros((expected_batch_size, 64, 64, 3), dtype=image.dtype, device=image.device)
                    return (dummy_batch,)


            try:
                base_font_object = self.fonts[font]
            except KeyError:
                print(f"[AddLabelToImage ERROR] Font '{font}' not found. Outputting original images (as a batch).")
                try:
                    original_batch_bhwc = torch.stack(images_to_process_hwc, dim=0)
                    return (original_batch_bhwc,)
                except RuntimeError as e:
                    print(f"[AddLabelToImage ERROR] Collected original images have different sizes, cannot stack for passthrough: {e}")
                    dummy_batch = torch.zeros((expected_batch_size, 64, 64, 3), dtype=image.dtype, device=image.device)
                    return (dummy_batch,)

            # 解析顏色
            parsed_text_color = self._parse_color_with_alpha(text_color, 255) # 文字顏色默認完全不透明
            parsed_bg_color = self._parse_color_with_alpha(text_background_color, 128) # 背景顏色默認半透明 (128/255)
            print(f"    Parsed text_background_color '{text_background_color}' to RGBA: {parsed_bg_color}")


            for i in range(len(images_to_process_hwc)): # 迭代收集到的圖片
                current_image_tensor_hwc = images_to_process_hwc[i]
                # 將 HWC Tensor 轉換為 PIL Image (RGBA以支持透明背景)
                # 我們將直接在其上繪製，所以要做一個副本
                pil_image_original = to_pil_image(current_image_tensor_hwc.permute(2, 0, 1)).convert("RGBA")
                img_width, img_height = pil_image_original.size
                print(f"    Processing image {i} (Size: {img_width}x{img_height}, Mode: {pil_image_original.mode}) with label: '{current_label_text}'")

                # 先賦值 current_label_text
                current_label_text = self.parsed_label_lines[i % num_provided_labels] if num_provided_labels > 0 else ""
                # 再打印日誌
                print(f"    Processing image {i} (Size: {img_width}x{img_height}, Mode: {pil_image_original.mode}) with label: '{current_label_text}'")

                # --- 文字繪製邏輯 ---
                draw = ImageDraw.Draw(pil_image_original) # 直接在原始圖片的 RGBA 副本上繪製

                # 1. 計算文字尺寸和最終使用的字型大小 (以適應圖片寬度減去邊距)
                # max_text_width = img_width - (margin * 2) # 文字可用的最大寬度
                # 我們不再使用舊的 calculate_label_dimensions，因為它計算的是額外條帶的高度
                # 我們需要一個新的計算邏輯，或者直接使用 Pillow 的 textbbox
                
                current_font_size_iter = text_size
                min_font_size = 8
                sized_font = None
                text_bbox = (0,0,0,0) # x1, y1, x2, y2

                # 嘗試自動縮小字型以適應圖片寬度 (減去邊距)
                while True:
                    sized_font = base_font_object.font_variant(size=current_font_size_iter)
                    # 使用 textbbox 獲取多行文字的邊界框
                    # xy=(0,0) 只是為了計算，實際位置後面再定
                    text_bbox = draw.multiline_textbbox((0,0), current_label_text, font=sized_font, spacing=line_spacing, align="center")
                    actual_text_width = text_bbox[2] - text_bbox[0]
                    
                    if actual_text_width <= (img_width - margin * 2): # 檢查是否在寬度限制內
                        break
                    if current_font_size_iter <= min_font_size:
                        # 如果已是最小字型，即使超出也用它 (或者可以選擇截斷文字等策略)
                        break
                    current_font_size_iter -= 1
                
                actual_text_width = text_bbox[2] - text_bbox[0]
                actual_text_height = text_bbox[3] - text_bbox[1]

                # 2. 計算文字的繪製位置 (x, y) 基於 text_position 和 margin
                # text_bbox[0] 和 text_bbox[1] 是相對於 (0,0) 的左上角偏移，通常是0或接近0
                # 我們需要的是文字塊的實際寬高 actual_text_width, actual_text_height

                x, y = 0, 0 # 左上角繪製點
                anchor = "la" # left-ascent (Pillow >= 9.0.0) 或 left-top

                if "left" in text_position:
                    x = margin
                    anchor = "la" # 左上角對齊 (考慮字型基線)
                elif "right" in text_position:
                    x = img_width - margin - actual_text_width
                    anchor = "ra" # 右上角對齊
                elif "center" in text_position: # 水平居中
                    x = (img_width - actual_text_width) / 2
                    anchor = "ma" # 中上對齊 (基於x的中心點)

                if "top" in text_position:
                    y = margin
                elif "bottom" in text_position:
                    y = img_height - margin - actual_text_height
                elif "center" in text_position: # 垂直居中
                    y = (img_height - actual_text_height) / 2
                
                # 修正 anchor 以便 multiline_text 更易於定位
                # Pillow multiline_text 的 anchor 參數比較直觀
                # (x,y) 是錨點的位置
                text_draw_x, text_draw_y = 0,0
                final_anchor = "lt" # 預設左上角

                if text_position == "bottom_center":
                    text_draw_x = img_width / 2
                    text_draw_y = img_height - margin - (actual_text_height / 2) # 文字塊的中心y
                    final_anchor = "mm" # 中中對齊
                elif text_position == "top_center":
                    text_draw_x = img_width / 2
                    text_draw_y = margin + (actual_text_height / 2)
                    final_anchor = "mm"
                elif text_position == "bottom_left":
                    text_draw_x = margin
                    text_draw_y = img_height - margin - actual_text_height # 文字塊的左下y -> 左上y
                    final_anchor = "ls" # left-baseline/bottom
                elif text_position == "bottom_right":
                    text_draw_x = img_width - margin
                    text_draw_y = img_height - margin - actual_text_height
                    final_anchor = "rs" # right-baseline/bottom
                elif text_position == "top_left":
                    text_draw_x = margin
                    text_draw_y = margin
                    final_anchor = "lt" # left-top
                elif text_position == "top_right":
                    text_draw_x = img_width - margin
                    text_draw_y = margin
                    final_anchor = "rt" # right-top
                elif text_position == "center_center":
                    text_draw_x = img_width / 2
                    text_draw_y = img_height / 2
                    final_anchor = "mm"

                # 3. (可選) 繪製文字背景
                if text_background_color.lower() != "none" and parsed_bg_color[3] > 0: #檢查透明度
                    # 背景框的位置應該是文字邊界框 text_bbox 再加上 text_background_padding
                    # text_bbox 是相對於 (0,0) 計算的，我們需要將其平移到最終的文字位置
                    # 為了簡化，我們先計算文字繪製後的實際邊界，然後畫背景
                    # 或者，直接計算背景框的位置和大小
                    bg_x1 = text_draw_x
                    bg_y1 = text_draw_y
                    # 這裡需要根據 final_anchor 調整背景框的實際左上角和右下角
                    # 例如，如果 anchor 是 "mm", text_draw_x,y 是中心點
                    # 背景框的左上角是 (text_draw_x - actual_text_width/2 - tbp, text_draw_y - actual_text_height/2 - tbp)
                    # 右下角是 (text_draw_x + actual_text_width/2 + tbp, text_draw_y + actual_text_height/2 + tbp)
                    # (tbp = text_background_padding)
                    # 為了簡化，我們直接在文字位置周圍畫一個矩形
                    # 注意：Pillow 的 multiline_text 的 anchor 很方便，但獲取精確繪製後的邊界來畫背景有點麻煩
                    # 一個簡單的辦法是先畫一個有 padding 的背景，再在上面畫文字（可能需要一個臨時 Image）
                    
                    # 讓我們嘗試直接計算背景框
                    # 假設 text_draw_x, text_draw_y 是文字塊的左上角 (如果 anchor 是 lt)
                    # (如果 anchor 是 mm，則它們是中心點)
                    # Pillow 的 textlength 可以獲取單行文字寬度，但多行比較麻煩
                    
                    # 簡化策略：先畫背景，再畫文字
                    # 背景矩形座標：(基於文字的 bbox，然後應用 anchor 的反向邏輯得到左上角，再加 padding)
                    # 這部分比較複雜，先跳過完美對齊的背景，專注於文字本身
                    # 簡易背景：以文字的 bounding_box 為基礎擴展
                    # (text_bbox[0]+text_draw_x-tbp, text_bbox[1]+text_draw_y-tbp, text_bbox[2]+text_draw_x+tbp, text_bbox[3]+text_draw_y+tbp)
                    # 這假設了 text_draw_x,y 是文字的左上角繪製起點 (當 anchor='lt')
                    # 如果用 multiline_text 的 anchor，定位更方便
                    # 先畫文字，再根據文字的實際佔用區域（可能需要估算或用更複雜的方法獲取）畫背景
                    # 為了實用，先畫一個基於文字 bbox 和 anchor 的估算背景
                    
                    # 估算文字最終位置的 bbox (基於 anchor)
                    est_x1, est_y1, est_x2, est_y2 = actual_text_width, actual_text_height, 0, 0
                    if final_anchor == "lt": est_x1, est_y1, est_x2, est_y2 = text_draw_x, text_draw_y, text_draw_x + actual_text_width, text_draw_y + actual_text_height
                    elif final_anchor == "ls": est_x1, est_y1, est_x2, est_y2 = text_draw_x, text_draw_y - actual_text_height, text_draw_x + actual_text_width, text_draw_y
                    elif final_anchor == "rt": est_x1, est_y1, est_x2, est_y2 = text_draw_x - actual_text_width, text_draw_y, text_draw_x, text_draw_y + actual_text_height
                    elif final_anchor == "rs": est_x1, est_y1, est_x2, est_y2 = text_draw_x - actual_text_width, text_draw_y - actual_text_height, text_draw_x, text_draw_y
                    elif final_anchor == "mm":
                        est_x1 = text_draw_x - actual_text_width / 2
                        est_y1 = text_draw_y - actual_text_height / 2
                        est_x2 = text_draw_x + actual_text_width / 2
                        est_y2 = text_draw_y + actual_text_height / 2
                    # 其他 anchor 情況...

                    bg_rect_x1 = est_x1 - text_background_padding
                    bg_rect_y1 = est_y1 - text_background_padding
                    bg_rect_x2 = est_x2 + text_background_padding
                    bg_rect_y2 = est_y2 + text_background_padding
                    
                    # 確保背景框在圖片範圍內 (可選)
                    bg_rect_x1 = max(0, bg_rect_x1)
                    bg_rect_y1 = max(0, bg_rect_y1)
                    bg_rect_x2 = min(img_width, bg_rect_x2)
                    bg_rect_y2 = min(img_height, bg_rect_y2)

                    if bg_rect_x1 < bg_rect_x2 and bg_rect_y1 < bg_rect_y2 : # 確保矩形有效
                        # 創建一個臨時的 ImageDraw 物件來畫帶透明度的背景，或者直接畫
                        # 如果 pil_image_original 是 RGBA，可以直接畫帶 alpha 的矩形
                        draw.rectangle(
                            [bg_rect_x1, bg_rect_y1, bg_rect_x2, bg_rect_y2],
                            fill=parsed_bg_color # RGBA 元組
                        )

                # 4. 繪製文字
                draw.multiline_text(
                    xy=(text_draw_x, text_draw_y),
                    text=current_label_text,
                    fill=parsed_text_color, # RGBA 元組 (alpha 通常是255)
                    font=sized_font,
                    anchor=final_anchor, # 使用計算好的錨點
                    spacing=line_spacing,
                    align="center" # 文字塊內部居中對齊
                )
                # --- 文字繪製結束 ---

                # 將處理後的 PIL Image (RGBA) 轉換回 Tensor (CHW, 0-1範圍)
                # 如果原圖是 RGB，我們需要確保輸出也是 RGB，或者下游能處理 RGBA
                # to_image 會保留通道數，但如果原圖是 RGB 傳入，這裡變 RGBA，再轉回可能要去掉 Alpha
                final_pil_image_rgb = pil_image_original.convert("RGB") # 轉換回 RGB
                output_tensor_chw = to_image(final_pil_image_rgb) / 255.0
                processed_images_tensors_chw.append(output_tensor_chw)
            
            # ... (後續的 stack 和 permute 邏輯，與上一版本處理批次後的部分相同) ...
            if not processed_images_tensors_chw: # 應該不會發生，除非 len(images_to_process_hwc) 是0
                print("[AddLabelToImage WARNING] Batch processing resulted in no images. Returning empty placeholder.")
                dummy_processed_batch = torch.zeros((len(images_to_process_hwc), 64, 64, 3), dtype=image.dtype, device=image.device) # 使用 image.dtype 和 device
                return (dummy_processed_batch,)

            stacked_images_bchw = torch.stack(processed_images_tensors_chw, dim=0)
            final_output_tensor_bhwc = stacked_images_bchw.permute(0, 2, 3, 1)
            
            print(f"  Batch processed successfully. Output shape: {final_output_tensor_bhwc.shape}")
            print(f"[AddLabelToImage EXECUTE END - Processed Batch]")
            return (final_output_tensor_bhwc,)
        else:
            # ... (批次未滿，輸出佔位符的邏輯不變) ...
            print(f"  Batch not full ({len(self.collected_images_list)}/{self.cached_expected_batch_size}). Passing through dummy image.")
            dummy_passthrough = torch.zeros_like(image) 
            print(f"[AddLabelToImage EXECUTE END - Accumulating]")
            return (dummy_passthrough,)

    # calculate_label_dimensions 方法不再需要，因為我們用 textbbox
    # draw_label 方法也不再需要，因為我們直接在原圖上繪製

# ... (可以保留 __main__ 用於非常獨立的測試，但現在節點複雜度增加了)