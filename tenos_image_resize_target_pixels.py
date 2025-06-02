# --- START OF FILE tenos_image_resize_target_pixels.py ---
import math
import torch
import torch.nn.functional as F

class TenosResizeToTargetPixels:
    TARGET_TOTAL_PIXELS = 1_000_000
    INTERPOLATION_MODES = ["area", "bicubic", "bilinear", "nearest"]
    DIMENSION_MULTIPLE = 64

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "interpolation": (cls.INTERPOLATION_MODES,),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "execute"
    CATEGORY = "TenosNodes/Image Processing"

    @staticmethod
    def _round_to_multiple(value: float, multiple: int) -> int:
        return max(multiple, round(value / multiple) * multiple)

    @staticmethod
    def _calculate_target_dimensions(original_width: int, original_height: int,
                                    target_pixels: int = TARGET_TOTAL_PIXELS,
                                    dimension_multiple: int = DIMENSION_MULTIPLE) -> tuple[int, int]:
        if original_width <= 0 or original_height <= 0:
            return dimension_multiple, dimension_multiple

        original_aspect_ratio = float(original_width) / float(original_height)

        ideal_height_float = math.sqrt(target_pixels / original_aspect_ratio)
        ideal_width_float = ideal_height_float * original_aspect_ratio
        
        # Option 1: Round height first, then calculate width, then round both to multiple
        h1_initial_round = max(1, round(ideal_height_float))
        w1_from_h1 = max(1, round(h1_initial_round * original_aspect_ratio))
        h1_final = TenosResizeToTargetPixels._round_to_multiple(h1_initial_round, dimension_multiple)
        w1_final = TenosResizeToTargetPixels._round_to_multiple(w1_from_h1, dimension_multiple)
        # Recalculate one dimension based on the other's final rounded value to better preserve AR before final rounding of the second
        if abs(h1_initial_round - h1_final) < abs(w1_from_h1 - w1_final) : # Height was closer to its multiple
            w1_final = TenosResizeToTargetPixels._round_to_multiple(h1_final * original_aspect_ratio, dimension_multiple)
        else: # Width was closer or equal
            h1_final = TenosResizeToTargetPixels._round_to_multiple(w1_final / original_aspect_ratio, dimension_multiple)
        pixels1 = w1_final * h1_final
        diff1 = abs(pixels1 - target_pixels)

        # Option 2: Round width first, then calculate height, then round both to multiple
        w2_initial_round = max(1, round(ideal_width_float))
        h2_from_w2 = max(1, round(w2_initial_round / original_aspect_ratio))
        w2_final = TenosResizeToTargetPixels._round_to_multiple(w2_initial_round, dimension_multiple)
        h2_final = TenosResizeToTargetPixels._round_to_multiple(h2_from_w2, dimension_multiple)
        if abs(w2_initial_round - w2_final) < abs(h2_from_w2 - h2_final): # Width was closer to its multiple
            h2_final = TenosResizeToTargetPixels._round_to_multiple(w2_final / original_aspect_ratio, dimension_multiple)
        else: # Height was closer or equal
            w2_final = TenosResizeToTargetPixels._round_to_multiple(h2_final * original_aspect_ratio, dimension_multiple)
        pixels2 = w2_final * h2_final
        diff2 = abs(pixels2 - target_pixels)

        if diff1 <= diff2:
            return int(w1_final), int(h1_final)
        else:
            return int(w2_final), int(h2_final)


    def execute(self, image: torch.Tensor, interpolation: str):
        if not isinstance(image, torch.Tensor) or image.ndim != 4:
            return (image,)

        batch_size, original_height, original_width, num_channels = image.shape

        if batch_size == 0:
            return (image,)
        
        interpolation_mode_str = interpolation.lower().strip()
        if interpolation_mode_str not in self.INTERPOLATION_MODES:
            interpolation_mode_str = "bicubic"
        
        output_images_list = []

        for i in range(batch_size):
            current_image_hwc = image[i:i+1, ...]
            _, h, w, _ = current_image_hwc.shape

            target_w, target_h = self._calculate_target_dimensions(w, h, self.TARGET_TOTAL_PIXELS, self.DIMENSION_MULTIPLE)

            if target_w == w and target_h == h:
                output_images_list.append(current_image_hwc)
                continue

            current_image_nchw = current_image_hwc.permute(0, 3, 1, 2)

            align_corners_val = None
            if interpolation_mode_str != "nearest":
                align_corners_val = False
            
            antialias_param = {}
            if interpolation_mode_str in ["bilinear", "bicubic"]:
                if (target_h < h or target_w < w): # Only apply antialias if downscaling
                     antialias_param['antialias'] = True
            
            resized_image_nchw = F.interpolate(
                current_image_nchw,
                size=(target_h, target_w),
                mode=interpolation_mode_str,
                align_corners=align_corners_val,
                **antialias_param
            )
            
            resized_image_nhwc = resized_image_nchw.permute(0, 2, 3, 1)
            output_images_list.append(resized_image_nhwc)

        final_output_image = torch.cat(output_images_list, dim=0)
        return (final_output_image,)

NODE_CLASS_MAPPINGS = {
    "TenosResizeToTargetPixels": TenosResizeToTargetPixels
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "TenosResizeToTargetPixels": "Tenos Resize to ~1M Pixels"
}
# --- END OF FILE tenos_image_resize_target_pixels.py ---