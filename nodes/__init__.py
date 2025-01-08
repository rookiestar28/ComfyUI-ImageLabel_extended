from .add_label_to_image import AddLabelToImage

NODE_CLASS_MAPPINGS = {
    "gremlation:ComfyUI-ImageLabel:ImageLabel": AddLabelToImage,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "gremlation:ComfyUI-ImageLabel:ImageLabel": "Add Label to Image ~🅖",
}
