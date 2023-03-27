import os
import sys
import traceback

import numpy as np
from PIL import Image
from basicsr.utils.download_util import load_file_from_url
from realesrgan import RealESRGANer

from modules.upscaler import Upscaler, UpscalerData
from modules.shared import cmd_opts, opts


class UpscalerRealESRGAN(Upscaler):
    def __init__(self, path):
        self.name = "RealESRGAN"
        self.user_path = path
        super().__init__()
        try:
            from basicsr.archs.rrdbnet_arch import RRDBNet
            from realesrgan import RealESRGANer
            from realesrgan.archs.srvgg_arch import SRVGGNetCompact
            self.enable = True
            self.scalers = []
            scalers = self.load_models(path)
            for scaler in scalers:
                if scaler.name in opts.realesrgan_enabled_models:
                    self.scalers.append(scaler)

        except Exception:
            print("Error importing Real-ESRGAN:", file=sys.stderr)
            print(traceback.format_exc(), file=sys.stderr)
            self.enable = False
            self.scalers = []

    def do_upscale(self, img, path):
        if not self.enable:
            return img

        info = self.load_model(path)
        if not os.path.exists(info.local_data_path):
            print("Unable to load RealESRGAN model: %s" % info.name)
            return img

        upsampler = RealESRGANer(
            scale=info.scale, # scale (int): Upsampling scale factor used in the networks. It is usually 2 or 4.
            model_path=info.local_data_path, # model_path (str): The path to the pretrained model. It can be urls (will first download it automatically).
            model=info.model(), # model (nn.Module): The defined network. Default: None.
            half=not cmd_opts.no_half and not cmd_opts.upcast_sampling, # half (float): Whether to use half precision during inference. Default: False.
            tile=opts.ESRGAN_tile, # tile (int): As too large images result in the out of GPU memory issue, so this tile option will first crop
                                    # input images into tiles, and then process each of them. Finally, they will be merged into one image.
                                    # 0 denotes for do not use tile. Default: 0.
            tile_pad=opts.ESRGAN_tile_overlap,  # tile_pad (int): The pad size for each tile, to remove border artifacts. Default: 10.
        )
        # np.array(img).shape = (512, 512, 3) w, h, channel
        upsampled, mode = upsampler.enhance(np.array(img), outscale=info.scale)
        image = Image.fromarray(upsampled, mode=mode)
        output_file = "./sample_test_output/resrgan_512_512_test.jpg"
        image.save(output_file)
        print(f"success")
        exit()

        return image

    def load_model(self, path):
        try:
            info = next(iter([scaler for scaler in self.scalers if scaler.data_path == path]), None)

            if info is None:
                print(f"Unable to find model info: {path}")
                return None

            info.local_data_path = load_file_from_url(url=info.data_path, model_dir=self.model_path, progress=True)
            return info
        except Exception as e:
            print(f"Error making Real-ESRGAN models list: {e}", file=sys.stderr)
            print(traceback.format_exc(), file=sys.stderr)
        return None

    def load_models(self, _):
        return get_realesrgan_models(self)


def get_realesrgan_models(scaler):
    try:
        from basicsr.archs.rrdbnet_arch import RRDBNet
        from realesrgan.archs.srvgg_arch import SRVGGNetCompact
        models = [
            UpscalerData(
                name="R-ESRGAN General 4xV3",
                path="https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-general-x4v3.pth",
                scale=4,
                upscaler=scaler,
                model=lambda: SRVGGNetCompact(num_in_ch=3, num_out_ch=3, num_feat=64, num_conv=32, upscale=4, act_type='prelu')
            ),
            UpscalerData(
                name="R-ESRGAN General WDN 4xV3",
                path="https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-general-wdn-x4v3.pth",
                scale=4,
                upscaler=scaler,
                model=lambda: SRVGGNetCompact(num_in_ch=3, num_out_ch=3, num_feat=64, num_conv=32, upscale=4, act_type='prelu')
            ),
            UpscalerData(
                name="R-ESRGAN AnimeVideo",
                path="https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-animevideov3.pth",
                scale=4,
                upscaler=scaler,
                model=lambda: SRVGGNetCompact(num_in_ch=3, num_out_ch=3, num_feat=64, num_conv=16, upscale=4, act_type='prelu')
            ),
            UpscalerData(
                name="R-ESRGAN 4x+",
                path="https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth",
                scale=4,
                upscaler=scaler,
                model=lambda: RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
            ),
            UpscalerData(
                name="R-ESRGAN 4x+ Anime6B",
                path="https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.2.4/RealESRGAN_x4plus_anime_6B.pth",
                scale=4,
                upscaler=scaler,
                model=lambda: RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=6, num_grow_ch=32, scale=4)
            ),
            UpscalerData(
                name="R-ESRGAN 2x+",
                path="https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth",
                scale=2,
                upscaler=scaler,
                model=lambda: RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=2)
            ),
        ]
        return models
    except Exception as e:
        print("Error making Real-ESRGAN models list:", file=sys.stderr)
        print(traceback.format_exc(), file=sys.stderr)
