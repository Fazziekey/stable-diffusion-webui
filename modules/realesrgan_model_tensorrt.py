import cv2
import math
import numpy as np
import os
import queue
import threading
import torch
from basicsr.utils.download_util import load_file_from_url
import time
from modules.torch2trt_update import torch2trt
#from torch2trt import torch2trt
from torch.nn import functional as F
from realesrgan import RealESRGANer
from modules.shared import cmd_opts, opts
#import convert_tensorrt

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def get_trt_size(trt_model_file):
    if 'resrgan_dynamic_v5.pth' in trt_model_file:
        min_shape = (1, 3, 256, 256)
        max_shape = (1, 3, 976, 976)
    elif 'resrgan_dynamic_v4.pth' in trt_model_file:
        min_shape = (1, 3, 256, 256)
        max_shape = (4, 3, 976, 976)
    elif 'resrgan_dynamic_v3.pth' in trt_model_file:
        min_shape = (1, 3, 256, 256)
        max_shape = (16, 3, 1024, 1024)
    else:
        min_shape, max_shape = get_dynamic_shape()
    return min_shape, max_shape

def get_dynamic_shape():
    min_h = cmd_opts.min_h
    min_w = cmd_opts.min_w
    max_h = cmd_opts.max_h
    max_w = cmd_opts.max_w
    min_batch_size = cmd_opts.min_batch_size
    max_batch_size = cmd_opts.max_batch_size
    
    # the width and height of picture should be divisible by 8
    min_h -= min_h % 8
    min_w -= min_w % 8
    if max_h % 8 != 0:
        max_h -= max_h % 8 + 8 * 3
    else:
        max_h += 8 * 2
    if max_w % 8 != 0:
        max_w -= max_w % 8 + 8 * 3
    else:
        max_w += 8 * 2

    min_shape = (min_batch_size, 3, min_h, min_w)
    max_shape = (max_batch_size, 3, max_h, max_w)

    return min_shape, max_shape 

class RealESRGANerRT(RealESRGANer):
    def __init__(self,
                 scale,
                 model_path,
                 dni_weight=None,
                 model=None,
                 tile=0,
                 tile_pad=10,
                 pre_pad=10,
                 half=False,
                 device=None,
                 gpu_id=None):
        self.scale = scale
        self.tile_size = tile
        self.tile_pad = tile_pad
        self.pre_pad = pre_pad
        self.mod_scale = None
        self.half = half

        # initialize model
        if gpu_id:
            self.device = torch.device(
                f'cuda:{gpu_id}' if torch.cuda.is_available() else 'cpu') if device is None else device
        else:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') if device is None else device

        if isinstance(model_path, list):
            # dni
            assert len(model_path) == len(dni_weight), 'model_path and dni_weight should have the save length.'
            loadnet = self.dni(model_path[0], model_path[1], dni_weight)
        else:
            # if the model_path starts with https, it will first download models to the folder: weights
            if model_path.startswith('https://'):
                model_path = load_file_from_url(
                    url=model_path, model_dir=os.path.join(ROOT_DIR, 'weights'), progress=True, file_name=None)
            loadnet = torch.load(model_path, map_location=torch.device('cpu'))

        # prefer to use params_ema
        if 'params_ema' in loadnet:
            keyname = 'params_ema'
        else:
            keyname = 'params'
        model.load_state_dict(loadnet[keyname], strict=True)

        model.eval()
        self.model = model.to(self.device)
        if self.half:
            self.model = self.model.half()
        
        self.model_trt = None
    
    def get_trt_model(self, trt_model_file):
        if os.path.exists(trt_model_file):
            print(f"Find existing TensorRT model, begin loading...")
            from torch2trt import TRTModule
            model_trt = TRTModule()
            model_trt.load_state_dict(torch.load(trt_model_file))
            self.model_trt = model_trt.to(self.device)
        else:
            self.model_trt = self.convert_trt_model(trt_model_file)
        
    def convert_trt_model(self, trt_model_file, save=True):
        import tensorrt as trt
        print(f"Tensor RT model not found, start converting Tensor RT model...")

        min_shape, max_shape = get_dynamic_shape()
        opt_h = (min_shape[2] + max_shape[2]) // 2 - (min_shape[2] + max_shape[2]) % 8
        opt_w = (min_shape[3] + max_shape[3]) // 2 - (min_shape[3] + max_shape[3]) % 8
        opt_batch_size = (min_shape[0] + max_shape[0]) // 2 

        # dynamic tensorrt model
        opt_shape_param = [
                min_shape,
                (opt_batch_size, 3, opt_h, opt_w),
                max_shape
            ]

        inputs = torch.zeros(opt_shape_param[0], dtype=torch.float32, device=self.device)

        trt_model = torch2trt(self.model, [inputs], fp16_mode=True, log_level=trt.Logger.INFO, max_workspace_size=cmd_opts.max_workspace_size,\
                              min_shapes=opt_shape_param[0],max_shapes=opt_shape_param[2], opt_shapes=opt_shape_param[1])
        print("Convert completed.")

        if save:
            torch.save(trt_model.state_dict(), trt_model_file)
            print("Saved pth file!")
            with open(trt_model_file, "wb") as f:
                f.write(trt_model.engine.serialize())
            print("Saved tensorrt engine!")
    
    def trt_process(self):
        # the current tensor rt model only supports fp32 inputs
        self.img = self.img.float()
        self.output = self.model_trt(self.img)
    
    def tile_process(self):
        """It will first crop input images to tiles, and then process each tile.
        Finally, all the processed tiles are merged into one images.

        Modified from: https://github.com/ata4/esrgan-launcher
        """
        batch, channel, height, width = self.img.shape
        output_height = height * self.scale
        output_width = width * self.scale
        output_shape = (batch, channel, output_height, output_width)

        # start with black image
        self.output = self.img.new_zeros(output_shape)
        tiles_x = math.ceil(width / self.tile_size)
        tiles_y = math.ceil(height / self.tile_size)
        # loop over all tiles
        for y in range(tiles_y):
            for x in range(tiles_x):
                # extract tile from input image
                ofs_x = x * self.tile_size
                ofs_y = y * self.tile_size
                # input tile area on total image
                input_start_x = ofs_x
                input_end_x = min(ofs_x + self.tile_size, width)
                input_start_y = ofs_y
                input_end_y = min(ofs_y + self.tile_size, height)

                # input tile area on total image with padding
                input_start_x_pad = max(input_start_x - self.tile_pad, 0)
                input_end_x_pad = min(input_end_x + self.tile_pad, width)
                input_start_y_pad = max(input_start_y - self.tile_pad, 0)
                input_end_y_pad = min(input_end_y + self.tile_pad, height)

                # input tile dimensions
                input_tile_width = input_end_x - input_start_x
                input_tile_height = input_end_y - input_start_y
                tile_idx = y * tiles_x + x + 1
                input_tile = self.img[:, :, input_start_y_pad:input_end_y_pad, input_start_x_pad:input_end_x_pad]
                
                start = time.time()
                # upscale tile
                try:
                    with torch.no_grad():
                        min_shape, max_shape = get_trt_size(cmd_opts.realesrgan_trt)
                        if not cmd_opts.no_resrgan_trt and \
                            input_tile.shape[2] >= min_shape[2] and input_tile.shape[2] <= max_shape[2] \
                            and input_tile.shape[3] >= min_shape[3] and input_tile.shape[3] <= max_shape[3]: 
                            input_tile = input_tile.float()
                            output_tile = self.model_trt(input_tile)
                        else:
                            output_tile = self.model(input_tile)
                except RuntimeError as error:
                    print('Error', error)
                print(f'\tTile {tile_idx}/{tiles_x * tiles_y}, time: {time.time() - start :.4f}s')

                # output tile area on total image
                output_start_x = input_start_x * self.scale
                output_end_x = input_end_x * self.scale
                output_start_y = input_start_y * self.scale
                output_end_y = input_end_y * self.scale

                # output tile area without padding
                output_start_x_tile = (input_start_x - input_start_x_pad) * self.scale
                output_end_x_tile = output_start_x_tile + input_tile_width * self.scale
                output_start_y_tile = (input_start_y - input_start_y_pad) * self.scale
                output_end_y_tile = output_start_y_tile + input_tile_height * self.scale

                # put tile into output image
                self.output[:, :, output_start_y:output_end_y,
                            output_start_x:output_end_x] = output_tile[:, :, output_start_y_tile:output_end_y_tile,
                                                                       output_start_x_tile:output_end_x_tile]

    @torch.no_grad()
    def enhance(self, img, outscale=None, alpha_upsampler='realesrgan'):
        h_input, w_input = img.shape[0:2]
        # img: numpy
        img = img.astype(np.float32)
        if np.max(img) > 256:  # 16-bit image
            max_range = 65535
            print('\tInput is a 16-bit image')
        else:
            max_range = 255
        img = img / max_range
        if len(img.shape) == 2:  # gray image
            img_mode = 'L'
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        elif img.shape[2] == 4:  # RGBA image with alpha channel
            img_mode = 'RGBA'
            alpha = img[:, :, 3]
            img = img[:, :, 0:3]
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            if alpha_upsampler == 'realesrgan':
                alpha = cv2.cvtColor(alpha, cv2.COLOR_GRAY2RGB)
        else:
            img_mode = 'RGB'
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # ------------------- process image (without the alpha channel) ------------------- #
        self.pre_process(img)

        
        img_h, img_w = self.img.shape[2], self.img.shape[3]
        # the min and max input size supported by current Real-ESRGAN trt model
        trt_model_file = cmd_opts.realesrgan_trt
        min_shape, max_shape = get_trt_size(trt_model_file)

        if not cmd_opts.no_resrgan_trt and \
            (img_h >= min_shape[2] and img_h <= max_shape[2] and img_w >= min_shape[3] and img_w <= max_shape[3]):       
            print(f"Using Tensor RT to speed up inference...")
            # load tensor rt model
            self.get_trt_model(trt_model_file)
            start_time = time.time()
            self.trt_process()
            print(f"Inference using TensorRT costs: {(time.time() - start_time):.4f}s")

        else:
            if self.tile_size > 0:
                self.get_trt_model(trt_model_file)
                start_time = time.time()
                self.tile_process()
                print(f"Inference using tiles costs: {time.time() - start_time:.4f}s")
              
            else:
                start_time = time.time()
                self.process()
                print(f"Inference costs: {(time.time() - start_time):.4f}s")

        output_img = self.post_process()
        output_img = output_img.data.squeeze().float().cpu().clamp_(0, 1).numpy()
        output_img = np.transpose(output_img[[2, 1, 0], :, :], (1, 2, 0))

        if img_mode == 'L':
            output_img = cv2.cvtColor(output_img, cv2.COLOR_BGR2GRAY)

        # ------------------- process the alpha channel if necessary ------------------- #
        if img_mode == 'RGBA':
            if alpha_upsampler == 'realesrgan':
                self.pre_process(alpha)
                if self.tile_size > 0:
                    self.tile_process()
                else:
                    self.process()
                output_alpha = self.post_process()
                output_alpha = output_alpha.data.squeeze().float().cpu().clamp_(0, 1).numpy()
                output_alpha = np.transpose(output_alpha[[2, 1, 0], :, :], (1, 2, 0))
                output_alpha = cv2.cvtColor(output_alpha, cv2.COLOR_BGR2GRAY)
            else:  # use the cv2 resize for alpha channel
                h, w = alpha.shape[0:2]
                output_alpha = cv2.resize(alpha, (w * self.scale, h * self.scale), interpolation=cv2.INTER_LINEAR)

            # merge the alpha channel
            output_img = cv2.cvtColor(output_img, cv2.COLOR_BGR2BGRA)
            output_img[:, :, 3] = output_alpha

        # ------------------------------ return ------------------------------ #
        if max_range == 65535:  # 16-bit image
            output = (output_img * 65535.0).round().astype(np.uint16)
        else:
            output = (output_img * 255.0).round().astype(np.uint8)

        if outscale is not None and outscale != float(self.scale):
            output = cv2.resize(
                output, (
                    int(w_input * outscale),
                    int(h_input * outscale),
                ), interpolation=cv2.INTER_LANCZOS4)
        return output, img_mode
