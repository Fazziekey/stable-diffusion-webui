import os, time
import torch
from basicsr.archs.rrdbnet_arch import RRDBNet
from torch2trt import torch2trt
import tensorrt as trt
from torch2trt import TRTModule

from PIL import Image
import numpy as np
import cv2
from torch.nn import functional as F
# from modules import images


img_shape = (1,3,256,256)
onnx_path = "./onnx_file/resrgan_dynamic.onnx"
tensorrt_file = "./onnx_file/resrgan_dynamic_v5.pth"
engine_file_path = "./onnx_file/resrgan_dynamic_v5.engine"

repeat_time = 1   # used for testing inference time

pre_pad = 10
image_file = "./sample_test/sample_512_512.jpg"
output_file = "./sample_test_output/resrgan_512_512_outputx4.jpg"

def pre_process(image_file, pre_pad=pre_pad):
    image = Image.open(image_file)
    image.convert("RGB")

    img = np.array(image)
    h_input, w_input = img.shape[0:2]

    # img: numpy
    img = img.astype(np.float32)
    if np.max(img) > 256:  # 16-bit image
        max_range = 65535
        print('\tInput is a 16-bit image')
    else:
        max_range = 255
    img = img / max_range

    img_mode = 'RGB'
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    img = torch.from_numpy(np.transpose(img, (2, 0, 1))).float()
    img = img.unsqueeze(0).to("cuda")

    # pre_pad
    if pre_pad != 0:
        img = F.pad(img, (0, pre_pad, 0, pre_pad), 'reflect')
    return img, max_range

def post_process(output, max_range, pre_pad=pre_pad, scale=4):
    # remove prepad
    if pre_pad != 0:
        _, _, h, w = output.size()
        output = output[:, :, 0:h - pre_pad * scale, 0:w - pre_pad * scale]
    
    output = output.data.squeeze().float().cpu().clamp_(0, 1).numpy()
    output = np.transpose(output[[2, 1, 0], :, :], (1, 2, 0))

    # ------------------------------ return ------------------------------ #
    if max_range == 65535:  # 16-bit image
        output = (output * 65535.0).round().astype(np.uint16)
    else:
        output = (output * 255.0).round().astype(np.uint8)

    return output  


def load_model():
    # load model
    filename = "models/RealESRGAN/RealESRGAN_x4plus.pth"

    pretrained_model = torch.load(filename)
    model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
    if 'params_ema' in pretrained_model:
        keyname = 'params_ema'
    else:
        keyname = 'params'

    model.load_state_dict(pretrained_model[keyname], strict=True)
    model.to("cuda")
    print("Loaded model!")
    return model
        
    # params dtype: fp32
    # total params: 16,697,987

    # for p in model.parameters():
    # print(p.dtype)
    # total_num = sum(p.numel() for p in model.parameters())
    # print(f"total params: {total_num}")

def get_tensorrt():
    if not os.path.exists(tensorrt_file):
        print(f"Start converting to tensor RT...")
        model = load_model()

        inputs = torch.zeros(img_shape, dtype=torch.float32, device="cuda")
        opt_shape_param = [
                (1, 3, 256, 256),   # min
                (1, 3, 512, 512),   # opt
                (1, 3, 976, 976)   # max
            ]
        # trt_model = torch2trt(model, [inputs], fp16_mode=True, log_level=trt.Logger.INFO, max_workspace_size=10 * (1 << 30))
        trt_model = torch2trt(model, [inputs], fp16_mode=True, log_level=trt.Logger.INFO, max_workspace_size=10 * (1 << 30),\
                              min_shapes=opt_shape_param[0],max_shapes=opt_shape_param[2], opt_shapes=opt_shape_param[1])
        print("Completing converting!")
        torch.save(trt_model.state_dict(), tensorrt_file)
        print("Saved pth file!")
        with open(engine_file_path, "wb") as f:
            f.write(trt_model.engine.serialize())
        print("Saved tensorrt engine!")
    else:
         print(f"TensorRT model already exists!")
    

def inference(use_tensorrt):
    img, max_range = pre_process(image_file)
    print(img.dtype)
    print(img)
    exit()

    if use_tensorrt:
        print(f"Using tensort RT...")
        model_trt = TRTModule()
        model_trt.load_state_dict(torch.load(tensorrt_file))
        model_trt.to("cuda")

        print(f"Start inference...")
        acc_time = 0
        for i in range(repeat_time):
            #dummy_input = torch.randn(img_shape, device="cuda")
            start = time.time()
            output = model_trt(img)
            inference_time = time.time() - start
            print(f"Inference costs: {inference_time:.4f}s")
            if i >= 5:
                acc_time += inference_time
        if repeat_time >= 5:
            print(f"Average inference time: {acc_time/(repeat_time-5):.4f}s")
    else:
        print(f"Not using tensort RT...")
        model = load_model()

        print(f"Start inference...")
        acc_time = 0
        for i in range(repeat_time):
            #dummy_input = torch.randn(img_shape, device="cuda")
            start = time.time()
            output = model(img)
            inference_time = time.time() - start

            if i != repeat_time-1:
                del output
            print(f"Inference costs: {inference_time:.4f}s")
            if i >= 5:
                acc_time += inference_time
        if repeat_time >= 5:
            print(f"Average inference time: {acc_time/(repeat_time-5):.4f}s")
    print(f"CUDA allocated: {torch.cuda.max_memory_allocated()/1024**3:.2f} GB, CUDA reserved:{torch.cuda.max_memory_reserved()/1024**3:.2f} GB")

    output_img = post_process(output, max_range)
    print(f"the output shape is {output_img.shape}")

    # save image

    # PIL
    img = Image.fromarray(output_img, mode='RGB')
    img.save(output_file)

    # # cv2
    # img = cv2.cvtColor(output_img, cv2.COLOR_BGR2RGB)
    # cv2.imwrite(output_file, img)
    

# get_tensorrt()
inference(True)
# inference(False)