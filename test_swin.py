import os, sys, argparse, time
from copy import copy
from collections import OrderedDict
import onnx
import torch
# from extensions_builtin.SwinIR.swinir_model_arch_v2 import Swin2SR as net2
# from extensions_builtin.SwinIR.swinir_model_arch import SwinIR as net
import tensorrt as trt
from polygraphy.backend.common import bytes_from_path
from polygraphy.backend.trt import CreateConfig, Profile
from polygraphy.backend.trt import engine_from_bytes, engine_from_network, network_from_onnx_path, save_engine
from polygraphy.backend.trt import util as trt_util
from polygraphy import cuda
# import pycuda.autoinit
# import pycuda.driver as cuda

from cuda import cudart
import onnxruntime
import numpy as np

from PIL import Image
import numpy as np
import cv2
from torchvision import transforms
import torch
from torch.nn import functional as F
import ctypes
# import netron

G_LOGGER = trt.Logger(trt.Logger.WARNING)
# img_shape = (1,3,512,512)
# output_shape = (1,3,2048,2048)

pre_pad = 10

min_shape = (1,3,512,512)
optim_shap = (1,3,896,512)
max_shape = (1,3,912,528)

def parseArgs():
    parser = argparse.ArgumentParser(description="Options for testing swinir")
    parser.add_argument('--onnx_path', default = "./onnx_file/swinir_dynamic.onnx", help="ONNX model file")
    parser.add_argument('--engine_file_path', default = "./onnx_file/swinir_dynamic_v7.engine", help="TensorRT engine")
    parser.add_argument('--img_path', default = "./sample_test/sample_512_512.jpg", help="Input image file")
    parser.add_argument('--output_file', default = "./sample_test_output/swinir_512_512_outputx4.jpg", help="Output image file")
    return parser.parse_args()


# onnx_path = "./onnx_file/swinir_dynamic_v2_int32.onnx"
# engine_file_path = "./onnx_file/swinir_dynamic_v2_int32.engine"


def convert_onnx(onnx_path, swin_version = 1):
    if swin_version == 1:
        filename = "models/SwinIR/SwinIR_4x.pth"
    else:
        filename = "models/Swin2IR/SwinIR_4x.v2.pth"

    pretrained_model = torch.load(filename)

    if swin_version == 1:
        # the model has 28,013,059 parameters
        model = net(
                upscale=4,
                in_chans=3,
                img_size=64,
                window_size=8,
                img_range=1.0,
                depths=[6, 6, 6, 6, 6, 6, 6, 6, 6],
                embed_dim=240,
                num_heads=[8, 8, 8, 8, 8, 8, 8, 8, 8],
                mlp_ratio=2,
                upsampler="nearest+conv",
                resi_connection="3conv",
                )
        params = "params_ema"
        model.load_state_dict(pretrained_model[params], strict=True)
    else:
        model = net2(
                    upscale=4,
                    in_chans=3,
                    img_size=64,
                    window_size=8,
                    img_range=1.0,
                    depths=[6, 6, 6, 6, 6, 6],
                    embed_dim=180,
                    num_heads=[6, 6, 6, 6, 6, 6],
                    mlp_ratio=2,
                    upsampler="pixelshuffle",
                    #upsampler="nearest+conv",
                    resi_connection="1conv",
                    )
        model.load_state_dict(pretrained_model, strict=True)
        #onnx_path = "./onnx_file/swinirv2.onnx"

    model.to("cuda")
    # for p in model.parameters():
    #     print(type(p), p)
        
    # parameter: 28,013,059 (fp32)
    # total_num = sum(p.numel() for p in model.parameters())
    # print(total_num)


    onnx_opset = 11
    if not os.path.exists(onnx_path):
        print(f"Exporting model: {onnx_path}")
        with torch.inference_mode(), torch.autocast("cuda"):
            inputs = torch.zeros(img_shape, dtype=torch.float32, device="cuda")
            torch.onnx.export(model,
                    inputs,
                    onnx_path,
                    export_params=True,
                    opset_version=onnx_opset,
                    do_constant_folding=True,
                    input_names = ["input"],
                    output_names = ["output"],
                    dynamic_axes={"input":{0:"batch_size", 2:"image_height", 3:"image_width"},
                                  "output":{0:"batch_size", 2:"image_height_x", 3:"image_width_x"}},
            )
            print("Successfully export onnx model!")
    else:
        print(f"Found cached model: {onnx_path}")


def test_onnx_inference(onnx_path):
    img, max_range = pre_process(args.img_path)

    net = onnx.load(onnx_path)
    onnx.checker.check_model(net)
    onnx.helper.printable_graph(net.graph)

    session = onnxruntime.InferenceSession(onnx_path) 

    def to_numpy(tensor):
        return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

    # compute ONNX Runtime output prediction
    ort_inputs = {session.get_inputs()[0].name: to_numpy(img)}
    start = time.time()
    ort_outs = session.run(None, ort_inputs)
    print(f"inference time: {time.time()-start:.4f}s")
    ort_outs = torch.tensor(ort_outs[0])
    output_img = post_process(ort_outs, max_range)
    print(f"the output shape is {output_img.shape}")

    img = cv2.cvtColor(output_img, cv2.COLOR_BGR2RGB)
    cv2.imwrite(args.output_file, img)
    print("success!")



def ONNX2TRT(onnx_path, engine_file_path, calib=None):
    ''' convert onnx to tensorrt engine, use mode of 'fp16'
    :return: trt engine
    '''

    if os.path.exists(engine_file_path):
        print(f"TensorRT Engine already exists in {engine_file_path}")
        return
    else:
        with trt.Builder(G_LOGGER) as builder, \
            builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)) as network, \
            trt.OnnxParser(network, G_LOGGER) as parser: 

            print('Beginning ONNX file parsing')
            if not os.path.exists(onnx_path):
                sys.exit(f"ONNX file {onnx_path} not found")
            

            success = parser.parse_from_file(onnx_path)
            for idx in range(parser.num_errors):
                print(parser.get_error(idx))
            if not success:
                sys.exit("ONNX model parsing failed")
            print('Completed parsing of ONNX file')
    
            print('Building an engine from file {}; this may take a while...'.format(onnx_path))
            config = builder.create_builder_config()
            profile = builder.create_optimization_profile()
            
            profile.set_shape("input", min_shape, optim_shap, max_shape)
            config.add_optimization_profile(profile)

            config.max_workspace_size = 75 * (1 << 30)  # 75GB
            config.set_flag(trt.BuilderFlag.FP16)
            #config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 70 * (1 << 30))

            engine = builder.build_serialized_network(network, config)
            print("Created engine success! ")

            # save TRT engine
            print('Saving TRT engine file to path {}...'.format(engine_file_path))
            with open(engine_file_path, "wb") as f:
                f.write(engine)
            print('Engine file has already saved to {}!'.format(engine_file_path))
            return engine

def ONNX2TRT_polygraphy(onnx_path, engine_file_path):
    if not os.path.exists(engine_file_path):
        print(f"Building TensorRT engine...")
        p = Profile()
        p.add("input", min=min_shape, opt=optim_shap, max=max_shape)
        conf = CreateConfig(fp16=True, profiles=[p])

        engine = engine_from_network(network_from_onnx_path(onnx_path), config=conf)
        save_engine(engine, path=engine_file_path)
    else:
        print(f"TensorRT engine already exists in {engine_file_path}")

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

    img = img.unsqueeze(0)
    # img = img.unsqueeze(0).to("cuda")

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

    
def allocate_buffer(engine, context, input: torch.tensor, output_shape: tuple):
    # allocate_buffers
    tensors = OrderedDict()
    buffers = OrderedDict()

    for idx in range(trt_util.get_bindings_per_profile(engine)):
        # input output
        binding = engine[idx]

        # FLOAT32
        dtype = trt_util.np_dtype_from_trt(engine.get_binding_dtype(binding))

        if engine.binding_is_input(binding):
            context.set_binding_shape(idx, input.shape)

            tensors[binding] = input
            buffers[binding] = cuda.DeviceView(ptr=input.data_ptr(), shape=input.shape, dtype=dtype)
        else:
            # Workaround to convert np dtype to torch
            np_type_tensor = np.empty(shape=[], dtype=dtype)
            torch_type_tensor = torch.from_numpy(np_type_tensor)
            tensor = torch.empty(tuple(output_shape), dtype=torch_type_tensor.dtype).to(device="cuda")

            tensors[binding] = tensor
            buffers[binding] = cuda.DeviceView(ptr=tensor.data_ptr(), shape=output_shape, dtype=dtype)

    return tensors, buffers


def inference(engine_file_path, input):
    with torch.inference_mode(), torch.autocast("cuda"), trt.Runtime(G_LOGGER) as runtime:
        stream = cuda.Stream()
        # activate tensorrt engine
        engine = engine_from_bytes(bytes_from_path(engine_file_path))
        context = engine.create_execution_context()

        output_shape = tuple([1,3,4 * input.shape[2], 4 * input.shape[3]])

        # allocate buffers
        tensors, buffers = allocate_buffer(engine, context, input, output_shape)
        # print(f"Tensors input shape: {tensors['input'].shape}\ninput: {tensors['input']}")
        # print(f"Tensors output shape: {tensors['output'].shape}\noutput: {tensors['output']}")
        # print(f"Buffers input shape: {buffers['input'].shape}\ninput: {buffers['input']}")
        # print(f"Buffers output shape: {buffers['output'].shape}\noutput: {buffers['output']}")

        bindings =[buf.ptr for buf in buffers.values()]
        
        #cuda_wrapper = cuda.wrapper()

        # error = cuda_wrapper.memcpy(dst=tensors[engine[1]], src=buffers[engine[1]], nbytes = nbytes, kind = ctypes.c_int(2), stream_ptr=stream.ptr)
        # if error:
        #     print(f"Memcpy d2h failed.")

        e2e_tic = time.perf_counter()

        print(f"Starting inference...")
        noerror = context.execute_async_v2(bindings=bindings, stream_handle=stream.ptr)
        if not noerror:
            raise ValueError(f"ERROR: inference failed.")
            exit()

        # nbytes = 3 * 4 * input.shape[2] * 4 * input.shape[3] * 4  # fp32
        # error = cuda_wrapper.memcpy(dst=tensors[engine[1]], src=buffers[engine[1]], nbytes = nbytes, kind = ctypes.c_int(1), stream_ptr=stream.ptr)
        # if error:
        #     print(f"Memcpy h2d failed.")

        torch.cuda.synchronize()
        e2e_toc = time.perf_counter()
        print(f"Inference costs: {(e2e_toc-e2e_tic):.4f}")

    return tensors[engine[1]]

# def loadEngine2TensorRT(filepath):
#     # 反序列化引擎
#     with open(filepath, "rb") as f, trt.Runtime(G_LOGGER) as runtime:
#         engine = runtime.deserialize_cuda_engine(f.read())
#         return engine

# def do_inference(engine_file_path, input, output_shape):
#     # pycuda

#     engine = loadEngine2TensorRT(engine_file_path)

#     context = engine.create_execution_context()
#     output = np.empty(output_shape, dtype=np.float32)

#     # allocate buffer
#     input = np.array()
#     input_size = 1 * 3 * input.shape[2]*input.shape[3]

#     d_input = cuda.mem_alloc(1 * input_size * output.dtype.itemsize)
#     print(f"d_input = {d_input}")
#     d_output = cuda.mem_alloc(1 * input_size * 16 * output.dtype.itemsize)
#     bindings = [int(d_input), int(d_output)]
#     context.set_binding_shape(0, input.shape)

#     stream = cuda.Stream()
#     # 将输入数据放入device
#     cuda.memcpy_htod_async(d_input, input, stream)

#     start = time.time()
#     # 执行模型
#     context.execute_async_v2(bindings, stream.handle)
#     # 将预测结果从从缓冲区取出
#     cuda.memcpy_dtoh_async(output, d_output, stream)
#     end = time.time()

#     # 线程同步
#     stream.synchronize()

#     # print("output:", output)
#     print("time cost:", end - start)
#     return output

# def get_shape(engine):
#     for binding in engine:
#         print(f"binding is {binding}")
#         if engine.binding_is_input(binding):
#             input_shape = engine.get_binding_shape(binding)
#         else:
#             output_shape = engine.get_binding_shape(binding)
#     return input_shape, output_shape


if __name__ == "__main__":
    args = parseArgs()

    onnx_path = args.onnx_path
    engine_file_path = args.engine_file_path

    #convert_onnx(onnx_path, swin_version = 1)
    test_onnx_inference(onnx_path)
    # ONNX2TRT(onnx_path, engine_file_path)
    #ONNX2TRT_polygraphy(onnx_path, engine_file_path)

    # img, max_range = pre_process(args.img_path)

    # output = inference(engine_file_path, img)
    # print(f"output is {output}")

    # output_img = post_process(output, max_range)
    # print(f"the output shape is {output_img.shape}")

    # img = cv2.cvtColor(output_img, cv2.COLOR_BGR2RGB)
    # cv2.imwrite(args.output_file, img)
    

'''
    engine = loadEngine2TensorRT(engine_file_path)
    print(engine)
    #img = Image.open(args.img_path)
    input_shape = img_shape
    output_shape = tuple([1, 3, 4 * img_shape[2], 4 * img_shape[3]])
    
    print(f"input shape: {input_shape}, output shape: {output_shape}")

    dummy_img = np.random.random(input_shape)
    output = do_inference(engine, 1, dummy_img, output_shape)
    
    print(f"the shape of output is {output.shape}")
'''