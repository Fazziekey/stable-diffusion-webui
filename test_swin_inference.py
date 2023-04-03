from collections import OrderedDict
import time
import numpy as np
import torch
import tensorrt as trt
from PIL import Image
from polygraphy.backend.trt import util as trt_util
from polygraphy.backend.trt import engine_from_bytes
from polygraphy.backend.common import bytes_from_path
from polygraphy import cuda

G_LOGGER = trt.Logger(trt.Logger.WARNING)
pre_pad = 10

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
        print(f"engine is {engine}")

        context = engine.create_execution_context()

        output_shape = tuple([1,3,4 * input.shape[2], 4 * input.shape[3]])

        # allocate buffers
        tensors, buffers = allocate_buffer(engine, context, input, output_shape)
        print(f"Tensors input shape: {tensors['input'].shape}\ninput: {tensors['input']}")
        print(f"Tensors output shape: {tensors['output'].shape}\noutput: {tensors['output']}")
        print(f"Buffers input shape: {buffers['input'].shape}\ninput: {buffers['input']}")
        print(f"Buffers output shape: {buffers['output'].shape}\noutput: {buffers['output']}")
        print(f"tensor is {tensors}, buffer is {buffers}")

        bindings =[buf.ptr for buf in buffers.values()]
        
        #cuda_wrapper = cuda.wrapper()

        # error = cuda_wrapper.memcpy(dst=tensors[engine[1]], src=buffers[engine[1]], nbytes = nbytes, kind = ctypes.c_int(2), stream_ptr=stream.ptr)
        # if error:
        #     print(f"Memcpy d2h failed.")

        torch.cuda.synchronize()
        print(f"Starting inference...")
        e2e_tic = time.perf_counter()
        noerror = context.execute_async_v2(bindings=bindings, stream_handle=stream.ptr)
        if not noerror:
            raise ValueError(f"ERROR: inference failed.")

        # nbytes = 3 * 4 * input.shape[2] * 4 * input.shape[3] * 4  # fp32
        # error = cuda_wrapper.memcpy(dst=tensors[engine[1]], src=buffers[engine[1]], nbytes = nbytes, kind = ctypes.c_int(1), stream_ptr=stream.ptr)
        # if error:
        #     print(f"Memcpy h2d failed.")

        torch.cuda.synchronize()
        e2e_toc = time.perf_counter()
        print(f"Inference costs: {(e2e_toc-e2e_tic):.4f}s")

    return tensors[engine[1]]


engine_file_path = "./onnx_file/swinir_dynamic_v7.engine"
input = torch.rand((1,3,512,512), dtype=torch.float32, device="cuda")
output = inference(engine_file_path, input)
print(output)