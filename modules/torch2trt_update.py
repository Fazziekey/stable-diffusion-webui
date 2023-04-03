import io
import tensorrt as trt
import torch
from torch2trt.dataset_calibrator import (
    DatasetCalibrator,
    DEFAULT_CALIBRATION_ALGORITHM,
)
from torch2trt.dataset import (
    Dataset,
    TensorBatchDataset,
    ListDataset
)
from torch2trt.flattener import Flattener
from torch2trt.flatten_module import Flatten, Unflatten
from torch2trt import infer_dynamic_axes, default_input_names, default_output_names, ConversionContext, TRTModule

def torch2trt(module,
              inputs,
              input_names=None,
              output_names=None,
              log_level=trt.Logger.ERROR,
              fp16_mode=False,
              max_workspace_size=1<<25,
              strict_type_constraints=False,
              keep_network=True,
              int8_mode=False,
              int8_calib_dataset=None,
              int8_calib_algorithm=DEFAULT_CALIBRATION_ALGORITHM,
              use_onnx=False,
              default_device_type=trt.DeviceType.GPU,
              dla_core=0,
              gpu_fallback=True,
              device_types={},
              min_shapes=None,
              max_shapes=None,
              opt_shapes=None,
              onnx_opset=None,
              max_batch_size=None,
              **kwargs):

    # capture arguments to provide to context
    kwargs.update(locals())
    kwargs.pop('kwargs')
    # print(f"[0] success, CUDA allocated: {torch.cuda.max_memory_allocated()/1024**3:.2f} GB, CUDA reserved:{torch.cuda.max_memory_reserved()/1024**3:.2f} GB")

    # handle inputs as dataset of list of tensors
    if issubclass(inputs.__class__, Dataset):
        dataset = inputs
        if len(dataset) == 0:
            raise ValueError('Dataset must have at least one element to use for inference.')
        inputs = dataset[0]
    else:
        dataset = ListDataset()
        dataset.insert(inputs)
        inputs = dataset[0]

    outputs = module(*inputs)
    input_flattener = Flattener.from_value(inputs)
    output_flattener = Flattener.from_value(outputs)

    # infer default parameters from dataset

    if min_shapes == None:
        min_shapes_flat = [tuple(t) for t in dataset.min_shapes(flat=True)]

    else:
        # min_shapes_flat = input_flattener.flatten(min_shapes)
        min_shapes_flat = [min_shapes]

    if max_shapes == None:
        max_shapes_flat = [tuple(t) for t in dataset.max_shapes(flat=True)]
    else:
        # max_shapes_flat = input_flattener.flatten(max_shapes)
        max_shapes_flat = [max_shapes]
    
    if opt_shapes == None:
        opt_shapes_flat = [tuple(t) for t in dataset.median_numel_shapes(flat=True)]
    else:
        # opt_shapes_flat = input_flattener.flatten(opt_shapes)
        opt_shapes_flat = [opt_shapes]

    # handle legacy max_batch_size
    if max_batch_size is not None:
        min_shapes_flat = [(1,) + s[1:] for s in min_shapes_flat]
        max_shapes_flat = [(max_batch_size,) + s[1:] for s in max_shapes_flat]

    dynamic_axes_flat = infer_dynamic_axes(min_shapes_flat, max_shapes_flat)
    
    if default_device_type == trt.DeviceType.DLA:
        for value in dynamic_axes_flat:
            if len(value) > 0:
                raise ValueError('Dataset cannot have multiple shapes when using DLA')

    logger = trt.Logger(log_level)
    builder = trt.Builder(logger)
    config = builder.create_builder_config()

    if input_names is None:
        input_names = default_input_names(input_flattener.size)
    if output_names is None:
        output_names = default_output_names(output_flattener.size)

    if use_onnx:
        import onnx_graphsurgeon as gs
        import onnx
        
        module_flat = Flatten(module, input_flattener, output_flattener)
        inputs_flat = input_flattener.flatten(inputs)

        f = io.BytesIO()
        torch.onnx.export(
            module_flat, 
            inputs_flat, 
            f, 
            input_names=input_names, 
            output_names=output_names,
            dynamic_axes={
                name: {int(axis): 'axis_%d' % axis for axis in dynamic_axes_flat[index]}
                for index, name in enumerate(input_names)
            },
            opset_version=onnx_opset
        )
        f.seek(0)
        
        onnx_graph = gs.import_onnx(onnx.load(f))
        onnx_graph.fold_constants().cleanup()


        f = io.BytesIO()
        onnx.save(gs.export_onnx(onnx_graph), f)
        f.seek(0)

        onnx_bytes = f.read()
        network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
        parser = trt.OnnxParser(network, logger)
        parser.parse(onnx_bytes)

    else:
        network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
        
        with ConversionContext(network, torch2trt_kwargs=kwargs, builder_config=config, logger=logger) as ctx:
            
            inputs_flat = input_flattener.flatten(inputs)

            ctx.add_inputs(inputs_flat, input_names, dynamic_axes=dynamic_axes_flat)    

            outputs = module(*inputs)

            outputs_flat = output_flattener.flatten(outputs)
            
            ctx.mark_outputs(outputs_flat, output_names)

    # set max workspace size
    config.max_workspace_size = max_workspace_size

    if fp16_mode:
        config.set_flag(trt.BuilderFlag.FP16)

    config.default_device_type = default_device_type
    if gpu_fallback:
        config.set_flag(trt.BuilderFlag.GPU_FALLBACK)
    config.DLA_core = dla_core
    
    if strict_type_constraints:
        config.set_flag(trt.BuilderFlag.STRICT_TYPES)

    if int8_mode:

        # default to use input tensors for calibration
        if int8_calib_dataset is None:
            int8_calib_dataset = dataset

        config.set_flag(trt.BuilderFlag.INT8)

        #Making sure not to run calibration with QAT mode on
        if not 'qat_mode' in kwargs:
            calibrator = DatasetCalibrator(
                int8_calib_dataset, algorithm=int8_calib_algorithm
            )
            config.int8_calibrator = calibrator

    # OPTIMIZATION PROFILE
    profile = builder.create_optimization_profile()
    for index, name in enumerate(input_names):
        profile.set_shape(
            name,
            min_shapes_flat[index],
            opt_shapes_flat[index],
            max_shapes_flat[index]
        )
    config.add_optimization_profile(profile)

    if int8_mode:
        config.set_calibration_profile(profile)

    # BUILD ENGINE
    print(f"Begin building engine. Now, CUDA allocated: {torch.cuda.max_memory_allocated()/1024**3:.2f} GB, CUDA reserved:{torch.cuda.max_memory_reserved()/1024**3:.2f} GB")
    
    engine = builder.build_engine(network, config)
    # print(f"[3] success, CUDA allocated: {torch.cuda.max_memory_allocated()/1024**3:.2f} GB, CUDA reserved:{torch.cuda.max_memory_reserved()/1024**3:.2f} GB")

    module_trt = TRTModule(engine, input_names, output_names, input_flattener=input_flattener, output_flattener=output_flattener)
    # print(f"[4] success, CUDA allocated: {torch.cuda.max_memory_allocated()/1024**3:.2f} GB, CUDA reserved:{torch.cuda.max_memory_reserved()/1024**3:.2f} GB")

    if keep_network:
        module_trt.network = network

    return module_trt