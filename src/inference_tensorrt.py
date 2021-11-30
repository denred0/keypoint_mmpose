import numpy as np
import torch
import tensorrt as trt
import pycuda.autoinit
import pycuda.driver as cuda
import cv2

from onnx import ModelProto
from time import time


# albumentations_transform = A.Compose([
#     A.Resize(height=256, width=192),
#     A.Normalize(
#         mean=[0.485, 0.456, 0.406],
#         std=[0.229, 0.224, 0.225],
#     ),
#     ToTensorV2()
# ])


def preprocess_input(x, mean=None, std=None, input_space="RGB", input_range=None, **kwargs):
    if input_space == "BGR":  # Change colorspace, if needed
        x = x[..., ::-1].copy()

    if input_range is not None:  # Normalization
        if x.max() > 1 and input_range[1] == 1:
            x = x / 255.0

    if mean is not None:  # Substract mean value
        mean = np.array(mean)
        x = x - mean

    if std is not None:  # Supress deviation
        std = np.array(std)
        x = x / std

    x = np.moveaxis(x, -1, 0)

    return torch.from_numpy(x)


def main(input_path, output_path):
    batch_size = 1
    input_size = [192, 256]
    input_range = [0, 1]  # Parameter for preprocessing function
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    image_path = input_path
    image_orig = cv2.imread(image_path)
    h, w = image_orig.shape[:2]
    image_transformed = cv2.resize(image_orig, input_size, interpolation=cv2.INTER_NEAREST)
    image_transformed = preprocess_input(image_transformed, mean=mean, std=std, input_range=input_range)

    onnx_path = "mmpose_keypoint.onnx"
    engine_name = "mmpose_keypoint.plan"
    # convert_model_to_onnx(
    #     onnx_path=onnx_path, engine_name=engine_name, model=model_test, device=device, batch_size=batch_size)

    # TensorRT flow
    verbose = True
    TRT_LOGGER = trt.Logger(trt.Logger.WARNING) if verbose else trt.Logger()
    # trt_runtime = trt.Runtime(TRT_LOGGER)

    model = ModelProto()
    with open(onnx_path, "rb") as f:
        model.ParseFromString(f.read())

    d0 = model.graph.input[0].type.tensor_type.shape.dim[1].dim_value
    d1 = model.graph.input[0].type.tensor_type.shape.dim[2].dim_value
    d2 = model.graph.input[0].type.tensor_type.shape.dim[3].dim_value
    shape = [batch_size, d0, d1, d2]

    print('shape', shape)

    # builder = trt.Builder(TRT_LOGGER)

    engine = build_engine(TRT_LOGGER=TRT_LOGGER, onnx_path=onnx_path, shape=shape)
    save_engine(engine, engine_name)

    h_input, d_input, h_output, d_output, stream = allocate_buffers(engine)
    out_size = batch_size

    start = time()
    iterations = 1000
    for i in range(iterations):
        out = do_inference(engine, image_transformed, h_input, d_input,
                           h_output, d_output, stream, out_size)

    taken = time() - start
    print("FPS TensorRT: {:.1f} samples".format(iterations / taken))

    x = y = 0
    points = []
    for i, ou in enumerate(out):
        if i % 2 == 0:
            x = int(ou * w)
        else:
            y = int(ou * h)

        if x != 0 and y != 0:
            points.append([x, y])
            x = y = 0

    font = cv2.FONT_HERSHEY_SIMPLEX
    for i, p in enumerate(points):
        image_orig = cv2.circle(image_orig, (p[0], p[1]), 8, (0, 255, 0), thickness=-1)
        image_orig = cv2.putText(image_orig, str(i), (p[0], p[1]), font, 3, (0, 255, 0), 1)

    cv2.imwrite(output_path, image_orig)


def build_engine(TRT_LOGGER, onnx_path, shape, MAX_BATCH=100):
    with trt.Builder(TRT_LOGGER) as builder, builder.create_builder_config() as config, builder.create_network(
            1) as network, trt.OnnxParser(network, TRT_LOGGER) as parser:
        builder.fp16_mode = True
        builder.max_workspace_size = 1 << 30  # 1GB
        builder.max_batch_size = MAX_BATCH
        profile = builder.create_optimization_profile()
        config.max_workspace_size = (3072 << 20)
        config.add_optimization_profile(profile)
        with open(onnx_path, 'rb') as model:
            parser.parse(model.read())
        network.get_input(0).shape = shape
        engine = builder.build_cuda_engine(network)
        return engine


def save_engine(engine, file_name):
    buf = engine.serialize()
    with open(file_name, 'wb') as f:
        f.write(buf)


def load_engine(trt_runtime, engine_path):
    with open(engine_path, 'rb') as f:
        engine_data = f.read()
    engine = trt_runtime.deserialize_cuda_engine(engine_data)
    return engine


# def convert_model_to_onnx(onnx_path, engine_name, model, device, batch_size):
#     input_shape = (batch_size, 3, 299, 299)
#     inputs = torch.ones(*input_shape)
#     inputs = inputs.to(device)
#     torch.onnx.export(model, inputs, onnx_path,
#                       input_names=None, output_names=None, dynamic_axes=None)


def load_images_to_buffer(pics, pagelocked_buffer):
    preprocessed = np.asarray(pics).ravel()

    # last batch. Check if count of images is less than batch_size
    if pagelocked_buffer.size > preprocessed.size:
        pagelocked_buffer = np.zeros(preprocessed.size)

    np.copyto(pagelocked_buffer, preprocessed)


def do_inference(engine, pics_1, h_input_1, d_input_1, h_output, d_output, stream, batch_size):
    load_images_to_buffer(pics_1, h_input_1)

    with engine.create_execution_context() as context:
        # Transfer input data to the GPU.
        cuda.memcpy_htod_async(d_input_1, h_input_1, stream)

        # Run inference.
        context.profiler = trt.Profiler()
        context.execute(batch_size=batch_size, bindings=[
            int(d_input_1), int(d_output)])

        # Transfer predictions back from the GPU.
        cuda.memcpy_dtoh_async(h_output, d_output, stream)
        # Synchronize the stream
        stream.synchronize()
        # Return the host output.
        out = (h_output)
        return out


def allocate_buffers(engine):
    # host cpu mem
    h_in_size = trt.volume(engine.get_binding_shape(0))
    h_out_size = trt.volume(engine.get_binding_shape(1))

    h_in_dtype = trt.nptype(engine.get_binding_dtype(0))
    h_out_dtype = trt.nptype(engine.get_binding_dtype(1))

    h_input = cuda.pagelocked_empty(h_in_size, h_in_dtype)
    h_output = cuda.pagelocked_empty(h_out_size, h_out_dtype)

    # allocate gpu mem
    d_input = cuda.mem_alloc(h_input.nbytes)
    d_output = cuda.mem_alloc(h_output.nbytes)
    stream = cuda.Stream()

    return h_input, d_input, h_output, d_output, stream


if __name__ == '__main__':
    input_path = "data/inference/tensorrt/input/img_2_box.png"
    output_path = "data/inference/tensorrt/output/img_2_box.png"
    main(input_path, output_path)
