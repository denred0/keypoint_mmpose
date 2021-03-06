# Copyright (c) OpenMMLab. All rights reserved.
import argparse

import numpy as np
import torch
import torch._C as _C

from mmpose.apis import init_pose_model

try:
    import onnx
    import onnxruntime as rt
except ImportError as e:
    raise ImportError(f'Please install onnx and onnxruntime first. {e}')

try:
    from mmcv.onnx.symbolic import register_extra_symbolics
except ModuleNotFoundError:
    raise NotImplementedError('please update mmcv to version>=1.0.4')


def _convert_batchnorm(module):
    """Convert the syncBNs into normal BN3ds."""
    module_output = module
    if isinstance(module, torch.nn.SyncBatchNorm):
        module_output = torch.nn.BatchNorm3d(module.num_features, module.eps,
                                             module.momentum, module.affine,
                                             module.track_running_stats)
        if module.affine:
            module_output.weight.data = module.weight.data.clone().detach()
            module_output.bias.data = module.bias.data.clone().detach()
            # keep requires_grad unchanged
            module_output.weight.requires_grad = module.weight.requires_grad
            module_output.bias.requires_grad = module.bias.requires_grad
        module_output.running_mean = module.running_mean
        module_output.running_var = module.running_var
        module_output.num_batches_tracked = module.num_batches_tracked
    for name, child in module.named_children():
        module_output.add_module(name, _convert_batchnorm(child))
    del module
    return module_output


def pytorch2onnx(model,
                 input_shape,
                 opset_version=11,
                 show=False,
                 output_file='tmp.onnx',
                 verify=False):
    """Convert pytorch model to onnx model.

    Args:
        model (:obj:`nn.Module`): The pytorch model to be exported.
        input_shape (tuple[int]): The input tensor shape of the model.
        opset_version (int): Opset version of onnx used. Default: 11.
        show (bool): Determines whether to print the onnx model architecture.
            Default: False.
        output_file (str): Output onnx model name. Default: 'tmp.onnx'.
        verify (bool): Determines whether to verify the onnx model.
            Default: False.
    """
    model.cpu().eval()

    one_img = torch.randn(input_shape)

    register_extra_symbolics(opset_version)
    # TrainingMode = _C._onnx.TrainingMode

    torch.onnx.export(
        model,
        one_img,
        output_file,
        export_params=True,
        keep_initializers_as_inputs=True,
        verbose=show,
        opset_version=opset_version,
        # input_names=["x"],
        # dynamic_axes={
        #     # dict value: manually named axes
        #     "x": [0]
        # }
    )

    print(f'Successfully exported ONNX model: {output_file}')
    if verify:
        # check by onnx
        onnx_model = onnx.load(output_file)
        onnx.checker.check_model(onnx_model)

        # check the numerical value
        # get pytorch output
        pytorch_results = model(one_img)
        if not isinstance(pytorch_results, (list, tuple)):
            assert isinstance(pytorch_results, torch.Tensor)
            pytorch_results = [pytorch_results]

        # get onnx output
        input_all = [node.name for node in onnx_model.graph.input]
        input_initializer = [
            node.name for node in onnx_model.graph.initializer
        ]
        net_feed_input = list(set(input_all) - set(input_initializer))
        assert len(net_feed_input) == 1
        sess = rt.InferenceSession(output_file)
        onnx_results = sess.run(None,
                                {net_feed_input[0]: one_img.detach().numpy()})

        # compare results
        assert len(pytorch_results) == len(onnx_results)
        for pt_result, onnx_result in zip(pytorch_results, onnx_results):
            assert np.allclose(
                pt_result.detach().cpu(), onnx_result, atol=1.e-5
            ), 'The outputs are different between Pytorch and ONNX'
        print('The numerical values are same between Pytorch and ONNX')


# def parse_args():
#     parser = argparse.ArgumentParser(
#         description='Convert MMPose models to ONNX')
#     parser.add_argument('config', help='test config file path')
#     parser.add_argument('checkpoint', help='checkpoint file')
#     parser.add_argument('--show', action='store_true', help='show onnx graph')
#     parser.add_argument('--output-file', type=str, default='tmp.onnx')
#     parser.add_argument('--opset-version', type=int, default=11)
#     parser.add_argument(
#         '--verify',
#         action='store_true',
#         help='verify the onnx model output against pytorch output')
#     parser.add_argument(
#         '--shape',
#         type=int,
#         nargs='+',
#         default=[1, 3, 256, 192],
#         help='input size')
#     args = parser.parse_args()
#     return args


if __name__ == '__main__':

    # config = "configs_mmpose/body/2d_kpt_sview_rgb_img/deeppose/coco/res50_coco_256x192.py"
    # checkpoint = "pretrained_weights/mmpose/body/deeppose_res50_coco_256x192-f6de6c0e_20210205.pth"

    # config = "mmpose/configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/hrnet_w48_coco_384x288_dark.py"
    # checkpoint = "pretrained_weights/mmpose/body/hrnet_w48_coco_384x288_dark-e881a4b6_20210203.pth"

    # config = "mmpose/configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/2xmspn50_coco_256x192.py"
    # checkpoint = "pretrained_weights/mmpose/body/2xmspn50_coco_256x192-c8765a5c_20201123.pth"

    config = "mmpose/configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/hrnet_w32_coco_256x192_udp.py"
    checkpoint = "pretrained_weights/mmpose/body/hrnet_w32_coco_256x192_udp-aba0be42_20210220.pth"

    # shape = torch.randn(1, 3, 640, 480, requires_grad=True)

    # data_cfg = dict(image_size=[192, 256]) - ?????? ???????????? ?? ????????????
    # shape ???????? ???????????? ???????? ?? ?????????????? (batch, channels, weight,  height)
    # res50_coco_wholebody_256x192.py
    shape = (1, 3, 256, 192)  # (1, 3, 256, 192)
    batch = 1
    # shape = (batch, 3, 384, 288)  # (1, 3, 256, 192)

    model_type = "body"
    model_name = config.split("/")[len(config.split("/")) - 1].split(".")[0]
    name = f"{model_type}_topdownheatmap_batch_{batch}_{model_name}.onnx"
    # name = "body_deeppose_res50_coco_256x192.onnx"

    model = init_pose_model(config, checkpoint, device='cpu')
    model = _convert_batchnorm(model)

    # onnx.export does not support kwargs
    if hasattr(model, 'forward_dummy'):
        model.forward = model.forward_dummy
    else:
        raise NotImplementedError(
            'Please implement the forward method for exporting.')

    # convert model to onnx file
    pytorch2onnx(
        model,
        shape,
        opset_version=11,
        show=False,
        output_file="data/onnx_export/" + name,
        # output_file=name,
        verify=True)
