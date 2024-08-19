import torch
from executorch.exir import to_edge, ExecutorchBackendConfig
from executorch.backends.xnnpack.partition.xnnpack_partitioner import XnnpackPartitioner
from torch.ao.quantization.quantize_pt2e import convert_pt2e, prepare_pt2e
from torch.ao.quantization.quantizer.xnnpack_quantizer import (
        get_symmetric_quantization_config,
        XNNPACKQuantizer,
)
from torch.export import export, Dim
from torch._export import capture_pre_autograd_graph
from depth_anything_v2.dpt import DepthAnythingV2
import cv2

class Normalize:
    def __init__(self, mean, std):
        self.mean = torch.tensor(mean).view(-1, 1, 1)
        self.std = torch.tensor(std).view(-1, 1, 1)

    def __call__(self, tensor):
        # Apply normalization: (tensor - mean) / std
        normalized_tensor = (tensor - self.mean) / self.std
        return normalized_tensor

class DynamicDepthAnything(torch.nn.Module):
    def __init__(self):
        super().__init__()
        model_configs = {
            'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
        }
        encoder = 'vits'

        model = DepthAnythingV2(**model_configs[encoder])
        model.load_state_dict(torch.load(f'checkpoints/depth_anything_v2_{encoder}.pth', map_location='cpu'))
        model = model.to('cpu')
        self.dpt = model
        self.normalize = Normalize(
            mean = [0.485, 0.456, 0.406],
            std = [0.229, 0.224, 0.225],
        )

    def forward(self, x):
        w, h = x.shape[1], x.shape[2]
        inputs = torch.permute(x, (0, 3, 1, 2))
        inputs = inputs / 255.0
        inputs = self.normalize(inputs)
        inputs = self.dpt(inputs)
        min_ = torch.min(inputs)
        inputs = ((inputs - min_) / (torch.max(inputs) - min_)) * 255
        inputs = inputs.type(torch.uint8)
        return torch.permute(inputs, [0 , 1, 2])

def main():
    model = DynamicDepthAnything()
    example_args = (torch.randn([1, 700, 518, 3]),)
    # dynamic_shapes = (
    #     {
    #         2: Dim("height", min=1, max=4096),
    #         3: Dim("width", min=1, max=4096),
    #     },
    # )

    pre_autograd_aten_dialect = capture_pre_autograd_graph(model, example_args)
    quantizer = XNNPACKQuantizer().set_global(get_symmetric_quantization_config())
    prepared_graph = prepare_pt2e(pre_autograd_aten_dialect, quantizer)
    converted_graph = convert_pt2e(prepared_graph)

    aten_dialect = export(converted_graph, example_args)
    # print("ATen Dialect Graph")
    # print(aten_dialect)

    # diff_input = torch.randn([1, 3, 800, 518])
    # aten_dialect.module()(diff_input)

    edge_program = to_edge(aten_dialect)
    # print("Edge Dialect Graph")
    # print(edge_program.exported_program())

    edge_program = edge_program.to_backend(XnnpackPartitioner())

    executorch_program = edge_program.to_executorch()

    with open("model.pte", "wb") as file:
        file.write(executorch_program.buffer)

if __name__ == '__main__':
    main()
