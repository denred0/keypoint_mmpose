import numpy as np
from tqdm import tqdm

pytorch_file = "wholebody_res50_result.txt"
onnx_file = "results_onnx.txt"

with open(pytorch_file) as file:
    pytorch_lines = file.readlines()
    pytorch_lines = sorted([line.rstrip() for line in pytorch_lines])

with open(onnx_file) as file:
    onnx_lines = file.readlines()
    onnx_lines = sorted([line.rstrip() for line in onnx_lines])

dif_result = []
dif_result_bad = []

print(f"PyTorch count: {len(pytorch_lines)}")
print(f"ONNX count:    {len(onnx_lines)}")
print()

for pytorch in tqdm(pytorch_lines[:10000]):
    p_values = pytorch.split(" ")
    p_file = p_values[0]
    p_point = p_values[1]
    p_x = p_values[2]
    p_y = p_values[3]

    for onnx in onnx_lines[:10000]:
        o_values = onnx.split(" ")
        o_file = o_values[0]
        o_point = o_values[1]
        o_x = o_values[2]
        o_y = o_values[3]

        if o_file == p_file and o_point == p_point:
            dif_x = abs(int(p_x) - int(o_x))
            dif_y = abs(int(p_y) - int(o_y))

            dif_result.append([o_file, o_point, dif_x, dif_y])

            if dif_y > 15 or dif_y > 15:
                dif_result_bad.append([o_file, o_point, dif_x, dif_y])

dif_x_mean = np.mean([x[2] for x in dif_result])
dif_y_mean = np.mean([y[3] for y in dif_result])
dif_x_max = np.max([x[2] for x in dif_result])
dif_y_max = np.max([y[3] for y in dif_result])
dif_x_min = np.min([x[2] for x in dif_result])
dif_y_min = np.min([y[3] for y in dif_result])


print(f"X max difference: {str(dif_x_max)}")
print(f"X min difference: {str(dif_x_min)}")
print(f"X mean difference: {str(dif_x_mean)}")
print()
print(f"Y max difference: {str(dif_y_max)}")
print(f"Y min difference: {str(dif_y_min)}")
print(f"Y mean difference: {str(dif_y_mean)}")

with open("data/comparison/" + "results_dif_bad.txt", 'w') as f:
    for item in dif_result_bad:
        f.write("%s\n" % item)

with open("data/comparison/" + "results_dif.txt", 'w') as f:
    for item in dif_result:
        f.write("%s\n" % item)
