import datetime

import pycuda.driver as cuda
import pycuda.autoinit

import tensorrt as trt
from tensorrt.parsers import uffparser
import uff
import numpy as np

import python.util.plot_utils as plot_utils
import python.util.helper as helper

output_name = "Generator_variables/gen_image"
uff_model = uff.from_tensorflow_frozen_model("output.pb", [output_name])

num_image_width = 4
num_image_height = 4
num_batch = num_image_width * num_image_height
num_cat = 64
shape_z = (256 - num_cat, 1, 1)
shape_cat = (num_cat, 1, 1)
num_iter = int(1e3)

G_LOGGER = trt.infer.ConsoleLogger(trt.infer.LogSeverity.ERROR)
parser = uffparser.create_uff_parser()
parser.register_input("Placeholder/real_z", shape_z, 0)
parser.register_input("Placeholder/input_cat", shape_cat, 1)
parser.register_output(output_name)
engine = trt.utils.uff_to_trt_engine(G_LOGGER, uff_model, parser, num_batch, 1 << 20)
parser.destroy()

runtime = trt.infer.create_infer_runtime(G_LOGGER)
context = engine.create_execution_context()

inp1 = helper.random_normal(0, 1, shape_z)
inp2 = np.zeros(shape_cat, dtype=np.float32)
output = np.empty((num_batch, 22 * 22 * 3), dtype=np.float32)

d_input1 = cuda.mem_alloc(inp1.nbytes * num_batch)
d_input2 = cuda.mem_alloc(inp2.nbytes * num_batch)
d_output = cuda.mem_alloc(output.nbytes)

bindings = [int(d_input1), int(d_input2), int(d_output)]

stream = cuda.Stream()

prr = plot_utils.Plot_Reproduce_Performance("result_rt", num_image_width, num_image_height, 22, 22, 1)

inp1_list = list()
inp2_list = list()
output_list = list()

for i in range(num_iter):
    final_inp1 = bytes()
    final_inp2 = bytes()

    for i in range(num_batch):
        inp1 = helper.random_normal(0, 1, shape_z)
        inp2 = np.zeros(shape_cat, dtype=np.float32)
        cat_index = np.random.randint(64)
        inp2[cat_index] = 1

        final_inp1 = final_inp1 + inp1.tobytes()
        final_inp2 = final_inp2 + inp2.tobytes()

    inp1_list.append(final_inp1)
    inp2_list.append(final_inp2)

begin_time = datetime.datetime.now()

for i in range(num_iter):
    inp1 = inp1_list[i]
    inp2 = inp2_list[i]

    cuda.memcpy_htod_async(d_input1, inp1, stream)
    cuda.memcpy_htod_async(d_input2, inp2, stream)
    context.enqueue(num_batch, bindings, stream.handle, None)
    cuda.memcpy_dtoh_async(output, d_output, stream)
    stream.synchronize()

    output_list.append(output.copy())

end_time = datetime.datetime.now()

diff = end_time - begin_time
diff = int(diff.total_seconds() * 1000)

print("Diff: %d" % diff)

for i, output in enumerate(output_list):
    prr.save_pngs(output, 3, "output%d.png" % i)

context.destroy()
engine.destroy()
runtime.destroy()
