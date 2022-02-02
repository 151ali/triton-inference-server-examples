"""
https://towardsdatascience.com/creating-editing-and-merging-onnx-pipelines-897e55e98bb0
https://github.com/scailable/sclblonnx/blob/master/examples/example_05.py
"""


import sclblonnx as so

onnx_file = "/home/lina/Desktop/TRITON/model_repository/yolov3/1/model.onnx"

# Open the graph. 
g = so.graph_from_file(onnx_file)

# rename onnx input and output
g = so.rename_input(g, "input.1", "input")

g = so.rename_output(g, "662", "output_13")
g = so.rename_output(g, "715", "output_26")
g = so.rename_output(g, "768", "output_52")


g = so.rename_output(g, "769", "detections")



so.graph_to_file(g, onnx_file+".out")
