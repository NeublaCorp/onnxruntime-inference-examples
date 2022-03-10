import os
import sys
import numpy as np
import re
import abc
import subprocess
import json
import argparse
import time
from PIL import Image

import onnx
import onnxruntime
from onnx import helper, TensorProto, numpy_helper
from onnxruntime.quantization import quantize_static, CalibrationDataReader, QuantFormat, QuantType, CalibrationMethod
from data_reader_v2 import YoloV3DataReader as DataReader
from evaluate_v2 import YoloV3Evaluator as Evaluator
import pdb


def benchmark(model_path):

    session = onnxruntime.InferenceSession(model_path, providers=['CPUExecutionProvider'])
    input_name = session.get_inputs()[0].name

    total = 0.0
    runs = 10
    input_data = np.random.rand((1, 3, 416, 416), np.float32)
    #input_data = np.zeros((1, 3, 416, 416), np.float32)
    # Warming up
    _ = session.run([], {input_name: input_data})
    for i in range(runs):
        start = time.perf_counter()
        _ = session.run([], {input_name: input_data})
        end = (time.perf_counter() - start) * 1000
        total += end
        print(f"{end:.2f}ms")
    total /= runs
    print(f"Avg: {total:.2f}ms")


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_model", required=True, help="input model")
    parser.add_argument("--output_model", required=True, help="output model")
    parser.add_argument("--calibrate_dataset", default="./calib", help="calibration data set")
    parser.add_argument("--valid_dataset", default="./val2017", help="validation data set")
    parser.add_argument("--quant_format",
                        default=QuantFormat.QOperator,
                        type=QuantFormat.from_string,
                        choices=list(QuantFormat))
    parser.add_argument("--per_channel", default=True, type=bool)
    args = parser.parse_args()
    return args

def get_prediction_evaluation(model_path, validation_dataset, providers):
    data_reader = DataReader(validation_dataset,
                                   stride=1000,
                                   batch_size=1,
                                   model_path=model_path,
                                   is_evaluation=True)
    evaluator = Evaluator(model_path, data_reader, providers=providers)

    evaluator.predict()
    result = evaluator.get_result()

    annotations = './annotations/instances_val2017.json'
    evaluator.evaluate(result, annotations)


def main():
    args = get_args()
    input_model_path = args.input_model
    output_model_path = args.output_model
    calibration_dataset_path = args.calibrate_dataset
    dr = DataReader(calibration_dataset_path)

	# extract names of concat that does not support quantization
    model = onnx.load(input_model_path)
    exclude_nodes = []
    #for node in model.graph.node:
        #if 'concat' in node.name.lower():
        #    exclude_nodes.append(node.name)
        #if 'mul' in node.name.lower():
        #    exclude_nodes.append(node.name)
        #elif 'add' in node.name.lower():
        #    exclude_nodes.append(node.name)
    add64_idx = [342, 461, 580, 635, 637, 687]
    mul64_idx = [341, 460, 579, 675, 634, 636, 686]
    add64_node = ['Add_'+str(n) for n in add64_idx]
    mul64_node = ['Mul_'+str(n) for n in mul64_idx]
    exclude_nodes = exclude_nodes + add64_node
    exclude_nodes = exclude_nodes + mul64_node
    
    # ptq
    quantize_static(input_model_path,
                    output_model_path,
                    dr,
                    quant_format=args.quant_format,
                    per_channel=args.per_channel,
                    activation_type=QuantType.QUInt8,
                    weight_type=QuantType.QUInt8,
		    nodes_to_exclude=exclude_nodes,
                    calibrate_method=CalibrationMethod.MinMax,
                    extra_options={'WeightSymmetric':False,
                                    'CalibMovingAverage':True
                                    })
    print('Calibrated and quantized model saved.')

    # measure speed
    #print('benchmarking fp32 model...')
    #benchmark(input_model_path)

    #print('benchmarking int8 model...')
    #benchmark(output_model_path)
    
    # measure evaluation
    get_prediction_evaluation(output_model_path, args.valid_dataset, ['CPUExecutionProvider']) 

    # measure evaluation
    #get_prediction_evaluation(output_model_path, args.valid_dataset, ['CPUExecutionProvider']) 

if __name__ == '__main__':
    main()
