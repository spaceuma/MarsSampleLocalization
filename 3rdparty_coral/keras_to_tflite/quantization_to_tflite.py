import tensorflow as tf
import numpy as np
import sys

def representative_dataset_gen():
    for _ in range(250):
        yield [np.random.uniform(0.0, 1.0, size=(1, 416, 416, 3)).astype(np.float32)]

model_fn = sys.argv[1]
out_fn = sys.argv[2]

# Convert and apply full integer quantization
converter = tf.compat.v1.lite.TFLiteConverter.from_keras_model_file(model_fn)
converter.optimizations = [tf.lite.Optimize.OPTIMIZE_FOR_LATENCY  ]
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8,  tf.lite.OpsSet.SELECT_TF_OPS]

# Set inputs and outputs of network to 8-bit unsigned integer
converter.inference_input_type = tf.uint8
converter.inference_output_type = tf.uint8
converter.representative_dataset = representative_dataset_gen

# https://github.com/google-coral/edgetpu/issues/169 tpu compiler doesn't support mlir
converter.experimental_new_converter = False 
tflite_model = converter.convert()    
open(sys.argv[2], "wb").write(tflite_model)
