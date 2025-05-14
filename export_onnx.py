import torch
import argparse
import onnx
from pathlib import Path

from styletts2_inference.models import StyleTTS2


def export(args):
    if args.hf_path:
        model = StyleTTS2(args.hf_path, device='cpu')
    else:
        model = StyleTTS2(config_path=args.config, weights_path=args.weights_path, device='cpu')
    tokens = model.tokenizer.encode(args.text)
    if model.config.model_params.multispeaker:
        style = model.predict_style_multi(args.audio_prompt, tokens)
    else:
        style = model.predict_style_single(tokens)
    
    model_inputs = {"tokens": tokens,
                    "speed": torch.tensor(1.0, dtype=torch.float32),
                    "s_prev": style
                    }
    input_names = ['tokens', 'speed', 's_prev']


    torch.onnx.export(model, kwargs=model_inputs,
                            f='styletts.onnx', dynamo=False, export_params=True, report=False, verify=True,
                            input_names=input_names,
                            output_names=["output_wav"],
                            training=torch.onnx.TrainingMode.EVAL,
                            opset_version=19,
                            dynamic_axes={
                                'tokens':{0: "seq_length"},
                                'output_wav':{0: "seq_length"}
                            }
                    )
    
    onnx_model = onnx.load('styletts.onnx')
    for node in onnx_model.graph.node:
        if node.op_type == "Transpose":
            if node.name == "/text_encoder_1/Transpose_7" or node.name == "/text_encoder_1/Transpose_20" or node.name == "/text_encoder_1/Transpose_14":  
                perm = list(node.attribute[0].ints)  
                perm = [2 if i == -1 else i for i in perm]  
                node.attribute[0].ints[:] = perm  
    onnx.save(onnx_model, "styletts.onnx")
    print('Exported!!!')




if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="export_onnx.py",
        description="Commandline tool for export styletts2 models to onnx"
    )
    parser.add_argument(
        "-c",
        "--config",
        type=Path,
        help="The path to model config file.",
    )

    parser.add_argument(
        "-hf",
        "--hf_path",
        type=str,
        default=None,
        help="The path to model in hugginface.",
    )


    parser.add_argument(
        "-m",
        "--model",
        type=Path,
        help="The model path.",
    )
    
    parser.add_argument(
        "-ap",
        "--audio_prompt",
        type=str,
        default=None,
        help="The path to audio prompt if model is multispeaker",
    )

    parser.add_argument(
        "-t",
        "--text",
        type=str,
        default=None,
        help="The dummy phonemized text to generate",
    )
    args = parser.parse_args()
    export(args)