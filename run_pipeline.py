from espnet2.bin.gan_codec_inference import AudioCoding

from src.metrics import get_model_size
from src.utils import load_config, save_to_json

def load_model(config: dict) -> AudioCoding:
    model = AudioCoding.from_pretrained(
        model_tag=config["model_tag"],
        quantize_model=config["quantize_model"],
        quantize_modules=config["quantize_modules"],
        quantize_dtype=config["quantize_dtype"],
    )
    return model

def main():
    # Parse config
    config = load_config()
    out = {}

    # Load the model
    codec = load_model(config)

    # Get general model info (name, bitrate, etc.)
    # ...

    # Get model size
    out["model_size"] = get_model_size(codec.model)

    # Get FLOPS, MACS
    # ...

    # Run profiler (latency, etc.)
    # ...

    # Run VERSA
    # ...

    # Save results
    save_to_json(out, config["output_dir"])

if __name__=="__main__":
    main()