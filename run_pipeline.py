from espnet2.bin.gan_codec_inference import AudioCoding
from torchaudio import load

from src.metrics import get_model_size, get_bitrate_kbps, get_codebook_size, get_FLOPs, get_n_operations
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
    codec.model.eval()
    no_quant = AudioCoding.from_pretrained(model_tag=config["model_tag"])

    n_codebooks = config["n_codebooks"] if config["n_codebooks"] else codec.model.codec.generator.quantizer.n_q

    sample, fs = load("audio/s.wav")

    # Get general model info (name, bitrate, etc.)
    # ...
    out["model_tag"] = config["model_tag"]
    out["n_codebooks"] = n_codebooks
    out["codebook_size"] = get_codebook_size(codec)
    out["kbps"] = get_bitrate_kbps(codec, inp=sample, fs=fs, n_q=n_codebooks)
    # Update function later to include all modules
    out["Total_FLOPs"] = {}
    out["Total_FLOPs"]["Encoder"] = get_FLOPs(no_quant.model.codec.generator.encoder, inp=sample)

    # Get model size
    out["model_size"] = get_model_size(codec)

    # Run profiler (latency, etc.)
    # ...

    # Get operations per second
    out["n_operations"] = get_n_operations(codec, inp=sample)
    # out["FLOPS"] = out["FLOPs"] / latency

    # Run VERSA
    # ...

    # Save results
    save_to_json(out, config["output_dir"])

if __name__=="__main__":
    main()