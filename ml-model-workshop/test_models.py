import onnxruntime as ort
import torchaudio
import torch
import numpy as np
from pathlib import Path

def run_test_pipeline():
    # Configuration
    sample_dir = Path("speech-samples")
    decompressed_base = sample_dir / "decompressed"
    decoder_path = Path("dac_decoder_16k.onnx")
    encoders = [
        ("dac_encoder_16k_int8.onnx", "quantized"),
        ("dac_encoder_16k.onnx", "unquantized")
    ]

    # Find all mp3 files recursively, excluding the decompressed directory
    mp3_files = [
        f for f in sample_dir.rglob("*.mp3") 
        if decompressed_base not in f.parents
    ]
    
    if not mp3_files:
        print(f"No MP3 files found in {sample_dir}")
        return

    if not decoder_path.exists():
        print(f"Error: Decoder model not found at {decoder_path}")
        return

    # Ensure output directory exists
    decompressed_base.mkdir(parents=True, exist_ok=True)

    print(f"Found {len(mp3_files)} MP3 files. Starting processing...")

    # Initialize decoder once to save time
    print(f"Loading decoder: {decoder_path}")
    decoder_session = ort.InferenceSession(str(decoder_path), providers=['CPUExecutionProvider'])

    for mp3_path in mp3_files:
        print(f"\nProcessing: {mp3_path.name}")
        
        try:
            # 1. Loading audio and ensuring 16kHz mono
            waveform, sample_rate = torchaudio.load(str(mp3_path))
            if sample_rate != 16000:
                waveform = torchaudio.functional.resample(waveform, sample_rate, 16000)
            if waveform.shape[0] > 1:
                waveform = torch.mean(waveform, dim=0, keepdim=True)
            
            audio_input = waveform.unsqueeze(0).numpy().astype(np.float32)

            for encoder_path_str, suffix in encoders:
                encoder_path = Path(encoder_path_str)
                if not encoder_path.exists():
                    print(f"  [Skip] {suffix} encoder not found: {encoder_path}")
                    continue

                print(f"  Running {suffix} pipeline...")
                
                # 2. Encoding
                encoder_session = ort.InferenceSession(str(encoder_path), providers=['CPUExecutionProvider'])
                encoder_inputs = {encoder_session.get_inputs()[0].name: audio_input}
                tokens = encoder_session.run(None, encoder_inputs)[0]
                
                # 3. Decoding (from the tokens in memory)
                tokens_int64 = tokens.astype(np.int64)
                decoder_inputs = {decoder_session.get_inputs()[0].name: tokens_int64}
                reconstructed_audio = decoder_session.run(None, decoder_inputs)[0]

                # 4. Saving Decompressed (Audio) to the decompressed folder
                rel_path = mp3_path.relative_to(sample_dir)
                decompressed_path = decompressed_base / rel_path.with_name(f"{mp3_path.stem}-decompressed-{suffix}.wav")
                
                # Create subdirectories if they exist in the source
                decompressed_path.parent.mkdir(parents=True, exist_ok=True)
                
                out_tensor = torch.from_numpy(reconstructed_audio).squeeze(0)
                torchaudio.save(str(decompressed_path), out_tensor, 16000)
                print(f"    -> Saved decompressed audio: {decompressed_path}")
        
        except Exception as e:
            print(f"  Error processing {mp3_path.name}: {e}")

if __name__ == "__main__":
    run_test_pipeline()
