import torch
import dac
from onnxruntime.quantization import quantize_dynamic, QuantType

# Strip dynamic weight math into static constants
def remove_weight_norm_recursively(model):
    """Recursively removes weight normalization from all sub-modules."""
    for module in model.modules():
        try:
            torch.nn.utils.remove_weight_norm(module)
        except ValueError:
            # This specific module didn't have weight_norm attached, which is fine
            pass

def export_models():
    print("1. Downloading and loading pre-trained 16kHz DAC model...")
    # We strictly target 16kHz to keep the model lightweight for edge compute
    model_path = dac.utils.download(model_type="16khz")
    model = dac.DAC.load(model_path)
    model.eval() # Crucial: disable dropout/batchnorm for static export

    print("1b. Baking Weight Normalization into static weights...")
    remove_weight_norm_recursively(model)

    # ---------------------------------------------------------
    # WRAPPER 1: The Client-Side Encoder
    # ---------------------------------------------------------
    class DACEncoderWrapper(torch.nn.Module):
        def __init__(self, dac_model):
            super().__init__()
            self.encoder = dac_model.encoder
            self.quantizer = dac_model.quantizer

        def forward(self, audio_data):
            # Encode continuous audio into latent space
            z = self.encoder(audio_data)
            # Pass through the vector quantizer
            # The quantizer returns multiple values, we only want the discrete codes
            _, codes, _, _, _ = self.quantizer(z)
            return codes

    # ---------------------------------------------------------
    # WRAPPER 2: The Server-Side Decoder
    # ---------------------------------------------------------
    class DACDecoderWrapper(torch.nn.Module):
        def __init__(self, dac_model):
            super().__init__()
            self.quantizer = dac_model.quantizer
            self.decoder = dac_model.decoder

        def forward(self, codes):
            # Reconstruct the continuous latent space from the discrete integers
            z_q, _, _ = self.quantizer.from_codes(codes)
            # Decode back into audio waveforms
            audio_out = self.decoder(z_q)
            return audio_out

    # Instantiate the wrappers
    encoder_wrapper = DACEncoderWrapper(model)
    decoder_wrapper = DACDecoderWrapper(model)

    # Create dummy data so PyTorch can trace the operations
    # Shape: [batch_size, channels, sequence_length]
    print("2. Generating dummy tensors for tracing...")
    dummy_audio = torch.randn(1, 1, 16000) # 1 second of 16kHz audio
    dummy_codes = encoder_wrapper(dummy_audio) # Automatically derives the correct code shape

    # ---------------------------------------------------------
    # EXPORT 1: The Encoder
    # ---------------------------------------------------------
    print("3. Exporting Encoder to ONNX...")
    torch.onnx.export(
        encoder_wrapper,
        dummy_audio,
        "dac_encoder_16k.onnx",
        export_params=True,
        opset_version=18,
        external_data=False, # combines weights (.onnx.data) and ops (.onnx) into a single file.
        input_names=['audio'],
        output_names=['codes'],
        # Dynamic axes are vital so the SDK isn't locked to 1-second chunks
        dynamic_axes={'audio': {2: 'seq_len'}, 'codes': {2: 'code_len'}} 
    )

    # ---------------------------------------------------------
    # EXPORT 2: The Decoder
    # ---------------------------------------------------------
    print("4. Exporting Decoder to ONNX...")
    torch.onnx.export(
        decoder_wrapper,
        dummy_codes,
        "dac_decoder_16k.onnx",
        export_params=True,
        opset_version=18,
        external_data=False, # combines weights (.onnx.data) and ops (.onnx) into a single file.
        input_names=['codes'],
        output_names=['audio'],
        dynamic_axes={'codes': {2: 'code_len'}, 'audio': {2: 'seq_len'}}
    )

    # ---------------------------------------------------------
    # QUANTIZATION: Crushing the Encoder for the Browser
    # ---------------------------------------------------------
    print("5. Quantizing the Encoder for WebAssembly (INT8)...")

    # We only quantize the encoder because it has to run on a weak browser CPU.
    # The decoder will run on a server, so we leave it as high-fidelity Float32.
    quantize_dynamic(
        model_input="dac_encoder_16k.onnx",
        model_output="dac_encoder_16k_int8.onnx",
        weight_type=QuantType.QUInt8,
        use_external_data_format=False # combines weights (.onnx.data) and ops (.onnx) into a single file.
    )
    
    print("Export Complete! You now have your portable Wasm and CLI assets.")

if __name__ == "__main__":
    export_models()
