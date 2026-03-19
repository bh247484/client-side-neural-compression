# Neural Audio Compression Pipeline (DAC + ONNX)

This repo is a quick POC for client side neural audio compression. It splits the **Descript Audio Codec (DAC)** pytorch model into a quantized onnx encoder and an unquantized onnx decoder. The onnx models are used to compress 16kHz mono audio into discrete tokens on the client side (browser) and reconstruct it on the server side (Rust CLI).

The rest of this documentation is gemini-cli generated with some human lights edits by me. Most of the pipeline is fairly straightforward and conventional. There's some interesting bits on tensor swizzling and time/code major encoded vector shapes necessitated by client side stream chunking. Read on if you're interested.

## Project Architecture & Data Flow

The pipeline is split into four main stages, moving from live audio capture to reconstructed WAV files:

1.  **Frontend & Edge Encoding (`@vite-poc-mki`)**
    *   **Capture**: Live 16kHz mono audio is captured via `AudioWorklet` (`audio-processor.js`).
    *   **Inference**: Raw PCM data is sent to a Web Worker (`encoder.worker.ts`) which runs a **quantized INT8 ONNX version of the DAC Encoder** using WebAssembly (WASM) with SIMD enabled.
    *   **Swizzling**: The encoder's output (Code-Major) is swizzled into **Time-Major** format (see "The Swizzle" below) for efficient streaming.
    *   **Streaming**: Tokens are streamed as raw `BigInt64Array` buffers over a WebSocket.

2.  **Ingestion Server (`@node-server`)**
    *   A TypeScript Node.js server acts as a high-throughput sink.
    *   It receives binary message chunks from the WebSocket and pipes them **directly to disk** as a `.dac` binary file, ensuring minimal memory overhead.

3.  **Rust Decoder CLI (`@cli-decoder`)**
    *   Rust cli script that reads the `.dac` token files.
    *   **Reverse Swizzling**: It re-arranges the Time-Major tokens back into the Code-Major shape expected by the ONNX model.
    *   **Reconstruction**: It runs the **Float32 ONNX DAC Decoder** to transform tokens back into high-fidelity 16kHz audio, saved as a `.wav` file.

4.  **Model Workshop (`@ml-model-workshop`)**
    *   Contains Python scripts to download the pre-trained DAC model, split it into two halves, bake in weight normalization, and export/quantize them for ONNX.

---

## Technical Deep Dives

### 1. DAC Model Splitting
Standard DAC models are often used as monolithic autoencoders. To enable edge-to-cloud streaming, we split the model into two distinct ONNX graphs:
*   **Encoder Wrapper**: Extracts the `encoder` and the `quantizer.encode` logic. It takes `[1, 1, samples]` and returns `[1, 12, frames]` (the discrete codes).
*   **Decoder Wrapper**: Extracts the `quantizer.from_codes` and `decoder` logic. It takes `[1, 12, frames]` and returns the reconstructed waveform.
*   **Static Weights**: We programmatically remove `weight_norm` from the PyTorch modules before export, baking those values into static weights to ensure compatibility with ONNX's static graph requirements.

### 2. Browser-Side ONNX & WASM Challenges
Running a neural encoder in the browser requires overcoming several hurdles:
*   **Quantization**: The raw Float32 encoder is too heavy for most client CPUs. We apply **INT8 Dynamic Quantization** to the encoder, significantly reducing its footprint and increasing inference speed in WASM.
*   **Memory Management**: When passing large `BigInt64Array` tokens out of the WASM heap, we use **detached array buffers** to prevent memory leaks and "heap vanished" errors during long streaming sessions.
*   **SIMD & Threads**: We explicitly enable `ort.env.wasm.simd` and tune thread counts to ensure real-time performance without UI blocking.

### 3. The "Swizzle": Vector Shape Manipulation
One of the most critical novelties in this pipeline is how we handle the layout of the codebook tokens.

**The Problem:**
By default, DAC outputs tokens in **Code-Major** format: `[codebook_1_all_time, codebook_2_all_time, ...]`. If a stream is interrupted, you lose the "tail" of every codebook simultaneously, making partial recovery difficult.

**The Solution (Time-Major Swizzle):**
In `encoder.worker.ts`, we swizzle the tensor so that all codes for `Time T` are adjacent in memory:
`[time_0_codes_1-12, time_1_codes_1-12, ...]`.
*   **Benefit**: This allows the Node server to treat the stream as a sequential series of temporal "frames." Even a partial file is theoretically decodable up to the last byte received.

**The Reverse Swizzle (Rust):**
Since the ONNX Decoder expects the original Code-Major format, the Rust CLI performs an efficient `ndarray` permutation:
```rust
// [batch, time, codes] -> [batch, codes, time]
let tokens_array = time_major_array.permuted_axes([0, 2, 1]).to_owned();
```
This ensures that we get the streaming benefits of Time-Major data while maintaining compatibility with the underlying neural architecture.

---

## Development & Usage

### Exporting Models
See `./ml-model-workshop/dev-notes.md` for info on dependency problems. Mileage on your machine may vary but I had to jump through some narrow and fickle hoops to get things to play nicely. Once you get dependencies installed successfully though just run `python3 export_models.py` and you should get the following .onnx files...

- Unquantized DAC decoder: `dac_decoder_16k.onnx`
- 8 Bit Quantized DAC encoder: `dac_encoder_16k_int8.onnx`
- Unquantized DAC encoder: `dac_encoder_16k.onnx`

```bash
python3 export_models.py
```

### Running the Pipeline
1.  **Start the Node Server**: `cd node-server && npm run dev`
2.  **Start the Frontend**: `cd vite-poc-mki && npm run dev`
3.  **Decode a Session**:
    ```bash
    cd cli-decoder
    cargo build --release
    ```
    - See Rust script source code for example cli commands after building the binary executable.
