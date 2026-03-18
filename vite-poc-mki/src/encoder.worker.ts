// src/encoder.worker.ts
import * as ort from 'onnxruntime-web';

// Force WebAssembly backend and enable SIMD/threading for max performance
ort.env.wasm.numThreads = 0;
ort.env.wasm.simd = true;
// ort.env.wasm.proxy = true;

let session: ort.InferenceSession | null = null;

self.onmessage = async (e: MessageEvent) => {
  const { type, payload } = e.data;

  if (type === 'INIT') {
    try {
      // Load the model from the public folder
      session = await ort.InferenceSession.create(`/${payload}`, { 
        executionProviders: ['wasm'],
        // executionProviders: ['webgpu'],
        enableCpuMemArena: true, 
        graphOptimizationLevel: 'all', 
        executionMode: 'sequential',
      });
      self.postMessage({ type: 'READY' });
    } catch (err) {
      console.error("Failed to load ONNX model:", err);
    }
  }

  if (type === 'ENCODE' && session) {
    const audioArray: Float32Array = payload;
    
    // Create the tensor: [batch_size=1, channels=1, sequence_length=length]
    const inputTensor = new ort.Tensor('float32', audioArray, [1, 1, audioArray.length]);
    
    // Start the stopwatch
    const startTime = performance.now();
    
    // Run the model
    const results = await session.run({ audio: inputTensor });
    
    // Stop the stopwatch
    const endTime = performance.now();
    const inferenceTimeMs = endTime - startTime;
    
    // Calculate RTF (assuming input is 16000 samples = 1000ms)
    const audioDurationMs = (audioArray.length / 16000) * 1000;
    const rtf = inferenceTimeMs / audioDurationMs;

    // Extract the tokens
    const tokens = results.codes.data;

    self.postMessage({ 
      type: 'METRICS', 
      inferenceTime: inferenceTimeMs.toFixed(2), 
      rtf: rtf.toFixed(3),
      tokensCount: tokens.length
    });
  }
};
