import * as ort from 'onnxruntime-web';

// Number of codes in DAC codebook.
const NUM_CODES = 12;

ort.env.wasm.numThreads = 0;
ort.env.wasm.simd = true;

let session: ort.InferenceSession | null = null;

self.onmessage = async (e: MessageEvent) => {
  const { type, payload } = e.data;

  if (type === 'INIT') {
    try {
      session = await ort.InferenceSession.create(`/${payload}`, { 
        executionProviders: ['wasm'],
        enableCpuMemArena: true, 
        graphOptimizationLevel: 'all', 
      });
      self.postMessage({ type: 'READY' });
    } catch (err) {
      console.error("Failed to load ONNX model:", err);
    }
    return;
  }

  if (type === 'ENCODE' && session) {
    try {
      const audioArray: Float32Array = payload;
      const inputTensor = new ort.Tensor('float32', audioArray, [1, 1, audioArray.length]);
      const results = await session.run({ audio: inputTensor });
      
      // Original output shape: [1, 12, frames]
      const tokens = results.codes.data as BigInt64Array;
      
      // 1. Create a detached array (solves the Wasm memory crash)
      // 2. Transpose the data so Time (frames) is the contiguous dimension
      const timeContiguousTokens = new BigInt64Array(tokens.length);
      const frames = tokens.length / NUM_CODES;

      for (let c = 0; c < NUM_CODES; c++) {
        for (let f = 0; f < frames; f++) {
          // Read from [c, f] in the original tensor
          // Write to [f, c] in the new tensor
          timeContiguousTokens[f * NUM_CODES + c] = tokens[c * frames + f];
        }
      }
    
      // Maintain Transferable optimization
      self.postMessage({ 
        type: 'ENCODED_TOKENS', 
        payload: timeContiguousTokens,
      });
    } catch (err) {
      console.error("Encoding error:", err);
    }
  }
};
