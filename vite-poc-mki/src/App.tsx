// src/App.tsx
import { useEffect, useRef, useState } from 'react';

export default function App() {
  const [isReady, setIsReady] = useState(false);
  const [metrics, setMetrics] = useState({ time: '0', rtf: '0' });
  const workerRef = useRef<Worker | null>(null);
  const audioContextRef = useRef<AudioContext | null>(null);

  // We will buffer 1 second of audio (16,000 samples) before sending to the worker
  const bufferRef = useRef<Float32Array>(new Float32Array(16000));
  const bufferIndexRef = useRef(0);

  useEffect(() => {
    // Spin up the worker
    workerRef.current = new Worker(new URL('./encoder.worker.ts', import.meta.url), { type: 'module' });
    
    workerRef.current.onmessage = (e) => {
      if (e.data.type === 'READY') setIsReady(true);
      if (e.data.type === 'METRICS') {
        setMetrics({ time: e.data.inferenceTime, rtf: e.data.rtf });
      }
    };

    // Initialize the quantized model by default
    workerRef.current.postMessage({ type: 'INIT', payload: 'dac_encoder_16k_int8.onnx' });

    return () => workerRef.current?.terminate();
  }, []);

const startRecording = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      
      // Force the hardware to downsample to 16kHz
      audioContextRef.current = new AudioContext({ sampleRate: 16000 });
      const source = audioContextRef.current.createMediaStreamSource(stream);
      
      // 1. Fetch and load the Worklet script from the public folder
      await audioContextRef.current.audioWorklet.addModule('/audio-processor.js');
      
      // 2. Instantiate the node using the exact string name we registered
      const workletNode = new AudioWorkletNode(
        audioContextRef.current, 
        'audio-capture-processor'
      );
      
      // 3. Listen for the 16,000-sample arrays coming from the background audio thread
      workletNode.port.onmessage = (e) => {
        const audioChunk = e.data;
        // Pass it straight to the ONNX Web Worker
        workerRef.current?.postMessage({ type: 'ENCODE', payload: audioChunk });
      };

      // 4. Connect the microphone to the Worklet
      source.connect(workletNode);
      
      // CRITICAL: Do NOT connect the workletNode to the audioContext destination,
      // otherwise the user's microphone will echo loudly back through their speakers.
      
    } catch (err) {
      console.error("Failed to start audio capture:", err);
    }
  };

  return (
    <div style={{ padding: '2rem', fontFamily: 'monospace' }}>
      <h1>B2B Audio Edge Pipeline PoC</h1>
      {!isReady ? (
        <p>Loading ONNX WebAssembly Backend...</p>
      ) : (
        <div>
          <button onClick={startRecording} style={{ padding: '1rem', fontSize: '1.2rem' }}>
            Start 16kHz Capture & Encode
          </button>
          
          <div style={{ marginTop: '2rem', padding: '1rem', background: '#f0f0f0' }}>
            <h2>Latest Inference Metrics</h2>
            <p><strong>Processing Time (per 1s chunk):</strong> {metrics.time} ms</p>
            <p>
              <strong>Real-Time Factor (RTF):</strong> <span style={{ color: Number(metrics.rtf) > 0.8 ? 'red' : 'green'}}>{metrics.rtf}</span>
            </p>
            <p style={{ fontSize: '0.8rem', color: '#666' }}>
              * If RTF is &gt; 1.0, the encoder is lagging behind real-time.
            </p>
          </div>
        </div>
      )}
    </div>
  );
}
