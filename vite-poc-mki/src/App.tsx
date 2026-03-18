// src/App.tsx
import { useEffect, useRef, useState } from 'react';

type SocketStatus = 'CONNECTING' | 'CONNECTED' | 'DISCONNECTED' | 'ERROR'
const SOCKET_PORT = 'ws://localhost:8080';

export default function App() {
  const [isReady, setIsReady] = useState(false);
  const [isStreaming, setIsStreaming] = useState(false);
  const [socketStatus, setSocketStatus] = useState<SocketStatus>('CONNECTING');
  const workerRef = useRef<Worker | null>(null);
  const audioContextRef = useRef<AudioContext | null>(null);
  const socketRef = useRef<WebSocket | null>(null);

  useEffect(() => {
    // 1. Initialize the WebSocket connection (placeholder port 8080)
    const socket = new WebSocket(SOCKET_PORT);
    socketRef.current = socket;
    
    socket.onopen = () => {
      console.log("WebSocket: Connected to backend");
      setSocketStatus('CONNECTED');
    };
    socket.onerror = (err) => {
      console.error("WebSocket: Error", err);
      setSocketStatus('ERROR');
    };
    socket.onclose = () => {
      console.log("WebSocket: Disconnected");
      setSocketStatus('DISCONNECTED');
    };

    // 2. Spin up the encoder worker
    workerRef.current = new Worker(new URL('./encoder.worker.ts', import.meta.url), { type: 'module' });
    
    workerRef.current.onmessage = (e) => {
      if (e.data.type === 'READY') {
        setIsReady(true);
      }
      
      if (e.data.type === 'ENCODED_TOKENS') {
        // Stream the tokens directly to the backend
        if (socketRef.current?.readyState === WebSocket.OPEN) {
          socketRef.current.send(e.data.payload.buffer); // Send the raw buffer for efficiency
        }
      }
    };

    // Initialize with the quantized model
    workerRef.current.postMessage({ type: 'INIT', payload: 'dac_encoder_16k_int8.onnx' });

    return () => {
      workerRef.current?.terminate();
      socket.close();
    };
  }, []);

  const startStreaming = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      
      // Force hardware to 16kHz
      audioContextRef.current = new AudioContext({ sampleRate: 16000 });
      const source = audioContextRef.current.createMediaStreamSource(stream);
      
      // Load Worklet
      await audioContextRef.current.audioWorklet.addModule('/audio-processor.js');
      
      const workletNode = new AudioWorkletNode(
        audioContextRef.current, 
        'audio-capture-processor'
      );
      
      // Pipe raw audio chunks to the encoder worker
      workletNode.port.onmessage = (e) => {
        workerRef.current?.postMessage({ type: 'ENCODE', payload: e.data });
      };

      source.connect(workletNode);
      setIsStreaming(true);
      
    } catch (err) {
      console.error("Failed to start audio stream:", err);
    }
  };

  return (
    <div style={{ padding: '2rem', fontFamily: 'monospace' }}>
      <h1>B2B Audio Edge Pipeline PoC</h1>
      {!isReady ? (
        <p>Initializing Encoder & Socket...</p>
      ) : (
        <div>
          <button 
            onClick={startStreaming} 
            disabled={isStreaming}
            style={{ 
              padding: '1rem', 
              fontSize: '1.2rem', 
              cursor: isStreaming ? 'not-allowed' : 'pointer',
              background: isStreaming ? '#ccc' : '#0070f3',
              color: 'white',
              border: 'none',
              borderRadius: '4px'
            }}
          >
            {isStreaming ? 'STREAMING ACTIVE' : 'START EDGE STREAM'}
          </button>
          
          <div style={{ marginTop: '2rem', padding: '1rem', border: '1px solid #ccc' }}>
            <h2>Stream Status</h2>
            <p><strong>Encoder:</strong> {isReady ? 'READY' : 'LOADING'}</p>
            <p><strong>Socket:</strong> {socketStatus}</p>
            <p><strong>Active:</strong> {isStreaming ? 'YES' : 'NO'}</p>
          </div>
        </div>
      )}
    </div>
  );
}
