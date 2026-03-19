// src/App.tsx
import { useEffect, useRef, useState } from 'react';

type SocketStatus = 'CONNECTING' | 'CONNECTED' | 'DISCONNECTED' | 'ERROR'
const SOCKET_URL = 'ws://localhost:8080';

export default function App() {
  const [isReady, setIsReady] = useState(false);
  const [isStreaming, setIsStreaming] = useState(false);
  const [socketStatus, setSocketStatus] = useState<SocketStatus>('CONNECTING');
  const workerRef = useRef<Worker | null>(null);
  const audioContextRef = useRef<AudioContext | null>(null);
  const socketRef = useRef<WebSocket | null>(null);

  // Setup encoder worker and load onnx dac model.
  useEffect(() => {
    workerRef.current = new Worker(new URL('./encoder.worker.ts', import.meta.url), { type: 'module' });
    
    workerRef.current.onmessage = (e) => {
      if (e.data.type === 'READY') {
        setIsReady(true);
      }
      
      if (e.data.type === 'ENCODED_TOKENS') {
        const tokens = e.data.payload;
        // Because socketRef is mutable, the worker will always see the currently active socket
        if (socketRef.current?.readyState === WebSocket.OPEN) {
          socketRef.current.send(tokens.buffer);
        }
      }
    };

    workerRef.current.postMessage({ type: 'INIT', payload: 'dac_encoder_16k_int8.onnx' });

    // We no longer brutally terminate the worker on unmount during dev
    // to prevent Vite from choking on severed Wasm streams.
    return () => {
      if (socketRef.current) socketRef.current.close();
    };
  }, []);

  // Streaming (Socket + Audio) spun up fresh on demand
  const startStreaming = async () => {
    try {
      // 1. Spin up a fresh WebSocket connection
      const socket = new WebSocket(SOCKET_URL);
      socketRef.current = socket;
      setSocketStatus('CONNECTING');
      
      socket.onopen = () => setSocketStatus('CONNECTED');
      socket.onerror = () => setSocketStatus('ERROR');
      socket.onclose = () => setSocketStatus('DISCONNECTED');

      // 2. Spin up the Audio Pipeline
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      audioContextRef.current = new AudioContext({ sampleRate: 16000 });
      const source = audioContextRef.current.createMediaStreamSource(stream);
      
      await audioContextRef.current.audioWorklet.addModule('/audio-processor.js');
      const workletNode = new AudioWorkletNode(audioContextRef.current, 'audio-capture-processor');
      
      workletNode.port.onmessage = (e) => {
        workerRef.current?.postMessage({ 
          type: 'ENCODE', 
          payload: e.data
        });
      };

      source.connect(workletNode);
      setIsStreaming(true);
      
    } catch (err) {
      console.error("Failed to start audio stream:", err);
      setSocketStatus('ERROR');
    }
  };

  const stopStreaming = () => {
    if (audioContextRef.current) {
      audioContextRef.current.close();
      audioContextRef.current = null;
    }
    
    if (socketRef.current) {
      socketRef.current.close(); 
      socketRef.current = null;
    }
    
    setIsStreaming(false);
  };

  return (
    <div style={{ padding: '2rem', fontFamily: 'monospace' }}>
      <h1>Client Side Neural Compression PoC</h1>
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
            {isStreaming ? 'STREAMING ACTIVE' : 'START STREAM'}
          </button>

          <button 
            onClick={stopStreaming} 
            disabled={!isStreaming}
            style={{ 
              padding: '1rem', 
              marginLeft: '1rem',
              fontSize: '1.2rem', 
              cursor: !isStreaming ? 'not-allowed' : 'pointer',
              background: !isStreaming ? '#ccc' : '#ff4444',
              color: 'white',
              border: 'none',
              borderRadius: '4px'
            }}
          >
            STOP STREAM
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
