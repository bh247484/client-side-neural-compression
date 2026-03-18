// public/audio-processor.js

class AudioCaptureProcessor extends AudioWorkletProcessor {
  constructor() {
    super();
    // Hardcode to our exact required chunk size
    this.CHUNK_SIZE = 16000; 
    this.buffer = new Float32Array(this.CHUNK_SIZE);
    this.index = 0;
  }

  process(inputs, outputs, parameters) {
    // inputs[0] is the first audio input (the mic)
    // inputs[0][0] is the first channel (mono)
    const channelData = inputs[0]?.[0];
    
    if (!channelData) return true;

    // Stream the incoming Float32 data into our chunk buffer
    for (let i = 0; i < channelData.length; i++) {
      if (this.index < this.CHUNK_SIZE) {
        this.buffer[this.index++] = channelData[i];
      }
    }

    // Once we hit exactly 16,000 samples (1 second)
    if (this.index >= this.CHUNK_SIZE) {
      // Send a copy of the buffer to the React main thread
      this.port.postMessage(this.buffer.slice());
      // Reset the index to start filling the next chunk
      this.index = 0;
    }

    // Must return true to keep the processor alive in the audio pipeline
    return true; 
  }
}

// Register it so the AudioContext can find it by name
registerProcessor('audio-capture-processor', AudioCaptureProcessor);
