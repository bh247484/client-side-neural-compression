// public/audio-processor.js

class AudioCaptureProcessor extends AudioWorkletProcessor {
  constructor() {
    super();
    this.CHUNK_SIZE = 16000; 
    this.buffer = new Float32Array(this.CHUNK_SIZE);
    this.index = 0;
  }

  process(inputs, outputs, parameters) {
    const channelData = inputs[0]?.[0];
    if (!channelData) return true;

    for (let i = 0; i < channelData.length; i++) {
      this.buffer[this.index++] = channelData[i];

      if (this.index >= this.CHUNK_SIZE) {
        // Send a copy of the full buffer
        this.port.postMessage(this.buffer.slice());
        // Reset index for the next chunk
        this.index = 0;
      }
    }

    return true; 
  }
}

registerProcessor('audio-capture-processor', AudioCaptureProcessor);
