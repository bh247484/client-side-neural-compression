import express from 'express';
import { WebSocketServer, WebSocket } from 'ws';
import http from 'http';
import fs from 'fs';
import path from 'path';

const app = express();
const server = http.createServer(app);
const wss = new WebSocketServer({ server });

const PORT = 8080;

wss.on('connection', (ws: WebSocket) => {
  console.log('Client connected');
  
  // Create a buffer array to store incoming token chunks
  const chunks: Buffer[] = [];
  const sessionId = Date.now();
  const outputFilePath = path.join(__dirname, '..', `session-${sessionId}.dac`);

  ws.on('message', (data: Buffer, isBinary: boolean) => {
    if (isBinary) {
      chunks.push(data);
      console.log(`Received binary chunk of size: ${data.length} bytes`);
    } else {
      console.log('Received text message:', data.toString());
    }
  });

  ws.on('close', () => {
    console.log('Client disconnected');
    
    if (chunks.length > 0) {
      // Concatenate all chunks and write to disk
      const fullBuffer = Buffer.concat(chunks);
      fs.writeFile(outputFilePath, fullBuffer, (err) => {
        if (err) {
          console.error('Failed to save .dac file:', err);
        } else {
          console.log(`Successfully saved tokens to: ${outputFilePath}`);
          console.log(`Total file size: ${fullBuffer.length} bytes`);
        }
      });
    } else {
      console.log('No tokens received, skipping file save.');
    }
  });

  ws.on('error', (err) => {
    console.error('WebSocket Error:', err);
  });
});

server.listen(PORT, () => {
  console.log(`Node (TypeScript) server listening on http://localhost:${PORT}`);
  console.log(`WebSocket server is active on ws://localhost:${PORT}`);
});
