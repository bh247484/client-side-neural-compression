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
  
  const sessionId = Date.now();
  const outputFilePath = path.join(__dirname, '..', `session-${sessionId}.dac`);
  
  // Maintain Direct-to-Disk Streaming
  const fileStream = fs.createWriteStream(outputFilePath);
  let totalBytesWritten = 0;
  let chunkCount = 0;

  ws.on('message', (data: Buffer, isBinary: boolean) => {
    if (isBinary) {
      // Roll back sequence header extraction, write raw data
      fileStream.write(data);
      totalBytesWritten += data.length;
      chunkCount++;
      
      if (chunkCount % 10 === 0) {
        console.log(`Received chunk ${chunkCount}, total size: ${totalBytesWritten} bytes`);
      }
    } else {
      console.log('Received text message:', data.toString());
    }
  });

  ws.on('close', () => {
    console.log('Client disconnected');
    fileStream.end();
    
    if (totalBytesWritten > 0) {
      console.log(`Successfully saved tokens to: ${outputFilePath}`);
      console.log(`Total file size: ${totalBytesWritten} bytes`);
    } else {
      console.log('No tokens received, removing empty file.');
      fs.unlink(outputFilePath, () => {});
    }
  });

  ws.on('error', (err) => {
    console.error('WebSocket Error:', err);
    fileStream.end();
  });
});

server.listen(PORT, () => {
  console.log(`Node (TypeScript) server listening on http://localhost:${PORT}`);
  console.log(`WebSocket server is active on ws://localhost:${PORT}`);
});
