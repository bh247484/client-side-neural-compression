package main

import (
	"bytes"
	"fmt"
	"log"
	"net/http"
	"os"
	"path/filepath"
	"time"

	"github.com/gorilla/websocket"
)

var upgrader = websocket.Upgrader{
	CheckOrigin: func(r *http.Request) bool {
		return true // Relaxed for PoC
	},
}

const PORT = 8080

func handleWebSocket(w http.ResponseWriter, r *http.Request) {
	conn, err := upgrader.Upgrade(w, r, nil)
	if err != nil {
		log.Printf("Upgrade error: %v", err)
		return
	}
	defer conn.Close()

	log.Println("Client connected")

	var chunks [][]byte
	sessionId := time.Now().UnixMilli()

	for {
		messageType, data, err := conn.ReadMessage()
		if err != nil {
			if websocket.IsUnexpectedCloseError(err, websocket.CloseGoingAway, websocket.CloseAbnormalClosure) {
				log.Printf("Read error: %v", err)
			}
			break
		}

		if messageType == websocket.BinaryMessage {
			chunks = append(chunks, data)
			log.Printf("Received binary chunk of size: %d bytes", len(data))
		} else {
			log.Printf("Received text message: %s", string(data))
		}
	}

	log.Println("Client disconnected")

	if len(chunks) > 0 {
		fullBuffer := bytes.Join(chunks, nil)
		fileName := fmt.Sprintf("session-%d.dac", sessionId)
		outputFilePath, _ := filepath.Abs(fileName)

		err := os.WriteFile(outputFilePath, fullBuffer, 0644)
		if err != nil {
			log.Printf("Failed to save .dac file: %v", err)
		} else {
			log.Printf("Successfully saved tokens to: %s", outputFilePath)
			log.Printf("Total file size: %d bytes", len(fullBuffer))
		}
	} else {
		log.Println("No tokens received, skipping file save.")
	}
}

func main() {
	http.HandleFunc("/", handleWebSocket)

	log.Printf("Go server listening on http://localhost:%d", PORT)
	log.Printf("WebSocket server is active on ws://localhost:%d", PORT)

	if err := http.ListenAndServe(fmt.Sprintf(":%d", PORT), nil); err != nil {
		log.Fatal("ListenAndServe error:", err)
	}
}
