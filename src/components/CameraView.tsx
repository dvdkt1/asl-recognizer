// src/components/CameraView.tsx
"use client";

import React, { useRef, useEffect, useState } from "react";
import Webcam from "react-webcam";
import type { Results } from "@mediapipe/hands";
import * as tf from "@tensorflow/tfjs";
import "@tensorflow/tfjs-backend-webgl";

export default function CameraView() {
  const webcamRef = useRef<Webcam>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [cameraReady, setCameraReady] = useState(false);
  const lastFrameTime = useRef<number>(0);
  const latencies = useRef<number[]>([]);

const predictHandState = (landmarks: any[]) => {
    // Logic: If Index Finger Tip (8) is higher (smaller y) than Index PIP (6), it is OPEN.
    const indexTip = landmarks[8].y;
    const indexPip = landmarks[6].y;
    return indexTip < indexPip ? "OPEN" : "CLOSED";
  };

  useEffect(() => {
    let hands: any;
    let camera: any;

    const loadModels = async () => {
      
      // imports libraries
      const { Hands } = await import("@mediapipe/hands");
      const { Camera } = await import("@mediapipe/camera_utils");

      hands = new Hands({
        locateFile: (file) => {
          return `https://cdn.jsdelivr.net/npm/@mediapipe/hands/${file}`;
        },
      });

      hands.setOptions({
        maxNumHands: 1,
        modelComplexity: 1,
        minDetectionConfidence: 0.5,
        minTrackingConfidence: 0.5,
      });

      hands.onResults(onResults);

      if (
        typeof window !== "undefined" &&
        webcamRef.current &&
        webcamRef.current.video
      ) {
        camera = new Camera(webcamRef.current.video, {
          onFrame: async () => {
            if (webcamRef.current?.video) {
              await hands.send({ image: webcamRef.current.video });
            }
          },
          width: 640,
          height: 480,
        });
        camera.start();
        setCameraReady(true);
      }
    };

    loadModels();

    // Cleanup function to stop camera when you leave the page
    return () => {
        if (camera) camera.stop();
        if (hands) hands.close();
    };
  }, []);

  const onResults = (results: Results) => {
    // Starts Timer
    const startTime = performance.now();

    if (!canvasRef.current || !webcamRef.current?.video) return;

    const videoWidth = webcamRef.current.video.videoWidth;
    const videoHeight = webcamRef.current.video.videoHeight;

    canvasRef.current.width = videoWidth;
    canvasRef.current.height = videoHeight;

    const canvasCtx = canvasRef.current.getContext("2d");
    if (!canvasCtx) return;

    canvasCtx.save();
    canvasCtx.clearRect(0, 0, videoWidth, videoHeight);

    // Mirror the canvas to match the video
    canvasCtx.scale(-1, 1);
    canvasCtx.translate(-videoWidth, 0);

    if (results.multiHandLandmarks && results.multiHandLandmarks.length > 0) {
      for (const landmarks of results.multiHandLandmarks) {
        // draws green dots
        for (const point of landmarks) {
          const x = point.x * videoWidth;
          const y = point.y * videoHeight;
          canvasCtx.beginPath();
          canvasCtx.arc(x, y, 5, 0, 2 * Math.PI);
          canvasCtx.fillStyle = "#00FF00";
          canvasCtx.fill();
        }

        // baseline prediction
        const prediction = predictHandState(landmarks);

        // draw the prediction text and make sure it isn't backwards
        canvasCtx.save();
        canvasCtx.scale(-1, 1); 
        canvasCtx.fillStyle = "red";
        canvasCtx.font = "48px Arial";
        canvasCtx.fillText(prediction, -200, 50); // Draw near top right
        canvasCtx.restore();
      }
    }
    canvasCtx.restore();

    // End timer and logs metrics
    const endTime = performance.now();
    const latency = endTime - startTime;
    latencies.current.push(latency);

    // Log average every 100 frames
    if (latencies.current.length % 100 === 0) {
        const avgLatency = latencies.current.reduce((a, b) => a + b, 0) / latencies.current.length;
        console.log(`Average Latency (p50): ${avgLatency.toFixed(2)} ms`);
    }
  };

  return (
    <div className="relative w-[640px] h-[480px] mx-auto border-2 border-gray-800 rounded-lg overflow-hidden bg-black">
      <Webcam
        ref={webcamRef}
        className="absolute top-0 left-0 w-full h-full object-cover"
        mirrored={true}
      />
      <canvas
        ref={canvasRef}
        className="absolute top-0 left-0 w-full h-full object-cover z-10"
      />
      {!cameraReady && (
        <div className="absolute top-0 left-0 w-full h-full flex items-center justify-center z-20 bg-black/80 text-white font-mono">
          Initializing Computer Vision...
        </div>
      )}
    </div>
  );
}