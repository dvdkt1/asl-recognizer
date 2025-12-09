// src/components/CameraView.tsx
"use client";

import React, { useRef, useEffect, useState } from "react";
import Webcam from "react-webcam";
import type { Results } from "@mediapipe/hands";
import * as tf from "@tensorflow/tfjs";
import "@tensorflow/tfjs-backend-webgl";

const FINGER_JOINTS = [
  // Thumb
  [0, 1], [1, 2], [2, 3], [3, 4], 

  // Index Finger
  [0, 5], [5, 6], [6, 7], [7, 8],

  // Middle Finger
  [9, 10], [10, 11], [11, 12], 

  // Ring Finger
  [13, 14], [14, 15], [15, 16], 

  // Pinky Finger
  [0, 17], [17, 18], [18, 19], [19, 20],

  // Across the Palm
  [5, 9], [9, 13], [13, 17]
];

export default function CameraView() {
  const webcamRef = useRef<Webcam>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [cameraReady, setCameraReady] = useState(false);

  // State for UI
  const [recordingLabel, setRecordingLabel] = useState<string>(""); 
  const [isRecording, setIsRecording] = useState(false);
  const [sampleCount, setSampleCount] = useState(0); // Triggers re-render for UI counter

  // for camera loop
  // We mirror state into refs so the MediaPipe loop (which is created once) can see updates
  const isRecordingRef = useRef(false);
  const recordingLabelRef = useRef("");
  const collectedData = useRef<any[]>([]);

  const lastFrameTime = useRef<number>(0);
  const latencies = useRef<number[]>([]);

  // Sync State to Refs
  useEffect(() => {
    isRecordingRef.current = isRecording;
    recordingLabelRef.current = recordingLabel;
  }, [isRecording, recordingLabel]);

  const downloadData = () => {
    if (collectedData.current.length === 0) {
      alert("No data collected yet!");
      return;
    }
    const dataStr = "data:text/json;charset=utf-8," + encodeURIComponent(JSON.stringify(collectedData.current));
    const downloadAnchorNode = document.createElement('a');
    downloadAnchorNode.setAttribute("href", dataStr);
    downloadAnchorNode.setAttribute("download", "asl_dataset.json");
    document.body.appendChild(downloadAnchorNode);
    downloadAnchorNode.click();
    downloadAnchorNode.remove();
  };

  const predictHandState = (landmarks: any[]) => {
    // Logic: If Index Finger Tip (8) is higher (smaller y) than Index PIP (6), it is OPEN.
    const indexTip = landmarks[8].y;
    const indexPip = landmarks[6].y;
    return indexTip < indexPip ? "OPEN" : "CLOSED";
  };

  const onResults = (results: Results) => {
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

    // mirror the canvas to match video
    canvasCtx.scale(-1, 1);
    canvasCtx.translate(-videoWidth, 0);

    if (results.multiHandLandmarks && results.multiHandLandmarks.length > 0) {
      for (const landmarks of results.multiHandLandmarks) {
        
        // data collection (logic)
        if (isRecordingRef.current && recordingLabelRef.current) {
            // Flatten the array: [x1, y1, z1, x2, y2, z2, ...]
            const flatLandmarks = landmarks.flatMap((p: any) => [p.x, p.y, p.z]); 
            collectedData.current.push({ label: recordingLabelRef.current, features: flatLandmarks });
            
            // Only update UI every 10 frames to avoid lag
            if (collectedData.current.length % 10 === 0) {
                setSampleCount(collectedData.current.length);
            }
        }

        // draw skeleton (green lines)
        canvasCtx.strokeStyle = "#00FF00";
        canvasCtx.lineWidth = 2;

        for (let i = 0; i < FINGER_JOINTS.length; i++) {
            const startIdx = FINGER_JOINTS[i][0];
            const endIdx = FINGER_JOINTS[i][1];

            const startPoint = landmarks[startIdx];
            const endPoint = landmarks[endIdx];

            const x1 = startPoint.x * videoWidth;
            const y1 = startPoint.y * videoHeight;
            const x2 = endPoint.x * videoWidth;
            const y2 = endPoint.y * videoHeight;

            canvasCtx.beginPath();
            canvasCtx.moveTo(x1, y1);
            canvasCtx.lineTo(x2, y2);
            canvasCtx.stroke();
        }

        // draw joints (red dots)
        canvasCtx.fillStyle = "#FF0000";
        for (const point of landmarks) {
          const x = point.x * videoWidth;
          const y = point.y * videoHeight;
          canvasCtx.beginPath();
          canvasCtx.arc(x, y, 4, 0, 2 * Math.PI); 
          canvasCtx.fill();
        }

        // prediction text
        const prediction = predictHandState(landmarks);
        canvasCtx.save();
        canvasCtx.scale(-1, 1); 
        canvasCtx.fillStyle = "white"; 
        canvasCtx.font = "bold 40px Arial";
        canvasCtx.fillText(prediction, -200, 50);

        // Visual Feedback for Recording
        if (isRecordingRef.current) {
            canvasCtx.fillStyle = "red";
            canvasCtx.font = "bold 20px Arial";
            canvasCtx.fillText(`REC: ${recordingLabelRef.current}`, -200, 80);
        }

        canvasCtx.restore();
      }
    }
    canvasCtx.restore();

    const endTime = performance.now();
    latencies.current.push(endTime - startTime);
    if (latencies.current.length % 100 === 0) {
        const avg = latencies.current.reduce((a,b)=>a+b,0)/latencies.current.length;
        console.log(`Average Latency (p50): ${avg.toFixed(2)} ms`);
    }
  };

  useEffect(() => {
    let hands: any;
    let camera: any;

    const loadModels = async () => {
      
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

    return () => {
        if (camera) camera.stop();
        if (hands) hands.close();
    };
  }, []);

  return (
    <div className="flex flex-col items-center gap-6 p-4">
        
        {/* camera container */}
        <div className="relative w-[640px] h-[480px] border-2 border-gray-800 rounded-lg overflow-hidden bg-black shadow-lg">
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

        {/* controls container */}
        <div className="w-[640px] p-6 border border-gray-700 rounded-lg bg-gray-900 text-white shadow-lg">
            <h3 className="text-xl font-bold mb-4 text-center text-blue-400">Data Collection Console</h3>
            
            <div className="flex flex-row gap-4 items-center justify-center">
                <input 
                    type="text" 
                    placeholder="Label (e.g. A)" 
                    className="text-black text-center text-lg font-bold px-4 py-2 rounded w-32 focus:outline-none focus:ring-2 focus:ring-blue-500"
                    value={recordingLabel}
                    onChange={(e) => setRecordingLabel(e.target.value.toUpperCase())}
                    maxLength={1}
                />

                <button
                    onClick={() => {
                        if (!recordingLabel) {
                            alert("Please enter a label first!");
                            return;
                        }
                        setIsRecording(!isRecording);
                    }}
                    className={`px-6 py-2 rounded font-bold text-lg transition-colors duration-200 ${
                        isRecording 
                        ? 'bg-red-600 hover:bg-red-700 animate-pulse' 
                        : 'bg-green-600 hover:bg-green-700'
                    }`}
                >
                    {isRecording ? "STOP" : "START"}
                </button>

                <div className="flex-grow"></div>

                <button 
                    onClick={downloadData}
                    className="px-6 py-2 bg-blue-600 hover:bg-blue-700 rounded font-bold text-lg transition-colors duration-200"
                >
                    Download JSON
                </button>
            </div>

            <div className="mt-4 text-center text-sm text-gray-400 font-mono">
                Samples collected: <span className="text-white font-bold">{sampleCount}</span>
            </div>
            
            <div className="mt-2 text-xs text-gray-500 text-center">
                1. Enter letter. 2. Press START. 3. Move hand (angles/distance). 4. Press STOP. 5. Download when done.
            </div>
        </div>

    </div>
  );
}