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
  
  // app state
  const [mode, setMode] = useState<'PREDICT' | 'COLLECT'>('PREDICT');
  const [model, setModel] = useState<tf.LayersModel | null>(null);
  const [classes, setClasses] = useState<string[]>([]);
  const [prediction, setPrediction] = useState<string>("Loading...");

  // data collection state
  const [recordingLabel, setRecordingLabel] = useState<string>(""); 
  const [isRecording, setIsRecording] = useState(false);
  const [sampleCount, setSampleCount] = useState(0); 

  // refs
  const isRecordingRef = useRef(false);
  const recordingLabelRef = useRef("");
  const collectedData = useRef<any[]>([]);
  const modeRef = useRef<'PREDICT' | 'COLLECT'>('PREDICT');
  const modelRef = useRef<tf.LayersModel | null>(null);
  const classesRef = useRef<string[]>([]);

  // sync state to refs for the animation loop
  useEffect(() => {
    isRecordingRef.current = isRecording;
    recordingLabelRef.current = recordingLabel;
    modeRef.current = mode;
    modelRef.current = model;
    classesRef.current = classes;
  }, [isRecording, recordingLabel, mode, model, classes]);

  // load model on startup
  useEffect(() => {
    const loadResources = async () => {
        try {
            // loads the classes mapping
            const classesResponse = await fetch("/model_v2/classes.json");
            const classesData = await classesResponse.json();
            setClasses(classesData);
            
            // loads the tensorflow model
            const loadedModel = await tf.loadLayersModel("/model_v2/model.json");
            setModel(loadedModel);
            setPrediction("Ready");
            console.log("Model & Classes Loaded Successfully");
        } catch (error) {
            console.error("Failed to load model:", error);
            setPrediction("Error Loading Model");
        }
    };
    loadResources();
  }, []);

  // preprocessing (must match python exactly)
  const preprocessLandmarks = (landmarks: any[]) => {
    // converts to array of arrays
    const rawPoints = landmarks.map(p => [p.x, p.y, p.z]);

    // center: subtracts wrist from all points
    const wrist = rawPoints[0];
    const centered = rawPoints.map(p => [
        p[0] - wrist[0], 
        p[1] - wrist[1], 
        p[2] - wrist[2]
    ]);

    // scale: divide by max distance (from wrist)
    const norms = centered.map(p => Math.sqrt(p[0]**2 + p[1]**2 + p[2]**2));
    const maxDist = Math.max(...norms);
    
    // avoids division by zero
    const scale = maxDist > 0 ? maxDist : 1; 

    const normalized = centered.map(p => [
        p[0] / scale, 
        p[1] / scale, 
        p[2] / scale
    ]);

    // flatten to 1D array
    return normalized.flat();
  };

  const onResults = (results: Results) => {
    if (!canvasRef.current || !webcamRef.current?.video) return;

    const videoWidth = webcamRef.current.video.videoWidth;
    const videoHeight = webcamRef.current.video.videoHeight;
    canvasRef.current.width = videoWidth;
    canvasRef.current.height = videoHeight;

    const canvasCtx = canvasRef.current.getContext("2d");
    if (!canvasCtx) return;

    canvasCtx.save();
    canvasCtx.clearRect(0, 0, videoWidth, videoHeight);
    canvasCtx.scale(-1, 1);
    canvasCtx.translate(-videoWidth, 0);

    if (results.multiHandLandmarks && results.multiHandLandmarks.length > 0) {
      for (const landmarks of results.multiHandLandmarks) {
        
        // prediction
        if (modeRef.current === 'PREDICT' && modelRef.current && classesRef.current.length > 0) {
            const inputData = preprocessLandmarks(landmarks);
            
            // run inference inside tf.tidy to clean up tensors automatically
            const predLabel = tf.tidy(() => {
                const inputTensor = tf.tensor2d([inputData], [1, 63]); 
                const prediction = modelRef.current!.predict(inputTensor) as tf.Tensor;
                const index = prediction.argMax(1).dataSync()[0];
                return classesRef.current[index];
            });

            // update UI
            setPrediction(predLabel);

            // draws label on canvas 
            canvasCtx.fillStyle = "#00FF00";
            canvasCtx.font = "bold 60px Arial";
            canvasCtx.fillText(predLabel, -150, 80);
        }

        // collection
        if (modeRef.current === 'COLLECT' && isRecordingRef.current && recordingLabelRef.current) {
            const flatLandmarks = landmarks.flatMap((p: any) => [p.x, p.y, p.z]); 
            collectedData.current.push({ label: recordingLabelRef.current, features: flatLandmarks });
            if (collectedData.current.length % 10 === 0) setSampleCount(collectedData.current.length);
        }

        // draws skeleton
        canvasCtx.strokeStyle = modeRef.current === 'PREDICT' ? "#00FF00" : "#0088FF";
        canvasCtx.lineWidth = 2;
        for (let i = 0; i < FINGER_JOINTS.length; i++) {
            const [start, end] = FINGER_JOINTS[i];
            const p1 = landmarks[start];
            const p2 = landmarks[end];
            canvasCtx.beginPath();
            canvasCtx.moveTo(p1.x * videoWidth, p1.y * videoHeight);
            canvasCtx.lineTo(p2.x * videoWidth, p2.y * videoHeight);
            canvasCtx.stroke();
        }

        // draws joints
        canvasCtx.fillStyle = "#FF0000";
        for (const point of landmarks) {
          canvasCtx.beginPath();
          canvasCtx.arc(point.x * videoWidth, point.y * videoHeight, 4, 0, 2 * Math.PI); 
          canvasCtx.fill();
        }
      }
    }
    canvasCtx.restore();
  };

  useEffect(() => {
    let hands: any;
    let camera: any;
    const loadModels = async () => {
      const { Hands } = await import("@mediapipe/hands");
      const { Camera } = await import("@mediapipe/camera_utils");
      hands = new Hands({
        locateFile: (file) => `https://cdn.jsdelivr.net/npm/@mediapipe/hands/${file}`,
      });
      hands.setOptions({
        maxNumHands: 1,
        modelComplexity: 1,
        minDetectionConfidence: 0.5,
        minTrackingConfidence: 0.5,
      });
      hands.onResults(onResults);
      if (typeof window !== "undefined" && webcamRef.current && webcamRef.current.video) {
        camera = new Camera(webcamRef.current.video, {
          onFrame: async () => {
            if (webcamRef.current?.video) await hands.send({ image: webcamRef.current.video });
          },
          width: 640,
          height: 480,
        });
        camera.start();
        setCameraReady(true);
      }
    };
    loadModels();
    return () => { if (camera) camera.stop(); if (hands) hands.close(); };
  }, []);

  const downloadData = () => {
    const dataStr = "data:text/json;charset=utf-8," + encodeURIComponent(JSON.stringify(collectedData.current));
    const downloadAnchorNode = document.createElement('a');
    downloadAnchorNode.setAttribute("href", dataStr);
    downloadAnchorNode.setAttribute("download", "asl_dataset.json");
    document.body.appendChild(downloadAnchorNode);
    downloadAnchorNode.click();
    downloadAnchorNode.remove();
  };

  return (
    <div className="flex flex-col items-center gap-6 p-4">
        
        {/* video preview */}
        <div className="relative w-[640px] h-[480px] border-2 border-gray-800 rounded-lg overflow-hidden bg-black shadow-lg">
            <Webcam ref={webcamRef} className="absolute top-0 left-0 w-full h-full object-cover" mirrored={true} />
            <canvas ref={canvasRef} className="absolute top-0 left-0 w-full h-full object-cover z-10" />
            {!cameraReady && <div className="absolute top-0 left-0 w-full h-full flex items-center justify-center z-20 bg-black/80 text-white font-mono">Initializing...</div>}
            
            {/* prediction overlay */}
            {mode === 'PREDICT' && (
                <div className="absolute top-4 right-4 bg-black/60 text-white px-4 py-2 rounded font-mono text-xl z-20 border border-white/20">
                    Prediction: <span className="text-green-400 font-bold">{prediction}</span>
                </div>
            )}
        </div>

        {/* controls */}
        <div className="w-[640px] p-4 border border-gray-700 rounded-lg bg-gray-900 text-white shadow-lg flex flex-col gap-4">
            
            {/* mode switcher */}
            <div className="flex justify-center bg-gray-800 p-1 rounded-lg self-center">
                <button 
                    onClick={() => setMode('PREDICT')}
                    className={`px-6 py-2 rounded-md font-bold transition-all ${mode === 'PREDICT' ? 'bg-green-600 text-white shadow' : 'text-gray-400 hover:text-white'}`}
                >
                    PREDICT
                </button>
                <button 
                    onClick={() => setMode('COLLECT')}
                    className={`px-6 py-2 rounded-md font-bold transition-all ${mode === 'COLLECT' ? 'bg-blue-600 text-white shadow' : 'text-gray-400 hover:text-white'}`}
                >
                    COLLECT
                </button>
            </div>

            {/* collect mode controls */}
            {mode === 'COLLECT' && (
                <div className="flex flex-row gap-4 items-center justify-center animate-in fade-in slide-in-from-top-4 duration-300">
                    <input 
                        type="text" placeholder="Label (A)" 
                        className="text-black text-center font-bold px-4 py-2 rounded w-24"
                        value={recordingLabel}
                        onChange={(e) => setRecordingLabel(e.target.value.toUpperCase())}
                        maxLength={1}
                    />
                    <button
                        onClick={() => {
                            if (!recordingLabel) return alert("Enter label!");
                            setIsRecording(!isRecording);
                        }}
                        className={`px-6 py-2 rounded font-bold w-32 ${isRecording ? 'bg-red-600 animate-pulse' : 'bg-blue-500 hover:bg-blue-600'}`}
                    >
                        {isRecording ? "STOP" : "START"}
                    </button>
                    <div className="text-mono text-gray-400 text-sm">Samples: <span className="text-white font-bold">{sampleCount}</span></div>
                    <button onClick={downloadData} className="ml-auto text-xs underline text-gray-400 hover:text-white">Download JSON</button>
                </div>
            )}
            
            {/* predict mode info */}
            {mode === 'PREDICT' && (
                <div className="text-center text-sm text-gray-400">
                    Model Status: <span className={model ? "text-green-400" : "text-yellow-400"}>{model ? "Active" : "Loading..."}</span>
                    <span className="mx-2">|</span>
                    Classes: {classes.length > 0 ? classes.join(", ") : "None"}
                </div>
            )}
        </div>
    </div>
  );
}