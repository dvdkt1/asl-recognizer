// src/app/page.tsx
import CameraView from "@/components/CameraView";

export default function Home() {
  return (
    <main className="flex min-h-screen flex-col items-center justify-center p-8 bg-gray-950 text-white">
      <h1 className="text-3xl font-bold mb-6 text-blue-400 tracking-tight">
        ASL Recognition <span className="text-gray-500 text-lg font-normal">| MVP Build</span>
      </h1>
      
      <div className="p-2 border border-gray-800 rounded-xl bg-gray-900 shadow-2xl">
        <CameraView />
      </div>

      <div className="mt-6 grid grid-cols-2 gap-4 text-sm text-gray-400 font-mono">
        <div className="bg-gray-900 p-3 rounded border border-gray-800">
          <span className="text-green-400">●</span> MediaPipe Active
        </div>
        <div className="bg-gray-900 p-3 rounded border border-gray-800">
          <span className="text-yellow-400">●</span> Training Mode: OFF
        </div>
      </div>
    </main>
  );
}