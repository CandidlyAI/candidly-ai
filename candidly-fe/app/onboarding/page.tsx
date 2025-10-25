"use client";

import { useState, useRef } from "react";
import { Button } from "@/components/ui/button";
import { useUploadAudio } from "@/app/hooks/use-upload-audio";
import { blobToWav } from "@/app/lib/encodeWav";

export default function LandingPage() {
  const [isRecording, setIsRecording] = useState(false);
  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const chunksRef = useRef<Blob[]>([]);
  const uploadAudio = useUploadAudio();

  async function startRecording() {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      const mediaRecorder = new MediaRecorder(stream);
      mediaRecorderRef.current = mediaRecorder;
      chunksRef.current = [];

      mediaRecorder.ondataavailable = (event) => {
        if (event.data.size > 0) chunksRef.current.push(event.data);
      };

      mediaRecorder.onstop = async () => {
        const audioBlob = new Blob(chunksRef.current, { type: "audio/webm" });
        const wavBlob = await blobToWav(audioBlob); // Convert to WAV
        uploadAudio.mutate(wavBlob);
      };

      mediaRecorder.start();
      setIsRecording(true);
    } catch (err) {
      console.error("Microphone error:", err);
      alert("Unable to access microphone");
    }
  }

  function stopRecording() {
    mediaRecorderRef.current?.stop();
    mediaRecorderRef.current?.stream.getTracks().forEach((track) => track.stop());
    setIsRecording(false);
  }

  return (
    <main className="flex min-h-screen items-center justify-center bg-background">
      <div className="text-center">
        <h1 className="text-3xl font-bold mb-8">Onboarding Recording</h1>

        <Button
          size="lg"
          variant={isRecording ? "destructive" : "default"}
          onClick={isRecording ? stopRecording : startRecording}
          className="rounded-full w-32 h-32 text-lg"
        >
          {isRecording ? "Stop" : "Record"}
        </Button>

        <p className="text-sm text-muted-foreground mt-6">
          {isRecording
            ? "Recording... click to stop."
            : "Click to start recording (saves as WAV)."}
        </p>

        {uploadAudio.isLoading && <p className="mt-4 text-sm">Uploading...</p>}
        {uploadAudio.isSuccess && (
          <p className="mt-4 text-green-500">
            ✅ Uploaded: {uploadAudio.data.filename}
          </p>
        )}
        {uploadAudio.isError && (
          <p className="mt-4 text-red-500">❌ Upload failed.</p>
        )}
      </div>
    </main>
  );
}
