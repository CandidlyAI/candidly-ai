"use client";

import { useState, useRef, useEffect } from "react";
import { Button } from "@/components/ui/button";
import { useUploadAudio } from "@/app/hooks/use-upload-audio";
import { blobToWav } from "@/app/lib/encodeWav";
import { usePollAudio } from "../hooks/usePollAudio";
import { useChat } from "../hooks/useChat";
import { useRouter } from "next/navigation"; // ✅ Import router

export default function LandingPage() {
  const router = useRouter(); // ✅ Initialize router
  const { isRecording, stopRecording, startRecording, uploadAudio, audioRef, audioUrl } = useChat({ uploadEndpoint: "/onboarding" })
  
  useEffect(() => {
    if (uploadAudio?.data?.is_done) {
      router.push("/conversation");
    }
  }, [uploadAudio?.data?.is_done, router]);

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

        {uploadAudio.isPending && <p className="mt-4 text-sm">Uploading...</p>}
        {uploadAudio.isSuccess && (
          <p className="mt-4 text-green-500">
            ✅ Uploaded: {uploadAudio.data.filename}
          </p>
        )}
        {uploadAudio.isError && (
          <p className="mt-4 text-red-500">❌ Upload failed.</p>
        )}
      </div>
        <audio ref={audioRef} className="hidden">
          {audioUrl && <source src={audioUrl} type="audio/wav" />}
        </audio>
    </main>
  );
}
