"use client";

import { useEffect } from "react";
import { useScenario } from "@/app/hooks/useScenario";
import { Avatar, AvatarFallback } from "@/components/ui/avatar";
import { Button } from "@/components/ui/button";
import { User } from "lucide-react";
import { useChat } from "../hooks/useChat";
import api from "@/app/lib/axios";
import { useRouter } from "next/navigation";

export default function ConversationPage() {
  const router = useRouter()
  const { data: scenario, isLoading, isError } = useScenario();

  const {
    isRecording,
    startRecording,
    stopRecording,
    uploadAudio,
    audioRef,
    audioUrl,
  } = useChat({ uploadEndpoint: "/conversation/turn" });


  // 🟢 Initialize conversation (backend state)
  useEffect(() => {
    async function initConversation() {
      try {
        const { data } = await api.get("/scenario"); // get current onboarding scenario
        await api.post("/conversation/reset");
      } catch (err) {
        console.error("Failed to initialize conversation:", err);
      }
    }
    initConversation();
  }, []);

  return (
    <main className="relative flex min-h-screen items-center justify-center bg-background text-foreground">
      {/* Top-left: Scenario text */}
      <div className="absolute top-6 left-6 max-w-sm p-4 bg-card rounded-lg shadow">
        {isLoading && <p className="text-muted-foreground">Loading scenario...</p>}
        {isError && <p className="text-red-500">Failed to load scenario.</p>}
        {scenario && (
          <p className="text-sm font-medium leading-relaxed">
            <span className="font-semibold">Scenario:</span> {scenario}
          </p>
        )}
      </div>

      {/* Center: Avatar + controls */}
      <div className="flex flex-col items-center">
        <Avatar
          className={`w-40 h-40 border-4 shadow-lg flex items-center justify-center transition-all duration-500 ${audioUrl ? "animate-pulse border-green-500" : "border-primary"}`}
        >
          <AvatarFallback className="bg-primary/10 text-primary">
            <User size={64} />
          </AvatarFallback>
        </Avatar>

        <p className="text-sm text-muted-foreground mt-6">
          {
            isRecording
              ? "Recording... click to stop."
              : "Click to start recording (saves as WAV)."
          }
        </p>

        <div className="mt-6 flex gap-4">
          <Button
            onClick={isRecording ? stopRecording : startRecording}
            variant={isRecording ? "destructive" : "default"}
            size="lg"
            className="rounded-full w-32 h-32 text-lg"
          >
            {isRecording ? "Stop" : "Record"}
          </Button>
                    {/* 🧭 Finish Conversation Button */}
        </div>
          <Button
            onClick={() => router.push("/evaluation")}
            variant="secondary"
            className="mt-6"
          >
            Finish Conversation
          </Button>

        {/* Hidden audio element */}
        <audio ref={audioRef} className="hidden">
          {audioUrl && <source src={audioUrl} type="audio/wav" />}
        </audio>
      </div>
    </main>
  );
}
