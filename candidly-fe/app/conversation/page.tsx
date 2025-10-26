"use client";

import { useEffect, useRef, useState } from "react";
import { useScenario } from "@/app/hooks/useScenario";
import { usePollAudio } from "@/app/hooks/usePollAudio";
import { Avatar, AvatarFallback } from "@/components/ui/avatar";
import { Button } from "@/components/ui/button";
import { User } from "lucide-react";
import { useChat } from "../hooks/useChat";
import api from "@/app/lib/axios";

export default function ConversationPage() {
  const { data: scenario, isLoading, isError } = useScenario();
  const [conversationReady, setConversationReady] = useState(false);
  const [pollingEnabled, setPollingEnabled] = useState(false);

  const {
    isRecording,
    startRecording,
    stopRecording,
    uploadAudio,
    audioRef,
    audioUrl,
  } = useChat({ uploadEndpoint: "/conversation/turn" });

  const { data: ttsUrl, isFetching } = usePollAudio("recording.wav", pollingEnabled);

  // 🟢 Auto-play TTS from backend
  useEffect(() => {
    if (ttsUrl && audioRef.current) {
      audioRef.current.load();
      audioRef.current.play().catch(() => {
        console.warn("Autoplay blocked by browser.");
      });
    }
  }, [ttsUrl]);

  // 🟢 Initialize conversation (backend state)
  useEffect(() => {
    async function initConversation() {
      try {
        const { data } = await api.get("/scenario"); // get current onboarding scenario
        await api.post("/conversation/reset", {
          role: "user",
          ai_role: "stakeholder",
          scenario: data?.scenario ?? "",
        });
        setConversationReady(true);
      } catch (err) {
        console.error("Failed to initialize conversation:", err);
      }
    }
    initConversation();
  }, []);

  // 🟢 When user uploads a recording, start polling for TTS
  useEffect(() => {
    if (uploadAudio.isSuccess) {
      setPollingEnabled(true);
    }
  }, [uploadAudio.isSuccess]);

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
          className={`w-40 h-40 border-4 shadow-lg flex items-center justify-center transition-all duration-500 ${
            ttsUrl ? "animate-pulse border-green-500" : "border-primary"
          }`}
        >
          <AvatarFallback className="bg-primary/10 text-primary">
            <User size={64} />
          </AvatarFallback>
        </Avatar>

        <p className="mt-4 text-lg text-muted-foreground">
          {isRecording
            ? "Recording... click to stop."
            : pollingEnabled
            ? isFetching
              ? "Generating reply..."
              : ttsUrl
              ? "Now playing response..."
              : "Ready for your next turn."
            : conversationReady
            ? "Click Record to start speaking."
            : "Loading conversation..."}
        </p>

        <div className="mt-6 flex gap-4">
          {conversationReady && (
            <Button
              onClick={isRecording ? stopRecording : startRecording}
              variant={isRecording ? "destructive" : "default"}
              size="lg"
              className="rounded-full w-32 h-32 text-lg"
            >
              {isRecording ? "Stop" : "Record"}
            </Button>
          )}
        </div>

        {/* Hidden audio element */}
        <audio ref={audioRef} className="hidden">
          {ttsUrl && <source src={ttsUrl} type="audio/wav" />}
        </audio>
      </div>
    </main>
  );
}
