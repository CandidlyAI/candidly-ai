"use client";

import { useEffect, useRef, useState } from "react";
import { useScenario } from "@/app/hooks/useScenario";
import { usePollAudio } from "@/app/hooks/usePollAudio";
import { Avatar, AvatarFallback } from "@/components/ui/avatar";
import { Button } from "@/components/ui/button";
import { User } from "lucide-react";

export default function ConversationPage() {
  const { data: scenario, isLoading, isError } = useScenario();
  const [pollingEnabled, setPollingEnabled] = useState(false);

  const { data: audioUrl, isFetching } = usePollAudio("recording.wav", pollingEnabled);
  const audioRef = useRef<HTMLAudioElement | null>(null);

  // Auto-play when audio URL changes
  useEffect(() => {
    if (audioUrl && audioRef.current) {
      audioRef.current.load();
      audioRef.current.play().catch(() => {
        console.warn("Autoplay blocked by browser.");
      });
    }
  }, [audioUrl]);

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
            audioUrl ? "animate-pulse border-green-500" : "border-primary"
          }`}
        >
          <AvatarFallback className="bg-primary/10 text-primary">
            <User size={64} />
          </AvatarFallback>
        </Avatar>

        <p className="mt-4 text-lg text-muted-foreground">
          {pollingEnabled
            ? isFetching
              ? "Checking for new audio..."
              : audioUrl
              ? "Now playing response..."
              : "Listening for new audio..."
            : "Press Start to begin polling."}
        </p>

        {/* Start / Stop polling buttons */}
        <div className="mt-6 flex gap-4">
          {!pollingEnabled ? (
            <Button onClick={() => setPollingEnabled(true)}>Start</Button>
          ) : (<></>
          )}
        </div>

        {/* Hidden audio element */}
        <audio ref={audioRef} className="hidden">
          {audioUrl && <source src={audioUrl} type="audio/wav" />}
        </audio>
      </div>
    </main>
  );
}
