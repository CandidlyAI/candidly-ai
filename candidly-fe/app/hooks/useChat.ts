import { useState, useRef, useEffect } from "react";
import { useUploadAudio } from "./use-upload-audio";
import { usePollAudio } from "./usePollAudio";
import { blobToWav } from "../lib/encodeWav";

export function useChat({ uploadEndpoint }: { uploadEndpoint: string }) {
    const [isRecording, setIsRecording] = useState(false);
    const [isWaitingForChatBot, setIsWaitingForChatBot] = useState(false)
    const mediaRecorderRef = useRef<MediaRecorder | null>(null);
    const chunksRef = useRef<Blob[]>([]);
    const uploadAudio = useUploadAudio({ endpoint: uploadEndpoint});

    const { data: audioUrl, isFetching } = usePollAudio("recording.wav", isWaitingForChatBot)

    const audioRef = useRef<HTMLAudioElement | null>(null);

    // Auto-play when audio URL changes
    useEffect(() => {
        if (audioUrl && audioRef.current) {
            audioRef.current.load();
            audioRef.current.play().then(() => {
                setIsWaitingForChatBot(false)
            }).catch(() => {
                console.warn("Autoplay blocked by browser.");
            });
        }
    }, [audioUrl]);

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

    console.log(isWaitingForChatBot)

    function stopRecording() {
        mediaRecorderRef.current?.stop();
        mediaRecorderRef.current?.stream.getTracks().forEach((track) => track.stop());
        setIsRecording(false);
        setIsWaitingForChatBot(true);
    }

    return {
        startRecording,
        stopRecording,
        isRecording,
        uploadAudio,
        audioRef,
        audioUrl
    }
}