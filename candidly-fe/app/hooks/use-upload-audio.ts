"use client";

import { useMutation } from "@tanstack/react-query";
import api from "@/app/lib/axios";

export function useUploadAudio() {
  return useMutation({
    mutationFn: async (audioBlob: Blob) => {
      const formData = new FormData();
      formData.append("file", audioBlob, "recording.wav");
      const { data } = await api.post("/upload-audio/", formData, {
        headers: { "Content-Type": "multipart/form-data" },
      });
      return data;
    },
  });
}
