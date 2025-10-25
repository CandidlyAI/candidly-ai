"use client";

import { useQuery } from "@tanstack/react-query";
import api from "@/app/lib/axios";

export function usePollAudio(filename: string, enabled: boolean) {
  return useQuery({
    queryKey: ["poll-audio", filename],
    queryFn: async () => {
      const { data } = await api.get(`/download-audio/${filename}`);
      return data.url as string | null;
    },
    refetchInterval: 5000, // poll every 5 seconds
    refetchIntervalInBackground: true,
    enabled
  });
}
