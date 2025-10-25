"use client";

import { useQuery } from "@tanstack/react-query";
import api from "@/app/lib/axios";

export function useScenario() {
  return useQuery({
    queryKey: ["scenario"],
    queryFn: async () => {
      const { data } = await api.get("/scenario");
      return data.scenario as string;
    },
  });
}
