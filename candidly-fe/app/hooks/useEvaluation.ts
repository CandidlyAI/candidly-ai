import { useQuery } from "@tanstack/react-query";
import api from "../lib/axios";

export function useEvaluation() {
    return useQuery({
        queryKey: ["evaluation"],
        queryFn: async () => {
            const { data } = await api.post("/conversation/evaluate", {});
            return data;
        },
    });
}

