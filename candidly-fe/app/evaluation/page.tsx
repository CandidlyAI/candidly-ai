"use client";

import { useEvaluation } from "../hooks/useEvaluation"



interface EvaluationData {
    rubric_version: string;
    overall_score: number;
    scores: {
        empathy: number;
        clarity: number;
        relevance: number;
        guidance: number;
        professionalism: number;
    };
    strengths: string[];
    improvements: string[];
    summary: string;
    tokens_hint: number;
}

export default function Evaluation() {
    const { data } = useEvaluation()

    if (!data) {
        return <div>Loading...</div>
    }

    return (
        <EvaluationResult data={data} />
    );
}

export function EvaluationResult({ data }: { data: EvaluationData }) {

    return (
        <div className="max-w-3xl mx-auto bg-white rounded-xl shadow-md p-6 space-y-6 mt-10">
            <h2 className="text-2xl font-bold text-center">Evaluation Summary</h2>

            {/* Overall Score */}
            <div className="text-center">
                <p className="text-lg font-semibold">
                    Overall Score: <span className="text-blue-600">{data.overall_score}/5</span>
                </p>
                <p className="text-sm text-gray-500">Rubric version: {data.rubric_version}</p>
            </div>

            {/* Subscores */}
            <div>
                <h3 className="text-xl font-semibold mb-2">Detailed Scores</h3>
                <div className="grid grid-cols-2 gap-2 text-sm">
                    {Object.entries(data.scores).map(([category, score]) => (
                        <div key={category} className="flex justify-between border-b py-1">
                            <span className="capitalize">{category}</span>
                            <span className="font-medium">{score}/5</span>
                        </div>
                    ))}
                </div>
            </div>

            {/* Strengths */}
            <div>
                <h3 className="text-xl font-semibold mb-2 text-green-700">✅ Strengths</h3>
                <ul className="list-disc list-inside text-gray-700 space-y-1">
                    {data.strengths.map((point, i) => (
                        <li key={i}>{point}</li>
                    ))}
                </ul>
            </div>

            {/* Improvements */}
            <div>
                <h3 className="text-xl font-semibold mb-2 text-red-700">⚠️ Improvements</h3>
                <ul className="list-disc list-inside text-gray-700 space-y-1">
                    {data.improvements.map((point, i) => (
                        <li key={i}>{point}</li>
                    ))}
                </ul>
            </div>

            {/* Summary */}
            <div className="border-t pt-3">
                <h3 className="text-xl font-semibold mb-2">📝 Summary</h3>
                <p className="text-gray-800 leading-relaxed">{data.summary}</p>
            </div>

            <p className="text-xs text-gray-400 text-right">
                Tokens processed: {data.tokens_hint}
            </p>
        </div>
    );

}
