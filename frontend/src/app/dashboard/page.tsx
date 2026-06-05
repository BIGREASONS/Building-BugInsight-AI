"use client";

import { useEffect, useState, useRef, useCallback, Suspense } from "react";
import { useSearchParams } from "next/navigation";

const API_BASE = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

// --- Types ---

interface AgentStatus {
  name: string;
  status: "pending" | "running" | "complete";
  completedAt?: number;
  duration?: number;
}

interface SwarmResults {
  index_stats?: {
    status?: string;
    files_indexed: number;
    language_count: Record<string, number>;
    size_mb: number;
    indexed_at?: number;
  };
  scanner_findings?: Array<{
    tool: string;
    rule: string;
    severity: string;
    file: string;
    line: number;
    description: string;
  }>;
  severity: string;
  confidence: number;
  root_cause: string;
  affected_file: string;
  vulnerable_code: string;
  exploit_example: string;
  risk_if_unfixed: string;
  suspected_files: string[];
  patch: string;
  fix_summary: string;
  patched_code: string;
  risk_assessment: string;
  pr_url: string;
  pr_mode: "mock" | "live";
  story_points: number;
  priority: string;
  sprint_recommendation: string;
  time_saved_hours: number;
  severity_reasoning: string[];
  regression_tests: string;
  validation_score: number;
  is_patch_valid: boolean;
  validation_reasoning: string;
  test_results: {
    passed: number;
    failed: number;
    stdout: string;
    stderr: string;
    duration: number;
  };
  tests_passed: boolean;
  rescan_passed: boolean;
}

const AGENT_ORDER = [
  "Repository Agent",
  "Scanner Agent",
  "Severity Agent",
  "Root Cause Agent",
  "Fix Agent",
  "Validation Agent",
  "Test Agent",
  "Test Execution Agent",
  "Auto-Rescan Agent",
  "GitHub Agent",
  "Sprint Agent",
];

const SEVERITY_COLORS: Record<string, string> = {
  Critical: "text-red-500",
  Major: "text-orange-500",
  Minor: "text-yellow-500",
  Trivial: "text-green-500",
};

const SEVERITY_RING: Record<string, string> = {
  Critical: "ring-red-500/40",
  Major: "ring-orange-500/40",
  Minor: "ring-yellow-500/40",
  Trivial: "ring-green-500/40",
};

const SEVERITY_BG: Record<string, string> = {
  Critical: "bg-red-500/10 border-red-500/30",
  Major: "bg-orange-500/10 border-orange-500/30",
  Minor: "bg-yellow-500/10 border-yellow-500/30",
  Trivial: "bg-green-500/10 border-green-500/30",
};

// --- Dashboard Content (uses useSearchParams) ---

function DashboardContent() {
  const searchParams = useSearchParams();
  const jobId = searchParams.get("job");

  const [agents, setAgents] = useState<AgentStatus[]>(
    AGENT_ORDER.map((name) => ({ name, status: "pending" }))
  );
  const [results, setResults] = useState<Partial<SwarmResults>>({});
  const [currentAgent, setCurrentAgent] = useState<string | null>(null);
  const [done, setDone] = useState(false);
  const [error, setError] = useState("");
  const [swarmStartTime, setSwarmStartTime] = useState<number | null>(null);
  const [elapsed, setElapsed] = useState(0);
  const lastAgentTime = useRef<number>(Date.now());

  // Timer — only starts counting when the first SSE event arrives
  useEffect(() => {
    if (done || !swarmStartTime) return;
    const interval = setInterval(() => {
      setElapsed(Math.floor((Date.now() - swarmStartTime) / 1000));
    }, 200);
    return () => clearInterval(interval);
  }, [done, swarmStartTime]);

  const markAgent = useCallback(
    (agentName: string, status: "running" | "complete", eventTimeMs?: number) => {
      const timeMs = eventTimeMs || Date.now();
      
      if (status === "complete") {
        const duration = (timeMs - lastAgentTime.current) / 1000;
        lastAgentTime.current = timeMs;
        
        setAgents((prev) =>
          prev.map((a) => {
            if (a.name === agentName) {
              // Ensure at least 0.1s is shown so it doesn't look like 0s bug
              return { ...a, status, completedAt: timeMs, duration: Math.max(0.1, Math.round(duration * 10) / 10) };
            }
            if (AGENT_ORDER.indexOf(a.name) < AGENT_ORDER.indexOf(agentName) && a.status !== "complete") {
              return { ...a, status: "complete" };
            }
            return a;
          })
        );
      } else {
        setAgents((prev) =>
          prev.map((a) => {
            if (a.name === agentName) {
              return { ...a, status };
            }
            if (AGENT_ORDER.indexOf(a.name) < AGENT_ORDER.indexOf(agentName) && a.status !== "complete") {
              return { ...a, status: "complete" };
            }
            return a;
          })
        );
      }
    },
    []
  );

  // SSE connection
  useEffect(() => {
    if (!jobId) return;

    const es = new EventSource(`${API_BASE}/api/swarm/stream/${jobId}`);

    es.onmessage = (event) => {
      try {
        const msg = JSON.parse(event.data);

        if (msg.type === "swarm_started") {
          const startTimeMs = msg.timestamp ? msg.timestamp * 1000 : Date.now();
          setSwarmStartTime(startTimeMs);
          lastAgentTime.current = startTimeMs;
          if (msg.data?.agents?.length > 0) {
            markAgent(msg.data.agents[0], "running", startTimeMs);
            setCurrentAgent(msg.data.agents[0]);
          }
        } else if (msg.type === "agent_complete") {
          const agentName = msg.data?.agent;
          if (agentName) {
            const completeTimeMs = msg.timestamp ? msg.timestamp * 1000 : Date.now();
            markAgent(agentName, "complete", completeTimeMs);

            const idx = AGENT_ORDER.indexOf(agentName);
            if (idx >= 0 && idx < AGENT_ORDER.length - 1) {
              const next = AGENT_ORDER[idx + 1];
              markAgent(next, "running", completeTimeMs);
              setCurrentAgent(next);
            }

            // Merge partial results
            setResults((prev) => ({ ...prev, ...msg.data }));
          }
        } else if (msg.type === "swarm_complete") {
          const completeTimeMs = msg.timestamp ? msg.timestamp * 1000 : Date.now();
          setSwarmStartTime((prevStart) => {
            if (prevStart) {
              setElapsed(Math.floor((completeTimeMs - prevStart) / 1000));
            }
            return prevStart;
          });
          setResults((prev) => ({ ...prev, ...msg.data }));
          setDone(true);
          setCurrentAgent(null);
          es.close();
        } else if (msg.type === "swarm_error") {
          setError(msg.data?.error || "Unknown error");
          setDone(true);
          es.close();
        }
      } catch {
        // Ignore parse errors
      }
    };

    es.onerror = () => {
      if (done) es.close();
    };

    return () => es.close();
  }, [jobId, done, markAgent]);

  if (!jobId) {
    return (
      <div className="flex-1 flex items-center justify-center">
        <p className="text-zinc-500">No job ID provided. Go back and start a swarm.</p>
      </div>
    );
  }

  const severity = results.severity || "";
  const confidence = results.confidence || 0;
  const severityColor = SEVERITY_COLORS[severity] || "text-zinc-400";
  const severityBg = SEVERITY_BG[severity] || "bg-zinc-900 border-zinc-800";
  const severityRing = SEVERITY_RING[severity] || "";

  // Determine effective PR mode — force mock if URL contains "mock" or "Error"
  const rawPrUrl = results.pr_url || "";
  const effectivePrMode =
    rawPrUrl.startsWith("mock://") ||
    rawPrUrl.includes("mock-org") ||
    rawPrUrl.toLowerCase().includes("error")
      ? "mock"
      : results.pr_mode || "mock";

  // Parse time saved from sprint recommendation
  const timeSavedMatch = results.sprint_recommendation?.match(
    /(\d+\.?\d*)\s*hours?/i
  );
  const timeSaved = timeSavedMatch ? parseFloat(timeSavedMatch[1]) : 0;

  return (
    <main className="flex-1 flex flex-col min-h-screen">
      {/* Top Bar */}
      <header className="border-b border-zinc-800 bg-zinc-950/80 backdrop-blur-sm px-6 py-3 flex items-center justify-between">
        <div className="flex items-center gap-3">
          <a href="/" className="text-lg font-semibold hover:opacity-80 transition-opacity">
            Bug<span className="text-indigo-400">Insight</span> Swarm
          </a>
          <span className="text-xs text-zinc-500 font-mono bg-zinc-900 px-2 py-0.5 rounded">
            Session: {jobId}
          </span>
          <div className={`ml-4 px-3 py-1 text-xs font-bold uppercase tracking-widest rounded-full border ${
            effectivePrMode === "live" 
              ? "bg-red-500/10 text-red-400 border-red-500/30" 
              : "bg-emerald-500/10 text-emerald-400 border-emerald-500/30"
          }`}>
            {effectivePrMode === "live" ? "Live GitHub Mode" : "Safe Demo Mode"}
          </div>
        </div>
        <div className="flex items-center gap-4 text-sm text-zinc-400">
          <span className="font-mono tabular-nums">{elapsed}s</span>
          {done ? (
            <span className="inline-flex items-center gap-1.5 text-green-400">
              <span className="h-2 w-2 rounded-full bg-green-500" />
              Complete
            </span>
          ) : (
            <span className="inline-flex items-center gap-1.5 text-indigo-400">
              <span className="h-2 w-2 rounded-full bg-indigo-500 animate-pulse" />
              Running
            </span>
          )}
        </div>
      </header>

      {/* Error Banner */}
      {error && (
        <div className="bg-red-950/50 border-b border-red-800 px-6 py-3 text-sm text-red-300">
          Swarm Error: {error}
        </div>
      )}

        {/* ── Horizontal Trace Bar ── */}
        <div className="border-b border-zinc-800 bg-zinc-950 px-6 py-3 flex items-center justify-between">
          <div className="flex items-center gap-4 overflow-x-auto whitespace-nowrap scrollbar-hide flex-1">
            {agents.map((agent, i) => (
              <div key={agent.name} className="flex items-center gap-3">
                <div className={`flex items-center gap-2 px-3 py-1.5 rounded-md border ${
                    agent.status === "complete"
                      ? "bg-zinc-900 border-zinc-800 text-zinc-300"
                      : agent.status === "running"
                      ? "bg-indigo-950/50 border-indigo-500/30 text-indigo-300"
                      : "border-transparent text-zinc-600"
                  }`}>
                  {agent.status === "complete" && (
                    <svg className="w-4 h-4 text-green-500" fill="none" viewBox="0 0 24 24" strokeWidth={2.5} stroke="currentColor">
                      <path strokeLinecap="round" strokeLinejoin="round" d="M4.5 12.75l6 6 9-13.5" />
                    </svg>
                  )}
                  {agent.status === "running" && (
                    <span className="inline-block h-3 w-3 rounded-full border-2 border-indigo-400 border-t-transparent animate-spin" />
                  )}
                  {agent.status === "pending" && (
                    <span className="inline-block h-3 w-3 rounded border border-zinc-700 bg-zinc-800" />
                  )}
                  <span className="text-sm font-medium">{agent.name}</span>
                  {agent.status === "complete" && agent.duration !== undefined && (
                    <span className="text-xs font-mono text-zinc-500">({agent.duration}s)</span>
                  )}
                </div>
                {i < agents.length - 1 && <span className="text-zinc-700">→</span>}
              </div>
            ))}
          </div>
          <div className="text-xs text-indigo-400 animate-pulse font-medium ml-4 shrink-0">
            {done
              ? "All agents completed successfully."
              : currentAgent === "Repository Agent"
              ? "🔍 Indexing repository..."
              : currentAgent === "Scanner Agent"
              ? "🔎 Running static analysis..."
              : currentAgent === "Severity Agent"
              ? "🧠 Running CodeBERT analysis..."
              : currentAgent === "Root Cause Agent"
              ? "🎯 Locating vulnerable file..."
              : currentAgent === "Fix Agent"
              ? "🛠 Generating secure patch..."
              : currentAgent === "Validation Agent"
              ? "🛡️ Validating patch against findings..."
              : currentAgent === "Test Agent"
              ? "🧪 Writing regression tests..."
              : currentAgent === "Test Execution Agent"
              ? "⚡ Running regression tests..."
              : currentAgent === "Auto-Rescan Agent"
              ? "🔎 Verifying vulnerability removed..."
              : currentAgent === "Sprint Agent"
              ? "📋 Creating sprint plan..."
              : currentAgent
              ? `Processing: ${currentAgent}…`
              : "Connecting…"}
          </div>
        </div>

      {/* ── Main Dashboard Content ── */}
      <div className="flex-1 p-6 space-y-6 max-w-[1600px] mx-auto w-full">
        
        {/* ── TOP ROW: Executive Summary ── */}
        <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-4 gap-6">
          
          {/* Repo Stats */}
          <div className="rounded-2xl border border-zinc-800 bg-zinc-900/50 p-6 flex flex-col justify-center">
            <h3 className="text-xs font-semibold uppercase tracking-wider text-zinc-500 mb-6 flex items-center gap-2">
              <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 7v10c0 2.21 3.582 4 8 4s8-1.79 8-4V7M4 7c0 2.21 3.582 4 8 4s8-1.79 8-4M4 7c0-2.21 3.582-4 8-4s8 1.79 8 4m0 5c0 2.21-3.582 4-8 4s-8-1.79-8-4" />
              </svg>
              Repository Indexed
              {results.index_stats?.status === "Cached" && (
                <span className="ml-auto inline-flex items-center gap-1 rounded-full bg-green-950/50 border border-green-800/50 px-2 py-0.5 text-[10px] font-semibold uppercase tracking-wider text-green-400">
                  ⚡ Cached
                </span>
              )}
            </h3>
            {results.index_stats ? (
              <div className="grid grid-cols-3 gap-4 text-center">
                <div>
                  <div className="text-3xl font-bold text-white">{results.index_stats.files_indexed}</div>
                  <div className="text-xs text-zinc-500 mt-1">Files</div>
                </div>
                <div>
                  <div className="text-3xl font-bold text-white">{Object.keys(results.index_stats.language_count || {}).length}</div>
                  <div className="text-xs text-zinc-500 mt-1">Languages</div>
                </div>
                <div>
                  <div className="text-3xl font-bold text-white">{results.index_stats.size_mb} MB</div>
                  <div className="text-xs text-zinc-500 mt-1">Size</div>
                </div>
              </div>
            ) : (
              <div className="grid grid-cols-3 gap-4">
                {[...Array(3)].map((_, i) => <div key={i} className="h-12 rounded bg-zinc-800 animate-pulse" />)}
              </div>
            )}
          </div>

          {/* Hero Severity */}
          <div className={`rounded-2xl border p-8 flex flex-col items-center justify-center text-center transition-all duration-700 ${severity ? `${severityBg} ring-2 ${severityRing} shadow-[0_0_50px_-15px_rgba(239,68,68,0.3)]` : "bg-zinc-900 border-zinc-800"}`}>
            <div className="inline-flex flex-wrap justify-center items-center gap-2 mb-6">
              <div className="inline-flex items-center gap-2 rounded-full bg-zinc-950/80 border border-zinc-800 px-4 py-1.5 text-xs shadow-sm">
                <span className={`h-2 w-2 rounded-full ${severity ? severityColor.replace('text-', 'bg-') : 'bg-indigo-500'}`} />
                <span className="text-zinc-300 uppercase tracking-widest font-bold">Severity Assessment</span>
              </div>
            </div>
            {severity ? (
              <>
                <div className={`text-4xl lg:text-5xl font-black uppercase tracking-tight ${severityColor}`}>{severity}</div>
                <div className={`text-6xl lg:text-7xl font-black tabular-nums mt-1 ${severityColor}`}>{Math.round(confidence * 100)}%</div>
                <div className="grid grid-cols-2 gap-8 text-sm mt-6 pt-6 border-t border-red-900/30 w-full px-4">
                  <div>
                    <div className="text-zinc-500 text-xs uppercase tracking-wider mb-1">Model</div>
                    <div className="font-mono font-medium text-zinc-200">CodeBERT</div>
                  </div>
                  <div>
                    <div className="text-zinc-500 text-xs uppercase tracking-wider mb-1">Inference</div>
                    <div className="font-mono font-medium text-zinc-200">Local</div>
                  </div>
                </div>
              </>
            ) : (
              <div className="space-y-4 w-full">
                <div className="h-12 w-3/4 mx-auto rounded-lg bg-zinc-800 animate-pulse" />
                <div className="h-16 w-1/2 mx-auto rounded-lg bg-zinc-800 animate-pulse" />
              </div>
            )}
          </div>

          {/* Engineering Impact */}
          <div className="rounded-2xl border border-zinc-800 bg-zinc-900/50 p-6 flex flex-col justify-center">
            <h3 className="text-xs font-semibold uppercase tracking-wider text-indigo-400 mb-6">
              Engineering Impact
            </h3>
            {done ? (
              <div className="grid grid-cols-2 gap-y-6 gap-x-4">
                <div>
                  <div className="text-3xl font-bold text-white">{timeSaved > 0 ? `${timeSaved}h` : "—"}</div>
                  <div className="text-xs text-zinc-500 mt-1">Time Saved</div>
                </div>
                <div>
                  <div className="text-3xl font-bold text-orange-400">{results.priority || "—"}</div>
                  <div className="text-xs text-zinc-500 mt-1">Priority</div>
                </div>
                <div>
                  <div className={`text-3xl font-bold ${results.patch ? "text-green-400" : "text-zinc-600"}`}>
                    {results.patch ? "Yes" : "No"}
                  </div>
                  <div className="text-xs text-zinc-500 mt-1">Patch Generated</div>
                </div>
                <div>
                  <div className="text-3xl font-bold text-white capitalize">{effectivePrMode}</div>
                  <div className="text-xs text-zinc-500 mt-1">PR Mode</div>
                </div>
              </div>
            ) : (
              </div>
            )}
          </div>
          
          {/* Patch Confidence */}
          <div className={`rounded-2xl border ${results.validation_score !== undefined ? (results.is_patch_valid ? 'border-green-900 bg-green-950/20' : 'border-red-900 bg-red-950/20') : 'border-zinc-800 bg-zinc-900/50'} p-6 flex flex-col`}>
            <h3 className="text-xs font-semibold uppercase tracking-wider text-blue-400 mb-6">
              Patch Confidence
            </h3>
            {results.validation_score !== undefined ? (
              <div className="flex flex-col h-full justify-between">
                <div>
                  <div className="flex items-end gap-2 mb-2">
                    <div className={`text-4xl font-bold ${results.is_patch_valid ? 'text-green-400' : 'text-red-400'}`}>
                      {results.validation_score}/100
                    </div>
                    <div className={`text-sm font-bold uppercase ${results.is_patch_valid ? 'text-green-500' : 'text-red-500'} pb-1`}>
                      {results.is_patch_valid ? 'APPROVED' : 'REJECTED'}
                    </div>
                  </div>
                  <p className="text-sm text-zinc-300 leading-relaxed mb-4">
                    {results.validation_reasoning}
                  </p>
                </div>
                {results.rescan_passed !== undefined && (
                  <div className={`mt-auto pt-4 border-t ${results.is_patch_valid ? 'border-green-900/30' : 'border-red-900/30'}`}>
                    <div className="flex items-center justify-between text-xs font-semibold uppercase">
                      <span className="text-zinc-500">Auto-Rescan</span>
                      <span className={results.rescan_passed ? 'text-green-400' : 'text-red-400'}>
                        {results.rescan_passed ? 'FINDINGS REMOVED ✓' : 'VULNERABILITY REMAINS ✗'}
                      </span>
                    </div>
                  </div>
                )}
              </div>
            ) : (
               <div className="space-y-4 w-full">
                  <div className="h-10 w-1/2 rounded bg-zinc-800 animate-pulse" />
                  <div className="h-16 w-full rounded bg-zinc-800 animate-pulse" />
               </div>
            )}
          </div>
        </div>

        {/* ── NEW ROW: Scanner Findings ── */}
        {results.scanner_findings && results.scanner_findings.length > 0 && (
          <div className="rounded-2xl border border-zinc-800 bg-zinc-900/50 p-6 shadow-sm">
            <h3 className="text-xs font-bold uppercase tracking-wider text-indigo-400 mb-4 flex items-center gap-2">
              <span className="text-indigo-500">🔎</span> Static Scanner Findings
            </h3>
            <div className="overflow-x-auto">
              <table className="w-full text-sm text-left">
                <thead className="text-xs text-zinc-500 uppercase bg-zinc-950">
                  <tr>
                    <th className="px-4 py-3 rounded-l-lg font-semibold tracking-wider">Tool</th>
                    <th className="px-4 py-3 font-semibold tracking-wider">Rule</th>
                    <th className="px-4 py-3 font-semibold tracking-wider">Severity</th>
                    <th className="px-4 py-3 font-semibold tracking-wider">File:Line</th>
                    <th className="px-4 py-3 rounded-r-lg font-semibold tracking-wider">Description</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-zinc-800/50">
                  {results.scanner_findings.map((f, i) => (
                    <tr key={i} className="hover:bg-zinc-800/20 transition-colors">
                      <td className="px-4 py-3 font-mono text-zinc-300">{f.tool}</td>
                      <td className="px-4 py-3 text-zinc-400">{f.rule}</td>
                      <td className="px-4 py-3">
                        <span className={`px-2 py-1 rounded text-[10px] font-bold tracking-wider uppercase border ${f.severity?.toUpperCase() === 'ERROR' || f.severity?.toUpperCase() === 'HIGH' ? 'bg-red-950/50 text-red-400 border-red-900/50' : 'bg-yellow-950/50 text-yellow-400 border-yellow-900/50'}`}>
                          {f.severity}
                        </span>
                      </td>
                      <td className="px-4 py-3 font-mono text-zinc-400">{f.file}:{f.line}</td>
                      <td className="px-4 py-3 text-zinc-300 max-w-xl truncate" title={f.description}>{f.description}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        )}

        {/* ── MIDDLE ROW: Context & Code ── */}
        {results.root_cause && (
          <div className="grid grid-cols-1 xl:grid-cols-2 gap-6">
            {/* Left Column (Context) */}
            <div className="space-y-6 flex flex-col">
              
              {/* Vuln + File combined */}
              <div className="rounded-2xl border border-red-900/50 bg-red-950/20 p-6 shadow-sm">
                <h3 className="text-sm font-bold uppercase tracking-wider text-red-500 mb-3 flex items-center gap-2">
                  🚨 {results.root_cause}
                </h3>
                <div className="mt-4 pt-4 border-t border-red-900/30">
                  <div className="text-xs text-zinc-500 uppercase tracking-wider mb-2">Affected File:</div>
                  <div className="text-sm text-zinc-300 font-mono bg-zinc-950 p-2.5 rounded-lg border border-zinc-800 break-all shadow-inner">
                    {results.affected_file || (results.suspected_files && results.suspected_files[0])}
                  </div>
                </div>
              </div>

              {/* Exploit */}
              {results.exploit_example && (
                <div className="rounded-2xl border border-orange-900/50 bg-orange-950/20 p-6 shadow-sm flex-1">
                  <h3 className="text-xs font-bold uppercase tracking-wider text-orange-500 mb-3 flex items-center gap-2">
                    🎯 Exploit Payload
                  </h3>
                  <pre className="text-base text-orange-300 bg-orange-950/40 border border-orange-900/50 p-4 rounded-xl overflow-x-auto whitespace-pre-wrap font-mono shadow-inner">
                    {results.exploit_example}
                  </pre>
                  {results.risk_if_unfixed && (
                    <div className="mt-4 pt-4 border-t border-orange-900/30">
                      <div className="text-xs text-zinc-500 uppercase tracking-wider mb-2">Risk if Unfixed:</div>
                      <div className="text-sm text-zinc-300 leading-relaxed">{results.risk_if_unfixed}</div>
                    </div>
                  )}
                </div>
              )}
            </div>

            {/* Right Column (Code) */}
            <div className="space-y-6 flex flex-col">
              
              {/* Vulnerable Code */}
              {results.vulnerable_code && (
                <div className="rounded-2xl border border-zinc-800 bg-zinc-900/50 p-6 shadow-sm flex-1 flex flex-col">
                  <h3 className="text-xs font-bold uppercase tracking-wider text-zinc-500 mb-3 flex items-center gap-2">
                    <span className="text-red-500">❌</span> Vulnerable Code
                  </h3>
                  <div className="relative group flex-1">
                    <pre className="text-sm text-red-300 bg-red-950/20 border border-red-900/30 p-4 rounded-xl overflow-x-auto whitespace-pre-wrap font-mono max-h-[250px] overflow-y-auto scrollbar-thin scrollbar-thumb-zinc-700 h-full">
                      {results.vulnerable_code}
                    </pre>
                  </div>
                </div>
              )}

              {/* Fixed Code */}
              {results.patched_code && (
                <div className="rounded-2xl border border-green-900/30 bg-green-950/10 p-6 shadow-sm flex-1 flex flex-col">
                  <h3 className="text-xs font-bold uppercase tracking-wider text-green-500 mb-3 flex items-center gap-2">
                    ✅ Fixed Code
                  </h3>
                  {results.fix_summary && <p className="text-sm text-zinc-400 mb-3">{results.fix_summary}</p>}
                  <div className="relative group flex-1">
                    <pre className="text-sm text-green-300 bg-green-950/20 border border-green-900/30 p-4 rounded-xl overflow-x-auto whitespace-pre-wrap font-mono max-h-[250px] overflow-y-auto scrollbar-thin scrollbar-thumb-zinc-700 h-full">
                      {results.patched_code}
                    </pre>
                  </div>
                </div>
              )}
            </div>
          </div>
        )}

        {/* ── REGRESSION TESTS (Full Width) ── */}
        {results.regression_tests && (
          <div className="rounded-2xl border border-zinc-800 bg-zinc-950 overflow-hidden shadow-xl">
            <div className="bg-zinc-900/80 px-6 py-4 border-b border-zinc-800 flex items-center justify-between">
              <h3 className="text-xs font-bold uppercase tracking-wider text-indigo-400 flex items-center gap-2">
                <span>🧪</span> Regression Tests Generated
              </h3>
            </div>
            <div className="p-6">
              <pre className="text-sm text-indigo-300 bg-indigo-950/10 border border-indigo-900/30 p-4 rounded-xl overflow-x-auto whitespace-pre-wrap font-mono">
                {results.regression_tests}
              </pre>
            </div>
          </div>
        )}

        {/* ── TEST EXECUTION OUTPUT (Full Width) ── */}
        {results.test_results && (
          <div className="rounded-2xl border border-zinc-800 bg-zinc-950 overflow-hidden shadow-xl">
            <div className="bg-zinc-900/80 px-6 py-4 border-b border-zinc-800 flex items-center justify-between">
              <h3 className="text-xs font-bold uppercase tracking-wider text-indigo-400 flex items-center gap-2">
                <span>⚡</span> Test Execution Output
              </h3>
              <div className="flex items-center gap-4">
                <span className="text-xs text-zinc-500">
                  Duration: {results.test_results.duration}s
                </span>
                <span className={`px-2 py-1 rounded text-xs font-bold uppercase ${results.tests_passed ? 'bg-green-950/50 text-green-400' : 'bg-red-950/50 text-red-400'}`}>
                  {results.tests_passed ? 'PASSED' : 'FAILED'}
                </span>
              </div>
            </div>
            <div className="p-6">
              <div className="flex gap-4 mb-4">
                <div className="bg-green-950/20 border border-green-900/30 rounded px-3 py-1.5 flex items-center gap-2">
                  <span className="text-green-500 font-bold">{results.test_results.passed}</span>
                  <span className="text-xs text-zinc-400 uppercase">Passed</span>
                </div>
                <div className="bg-red-950/20 border border-red-900/30 rounded px-3 py-1.5 flex items-center gap-2">
                  <span className="text-red-500 font-bold">{results.test_results.failed}</span>
                  <span className="text-xs text-zinc-400 uppercase">Failed</span>
                </div>
              </div>
              <pre className={`text-sm ${results.tests_passed ? 'text-zinc-300' : 'text-red-300'} bg-black p-4 rounded-xl overflow-x-auto whitespace-pre-wrap font-mono max-h-[400px] scrollbar-thin scrollbar-thumb-zinc-700`}>
                {results.test_results.stdout || results.test_results.stderr}
              </pre>
            </div>
          </div>
        )}

        {/* ── PATCH DIFF (Full Width) ── */}
        {results.patch && (
          <div className="rounded-2xl border border-zinc-800 bg-zinc-950 overflow-hidden shadow-xl">
            <div className="bg-zinc-900/80 px-6 py-4 border-b border-zinc-800 flex items-center justify-between">
              <h3 className="text-xs font-bold uppercase tracking-wider text-zinc-400">
                Patch Diff Viewer
              </h3>
            </div>
            <pre className="text-sm font-[family-name:var(--font-mono)] text-zinc-300 bg-[#0d1117] p-0 overflow-x-auto">
              <div className="max-h-[600px] overflow-y-auto scrollbar-thin scrollbar-thumb-zinc-700 py-4">
                {results.patch?.split("\n").map((line, i) => {
                  let lineClass = "px-6 py-0.5 whitespace-pre ";
                  if (line.startsWith("+") && !line.startsWith("+++"))
                    lineClass += "bg-[#2ea04326] text-[#7ee787]";
                  else if (line.startsWith("-") && !line.startsWith("---"))
                    lineClass += "bg-[#f8514926] text-[#ffa198]";
                  else if (line.startsWith("@@")) lineClass += "bg-[#388bfd1a] text-[#79c0ff]";
                  else if (line.startsWith("diff") || line.startsWith("---") || line.startsWith("+++"))
                    lineClass += "text-zinc-500 font-bold";
                  else lineClass += "text-[#c9d1d9]";
                  return (
                    <div key={i} className={lineClass}>
                      {line}
                    </div>
                  );
                })}
              </div>
            </pre>
          </div>
        )}

        {/* ── BOTTOM ROW: Sprint & PR ── */}
        {done && (
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            <div className="rounded-2xl border border-zinc-800 bg-zinc-900/50 p-6 shadow-sm">
              <h3 className="text-xs font-bold uppercase tracking-wider text-zinc-500 mb-3">
                Sprint Plan Recommendation
              </h3>
              <p className="text-sm text-zinc-300 leading-relaxed">
                {results.sprint_recommendation}
              </p>
            </div>
            <div className="rounded-2xl border border-zinc-800 bg-zinc-900/50 p-6 shadow-sm">
              <h3 className="text-xs font-bold uppercase tracking-wider text-zinc-500 mb-3">
                Generated Pull Request
              </h3>
              <div className="flex items-center gap-3">
                <span className={`inline-flex items-center rounded-md px-2.5 py-1 text-xs font-bold uppercase tracking-wider ${
                    effectivePrMode === "live"
                      ? "bg-green-500/20 text-green-400 border border-green-500/30"
                      : "bg-yellow-500/20 text-yellow-400 border border-yellow-500/30"
                  }`}>
                  {effectivePrMode === "live" ? "LIVE PR" : "SAFE DEMO MODE"}
                </span>
                {effectivePrMode === "live" ? (
                  <a href={rawPrUrl} target="_blank" rel="noopener noreferrer" className="text-sm text-indigo-400 hover:text-indigo-300 break-all font-mono hover:underline">
                    {rawPrUrl}
                  </a>
                ) : (
                  <span className="text-sm text-zinc-400 font-medium">
                    Mock Pull Request Generated (Ready for GitHub)
                  </span>
                )}
              </div>
            </div>
          </div>
        )}

        {/* ── Footer Value Statement ── */}
        {done && (
          <div className="pt-4 pb-8 text-center text-xs text-zinc-600 font-mono tracking-widest uppercase">
            Detect → Explain → Fix → Plan → PR
          </div>
        )}

      </div>
    </main>
  );
}

// --- Reusable Result Card ---

function ResultCard({
  title,
  ready,
  children,
}: {
  title: string;
  ready: boolean;
  children: React.ReactNode;
}) {
  return (
    <div className="rounded-xl border border-zinc-800 bg-zinc-900/50 p-4">
      <h3 className="text-xs font-semibold uppercase tracking-wider text-zinc-500 mb-3">
        {title}
      </h3>
      {ready ? (
        children
      ) : (
        <div className="space-y-2">
          <div className="h-4 w-3/4 rounded bg-zinc-800 animate-pulse" />
          <div className="h-4 w-1/2 rounded bg-zinc-800 animate-pulse" />
        </div>
      )}
    </div>
  );
}

// --- Page Export (with Suspense for useSearchParams) ---

export default function DashboardPage() {
  return (
    <Suspense
      fallback={
        <div className="flex-1 flex items-center justify-center bg-zinc-950 min-h-screen">
          <p className="text-zinc-500">Loading dashboard…</p>
        </div>
      }
    >
      <DashboardContent />
    </Suspense>
  );
}
