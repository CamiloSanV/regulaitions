"use client";

import { useState, useRef, useEffect } from "react";

// ── Types ──────────────────────────────────────────────────────────────────────
interface ToolCall {
  tool: string;
  input: string;
  output: string;
}

interface Message {
  role: "user" | "assistant";
  content: string;
  toolCalls?: ToolCall[];
  isStreaming?: boolean;
}

// ── Config ─────────────────────────────────────────────────────────────────────
const API_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";
//import { useRef } from "react"; // already imported, just making sure

// ── Tool metadata ──────────────────────────────────────────────────────────────
const TOOL_ICONS: Record<string, string> = {
  search_regulation:  "🔍",
  get_article:        "📄",
  check_obligations:  "✅",
  compare_versions:   "🔀",
  generate_checklist: "📋",
};

const TOOL_LABELS: Record<string, string> = {
  search_regulation:  "Searching corpus",
  get_article:        "Retrieving article",
  check_obligations:  "Classifying system",
  compare_versions:   "Comparing versions",
  generate_checklist: "Generating checklist",
};

// ── Highlight article citations ────────────────────────────────────────────────
function highlightCitations(text: string): React.ReactNode[] {
  const parts = text.split(/((?:Article|Annex|Recital)\s+[\dIVX]+(?:\s*[§,]\s*[\w\d]+)*)/gi);
  return parts.map((part, i) => {
    if (/^(Article|Annex|Recital)\s+/i.test(part)) {
      return (
        <span key={i} style={{
          display: "inline-block",
          fontSize: 11,
          fontWeight: 600,
          padding: "1px 6px",
          borderRadius: 4,
          background: "#1A3A8F",
          color: "#4F7FFF",
          margin: "0 2px",
          fontFamily: "monospace",
        }}>{part}</span>
      );
    }
    return <span key={i}>{part}</span>;
  });
}

// ── Suggested questions ────────────────────────────────────────────────────────
const SUGGESTIONS = [
  "What obligations apply to a credit scoring AI system?",
  "What does Article 9 say about risk management?",
  "Which AI systems are prohibited under the EU AI Act?",
  "What are the transparency requirements in Article 13?",
];

// ── Main component ─────────────────────────────────────────────────────────────
export default function ChatPage() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const [activeTools, setActiveTools] = useState<string[]>([]);
  const bottomRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLInputElement>(null);
  const sessionIdRef = useRef("demo-" + Math.random().toString(36).slice(2, 8));
  const SESSION_ID = sessionIdRef.current;  // ← add these two lines

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, activeTools]);

  async function sendMessage(text?: string) {
    const userText = (text || input).trim();
    if (!userText || loading) return;

    setInput("");
    setLoading(true);
    setActiveTools([]);

    // Add user message
    setMessages(prev => [...prev, { role: "user", content: userText }]);

    // Add placeholder assistant message
    setMessages(prev => [...prev, {
      role: "assistant",
      content: "",
      toolCalls: [],
      isStreaming: true,
    }]);

    try {
      const res = await fetch(`${API_URL}/chat/stream`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ message: userText, session_id: SESSION_ID }),
      });

      if (!res.ok) throw new Error(`API error: ${res.status}`);

      const reader = res.body!.getReader();
      const decoder = new TextDecoder();
      let buffer = "";
      let fullAnswer = "";
      const collectedTools: ToolCall[] = [];
      let thinkingSet = false;

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split("\n");
        buffer = lines.pop() || "";

        for (const line of lines) {
          if (!line.startsWith("data: ")) continue;
          const raw = line.slice(6).trim();
          if (!raw) continue;

          let event: any;
          try { event = JSON.parse(raw); } catch { continue; }

          // Only show thinking indicator once, ignore keepalive pings
          if (event.type === "thinking") {
            if (!thinkingSet) {
              setActiveTools(["thinking"]);
              thinkingSet = true;
            }
            continue;
          }

          if (event.type === "tool_call") {
            setActiveTools(prev => [
              ...prev.filter(t => t !== "thinking"),
              event.tool,
            ]);
            collectedTools.push({
              tool: event.tool,
              input: event.input || "",
              output: "",
            });
          }

          if (event.type === "token") {
            fullAnswer += event.content;
            setMessages(prev => {
              const updated = [...prev];
              updated[updated.length - 1] = {
                ...updated[updated.length - 1],
                content: fullAnswer,
                isStreaming: true,
              };
              return updated;
            });
          }

          if (event.type === "done") {
            setActiveTools([]);
            setMessages(prev => {
              const updated = [...prev];
              updated[updated.length - 1] = {
                role: "assistant",
                content: fullAnswer,
                toolCalls: event.tool_calls || collectedTools,
                isStreaming: false,
              };
              return updated;
            });
          }

          if (event.type === "error") {
            throw new Error(event.content);
          }
        }
      }

      // Fallback: stream ended without a "done" event
      if (fullAnswer) {
        setActiveTools([]);
        setMessages(prev => {
          const last = prev[prev.length - 1];
          if (last.isStreaming) {
            const updated = [...prev];
            updated[updated.length - 1] = {
              role: "assistant",
              content: fullAnswer,
              toolCalls: collectedTools,
              isStreaming: false,
            };
            return updated;
          }
          return prev;
        });
      }

    } catch (err: any) {
      setActiveTools([]);
      setMessages(prev => {
        const updated = [...prev];
        updated[updated.length - 1] = {
          role: "assistant",
          content: `⚠️ ${err.message || "Could not connect to API. Make sure it is running at " + API_URL}`,
          isStreaming: false,
        };
        return updated;
      });
    } finally {
      setLoading(false);
      inputRef.current?.focus();
    }
  }

  function handleKey(e: React.KeyboardEvent) {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      sendMessage();
    }
  }

  const isEmpty = messages.length === 0;

  // ── Styles ───────────────────────────────────────────────────────────────────
  const vars = {
    bg:      "#0F1117",
    bg2:     "#171B26",
    bg3:     "#1E2333",
    border:  "#2A3045",
    text:    "#E8EAF0",
    text2:   "#8B93A8",
    text3:   "#555E74",
    accent:  "#4F7FFF",
    accent2: "#1A3A8F",
    green:   "#22C55E",
  };

  return (
    <div style={{ display: "flex", flexDirection: "column", height: "100vh", background: vars.bg }}>

      {/* ── Header ── */}
      <header style={{
        padding: "12px 24px",
        borderBottom: `1px solid ${vars.border}`,
        display: "flex",
        alignItems: "center",
        justifyContent: "space-between",
        background: vars.bg2,
        flexShrink: 0,
      }}>
        <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
          <div style={{
            width: 30, height: 30, borderRadius: 8,
            background: vars.accent,
            display: "flex", alignItems: "center", justifyContent: "center",
            fontSize: 15, fontWeight: 700, color: "#fff",
          }}>r</div>
          <span style={{ fontWeight: 600, fontSize: 16, letterSpacing: "-0.02em", color: vars.text }}>
            regul<span style={{ color: vars.accent }}>AI</span>tions
          </span>
        </div>

        <div style={{ display: "flex", gap: 6 }}>
          {["EU AI Act", "GDPR"].map((s, i) => (
            <span key={i} style={{
              fontSize: 11, padding: "3px 10px", borderRadius: 20,
              background: vars.bg3, border: `1px solid ${vars.border}`,
              color: i === 0 ? vars.green : vars.text2,
            }}>
              {i === 0 ? "● " : ""}{s}
            </span>
          ))}
        </div>
      </header>

      {/* ── Messages area ── */}
      <div style={{ flex: 1, overflowY: "auto", padding: "24px 24px 0" }}>

        {/* Empty state */}
        {isEmpty && (
          <div style={{ maxWidth: 580, margin: "48px auto 0", textAlign: "center" }}>
            <div style={{ fontSize: 36, marginBottom: 14 }}>⚖️</div>
            <h1 style={{
              fontSize: 24, fontWeight: 700, marginBottom: 8,
              letterSpacing: "-0.03em", color: vars.text,
            }}>
              regul<span style={{ color: vars.accent }}>AI</span>tions
            </h1>
            <p style={{
              color: vars.text2, fontSize: 14, marginBottom: 28,
              lineHeight: 1.7,
            }}>
              EU AI Act & GDPR compliance assistant.<br />
              Describe your AI system or ask about a specific article.
            </p>
            <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
              {SUGGESTIONS.map((s, i) => (
                <button
                  key={i}
                  onClick={() => sendMessage(s)}
                  style={{
                    padding: "11px 16px",
                    background: vars.bg2,
                    border: `1px solid ${vars.border}`,
                    borderRadius: 10,
                    color: vars.text2,
                    fontSize: 13,
                    cursor: "pointer",
                    textAlign: "left",
                    transition: "all 0.15s",
                  }}
                  onMouseEnter={e => {
                    e.currentTarget.style.borderColor = vars.accent;
                    e.currentTarget.style.color = vars.text;
                  }}
                  onMouseLeave={e => {
                    e.currentTarget.style.borderColor = vars.border;
                    e.currentTarget.style.color = vars.text2;
                  }}
                >
                  {s}
                </button>
              ))}
            </div>
          </div>
        )}

        {/* Message list */}
        <div style={{ maxWidth: 760, margin: "0 auto", display: "flex", flexDirection: "column", gap: 18 }}>
          {messages.map((msg, i) => (
            <div
              key={i}
              style={{
                display: "flex",
                flexDirection: msg.role === "user" ? "row-reverse" : "row",
                gap: 10,
                alignItems: "flex-start",
              }}
            >
              {/* Avatar */}
              <div style={{
                width: 28, height: 28, borderRadius: 6, flexShrink: 0,
                background: msg.role === "user" ? vars.accent2 : vars.bg3,
                border: `1px solid ${vars.border}`,
                display: "flex", alignItems: "center", justifyContent: "center",
                fontSize: 11, fontWeight: 700,
                color: msg.role === "user" ? vars.accent : vars.text2,
              }}>
                {msg.role === "user" ? "U" : "r"}
              </div>

              <div style={{ flex: 1, maxWidth: "86%" }}>
                {/* Tool call pills */}
                {msg.toolCalls && msg.toolCalls.length > 0 && (
                  <div style={{ display: "flex", flexWrap: "wrap", gap: 6, marginBottom: 8 }}>
                    {msg.toolCalls.map((tc, j) => (
                      <span key={j} style={{
                        display: "inline-flex", alignItems: "center", gap: 5,
                        fontSize: 11, fontWeight: 500,
                        padding: "3px 9px", borderRadius: 20,
                        background: vars.bg3, border: `1px solid ${vars.border}`,
                        color: vars.text2,
                      }}>
                        {TOOL_ICONS[tc.tool] || "🔧"} {TOOL_LABELS[tc.tool] || tc.tool}
                      </span>
                    ))}
                  </div>
                )}

                {/* Message bubble */}
                <div style={{
                  padding: "12px 16px",
                  borderRadius: msg.role === "user"
                    ? "12px 4px 12px 12px"
                    : "4px 12px 12px 12px",
                  background: msg.role === "user" ? vars.accent2 : vars.bg2,
                  border: `1px solid ${msg.role === "user" ? vars.accent : vars.border}`,
                  fontSize: 14,
                  lineHeight: 1.7,
                  color: msg.role === "user" ? "#C8D8FF" : vars.text,
                  whiteSpace: "pre-wrap",
                }}>
                  {/* Typing indicator */}
                  {msg.isStreaming && !msg.content ? (
                    <span style={{ display: "flex", gap: 4, alignItems: "center" }}>
                      {[0, 0.2, 0.4].map((delay, k) => (
                        <span key={k} style={{
                          display: "inline-block",
                          width: 6, height: 6,
                          borderRadius: "50%",
                          background: vars.text3,
                          animation: `blink 1.2s infinite ${delay}s`,
                        }} />
                      ))}
                    </span>
                  ) : (
                    highlightCitations(msg.content)
                  )}
                </div>
              </div>
            </div>
          ))}

          {/* Active tool indicator */}
          {activeTools.length > 0 && (
            <div style={{ display: "flex", gap: 8, paddingLeft: 38, paddingBottom: 4 }}>
              {activeTools.map((tool, i) => (
                <span key={i} style={{
                  display: "inline-flex", alignItems: "center", gap: 6,
                  fontSize: 11, fontWeight: 500,
                  padding: "4px 10px", borderRadius: 20,
                  background: vars.bg3,
                  border: `1px solid ${vars.accent}`,
                  color: vars.accent,
                }}>
                  {tool === "thinking" ? "💭" : TOOL_ICONS[tool] || "🔧"}
                  {tool === "thinking"
                    ? "Thinking..."
                    : `Running ${TOOL_LABELS[tool] || tool}...`}
                </span>
              ))}
            </div>
          )}

          <div ref={bottomRef} style={{ paddingBottom: 24 }} />
        </div>
      </div>

      {/* ── Input area ── */}
      <div style={{
        padding: "14px 24px 18px",
        borderTop: `1px solid ${vars.border}`,
        background: vars.bg2,
        flexShrink: 0,
      }}>
        <div style={{ maxWidth: 760, margin: "0 auto", display: "flex", gap: 10 }}>
          <input
            ref={inputRef}
            value={input}
            onChange={e => setInput(e.target.value)}
            onKeyDown={handleKey}
            disabled={loading}
            placeholder="Ask about your AI system's obligations, a specific article..."
            style={{
              flex: 1,
              padding: "12px 16px",
              background: vars.bg3,
              border: `1px solid ${vars.border}`,
              borderRadius: 10,
              color: vars.text,
              fontSize: 14,
              outline: "none",
              fontFamily: "inherit",
              transition: "border-color 0.15s",
            }}
            onFocus={e => (e.target.style.borderColor = vars.accent)}
            onBlur={e => (e.target.style.borderColor = vars.border)}
          />
          <button
            onClick={() => sendMessage()}
            disabled={loading || !input.trim()}
            style={{
              padding: "12px 20px",
              background: loading || !input.trim() ? vars.bg3 : vars.accent,
              border: `1px solid ${loading || !input.trim() ? vars.border : vars.accent}`,
              borderRadius: 10,
              color: loading || !input.trim() ? vars.text3 : "#fff",
              fontSize: 14,
              fontWeight: 600,
              cursor: loading || !input.trim() ? "not-allowed" : "pointer",
              transition: "all 0.15s",
              whiteSpace: "nowrap",
              fontFamily: "inherit",
            }}
          >
            {loading ? "..." : "Send →"}
          </button>
        </div>
        <p style={{
          textAlign: "center", fontSize: 11,
          color: vars.text3, marginTop: 10,
        }}>
          For guidance only · Not legal advice · EU AI Act 2024-Q3
        </p>
      </div>

      {/* Blink animation */}
      <style>{`
        @keyframes blink {
          0%, 80%, 100% { opacity: 0.2; }
          40% { opacity: 1; }
        }
      `}</style>
    </div>
  );
}
