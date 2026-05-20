import { useEffect, useRef } from "react";
import { useChatbot } from "./ChatbotContext";
import ChatbotMessageBubble from "./ChatbotMessageBubble";
import ChatbotTypingIndicator from "./ChatbotTypingIndicator";

export default function ChatbotMessages() {
  const { messages, typing, streamingText } = useChatbot();
  const endRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    endRef.current?.scrollIntoView({ behavior: "smooth", block: "end" });
  }, [messages, typing, streamingText]);

  return (
    <div
      className="flex-1 overflow-y-auto bg-white px-3 py-4 space-y-3"
      role="log"
      aria-live="polite"
      aria-label="Tin nhắn trò chuyện"
    >
      {messages.map((m) => (
        <ChatbotMessageBubble key={m.id} message={m} />
      ))}
      {typing && streamingText && (
        <div className="flex justify-start animate-fade-up">
          <div className="flex items-end gap-2 max-w-[88%]">
            <div className="h-7 w-7 flex-shrink-0 rounded-full bg-black grid place-items-center text-white text-[10px] font-bold">
              AI
            </div>
            <div className="rounded-2xl rounded-bl-sm bg-[#F8F9FA] px-3.5 py-2 text-[13.5px] leading-relaxed text-[var(--color-ink)]">
              {streamingText}
              <span className="ml-0.5 inline-block w-1.5 h-3.5 align-middle bg-[var(--color-ink)] animate-pulse" aria-hidden />
            </div>
          </div>
        </div>
      )}
      {typing && !streamingText && <ChatbotTypingIndicator />}
      <div ref={endRef} />
    </div>
  );
}
