import type { ChatMessage } from "./types";
import { useChatbot } from "./ChatbotContext";
import { Link } from "react-router-dom";
import ChatbotProductCard from "./ChatbotProductCard";
import ChatbotOrderCard from "./ChatbotOrderCard";
import ChatbotTryOnGuideCard from "./ChatbotTryOnGuideCard";

export default function ChatbotMessageBubble({ message }: { message: ChatMessage }) {
  const { sendUser, triggerQuickAction } = useChatbot();
  const isBot = message.role === "bot";

  const onQuickReply = (value: string) => {
    if (value.startsWith("qa:")) {
      const key = value.slice(3) as "size" | "suggest" | "tryon" | "order" | "policy";
      triggerQuickAction(key);
    } else {
      sendUser(value);
    }
  };

  return (
    <div className={`flex ${isBot ? "justify-start" : "justify-end"} animate-fade-up`}>
      <div className={`flex items-end gap-2 max-w-[88%] ${isBot ? "" : "flex-row-reverse"}`}>
        {isBot && (
          <div className="h-7 w-7 flex-shrink-0 rounded-full bg-black grid place-items-center text-white text-[10px] font-bold">
            AI
          </div>
        )}
        <div className="flex flex-col gap-2 min-w-0">
          {message.text && (
            <div
              className={
                isBot
                  ? "rounded-2xl rounded-bl-sm bg-[#F8F9FA] px-3.5 py-2 text-[13.5px] leading-relaxed text-[var(--color-ink)]"
                  : "rounded-2xl rounded-br-sm bg-[#111827] px-3.5 py-2 text-[13.5px] leading-relaxed text-white"
              }
            >
              {message.text}
            </div>
          )}
          {message.products && message.products.length > 0 && (
            <div className="flex flex-col gap-2">
              {message.products.map((p) => (
                <ChatbotProductCard key={p.id} product={p} />
              ))}
            </div>
          )}
          {message.order && <ChatbotOrderCard order={message.order} />}
          {message.tryOnSteps && <ChatbotTryOnGuideCard steps={message.tryOnSteps} />}
          {message.requireLogin && (
            <Link
              to="/login"
              className="inline-flex items-center justify-center rounded-full bg-black px-4 py-1.5 text-[12px] font-medium text-white hover:bg-[var(--color-primary-hover)]"
            >
              Đăng nhập ngay
            </Link>
          )}
          {message.quickReplies && message.quickReplies.length > 0 && (
            <div className="flex flex-wrap gap-1.5 pt-0.5">
              {message.quickReplies.map((q) => (
                <button
                  key={q.value}
                  onClick={() => onQuickReply(q.value)}
                  className="rounded-full border border-[var(--color-line)] bg-white px-3 py-1 text-[11.5px] font-medium text-[var(--color-ink)] hover:border-[var(--color-accent)] hover:bg-[var(--color-bg-warm)]"
                >
                  {q.label}
                </button>
              ))}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
