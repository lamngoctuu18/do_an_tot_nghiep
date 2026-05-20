import { useEffect } from "react";
import { useChatbot } from "./ChatbotContext";
import ChatbotHeader from "./ChatbotHeader";
import ChatbotMessages from "./ChatbotMessages";
import ChatbotQuickActions from "./ChatbotQuickActions";
import ChatbotInput from "./ChatbotInput";

export default function ChatbotWindow() {
  const { open, minimized, setOpen } = useChatbot();

  useEffect(() => {
    if (!open) return;
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") setOpen(false);
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [open, setOpen]);

  if (!open || minimized) return null;

  return (
    <>
      <div
        className="fixed inset-0 z-[55] bg-black/20 sm:hidden"
        onClick={() => setOpen(false)}
        aria-hidden
      />
      <div
        role="dialog"
        aria-modal="false"
        aria-label="Trợ lý thời trang AI"
        className="fixed z-[60] flex flex-col bg-white shadow-[0_24px_60px_rgba(0,0,0,0.22)] animate-modal
          inset-x-0 bottom-0 max-h-[88vh] rounded-t-2xl
          sm:inset-auto sm:bottom-6 sm:right-6 sm:h-[600px] sm:max-h-[80vh] sm:w-[400px] sm:rounded-2xl
          border border-[var(--color-line)]"
      >
        <ChatbotHeader />
        <ChatbotMessages />
        <ChatbotQuickActions />
        <ChatbotInput />
      </div>
    </>
  );
}
