import { useChatbot } from "./ChatbotContext";
import type { QuickActionKey } from "./types";

const ACTIONS: { key: QuickActionKey; label: string; icon: string }[] = [
  { key: "size", label: "Tư vấn size", icon: "📏" },
  { key: "suggest", label: "Gợi ý sản phẩm", icon: "✨" },
  { key: "tryon", label: "Hướng dẫn thử đồ", icon: "👗" },
  { key: "order", label: "Kiểm tra đơn hàng", icon: "📦" },
  { key: "policy", label: "Chính sách đổi trả", icon: "↩️" },
];

export default function ChatbotQuickActions() {
  const { triggerQuickAction } = useChatbot();
  return (
    <div className="flex flex-wrap gap-1.5 border-t border-[var(--color-line)] bg-white px-3 py-2">
      {ACTIONS.map((a) => (
        <button
          key={a.key}
          onClick={() => triggerQuickAction(a.key)}
          className="flex items-center gap-1 rounded-full border border-[var(--color-line)] bg-white px-2.5 py-1 text-[11px] font-medium text-[var(--color-ink)] hover:border-[var(--color-accent)] hover:bg-[var(--color-bg-warm)] transition-colors"
        >
          <span>{a.icon}</span>
          <span>{a.label}</span>
        </button>
      ))}
    </div>
  );
}
