import { Link } from "react-router-dom";
import type { ChatTryOnStep } from "./types";

export default function ChatbotTryOnGuideCard({ steps }: { steps: ChatTryOnStep[] }) {
  return (
    <div className="rounded-xl border border-[var(--color-line)] bg-[var(--color-bg-warm)] p-3">
      <ul className="space-y-2">
        {steps.map((s) => (
          <li key={s.title} className="flex gap-2">
            <div className="h-6 w-6 flex-shrink-0 rounded-full bg-[#D6B98C] text-black text-[11px] font-bold grid place-items-center">
              {s.title.charAt(0)}
            </div>
            <div>
              <div className="text-[12px] font-semibold text-[var(--color-ink)]">{s.title}</div>
              <div className="text-[11px] text-[var(--color-ink-muted)]">{s.description}</div>
            </div>
          </li>
        ))}
      </ul>
      <Link
        to="/try-on"
        className="mt-2 block w-full rounded-md bg-black px-3 py-1.5 text-center text-[12px] font-medium text-white hover:bg-[var(--color-primary-hover)]"
      >
        Thử ngay
      </Link>
    </div>
  );
}
