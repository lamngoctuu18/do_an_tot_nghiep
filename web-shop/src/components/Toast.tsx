import { createContext, useCallback, useContext, useState } from "react";
import type { ReactNode } from "react";
import { Check, X, AlertCircle, Info } from "lucide-react";

type ToastKind = "success" | "error" | "info";
interface Toast {
  id: number;
  kind: ToastKind;
  message: string;
}
interface ToastCtx {
  show: (msg: string, kind?: ToastKind) => void;
}

const Ctx = createContext<ToastCtx | null>(null);

export function ToastProvider({ children }: { children: ReactNode }) {
  const [toasts, setToasts] = useState<Toast[]>([]);

  const show = useCallback((message: string, kind: ToastKind = "success") => {
    const id = Date.now() + Math.random();
    setToasts((t) => [...t, { id, kind, message }]);
    setTimeout(() => setToasts((t) => t.filter((x) => x.id !== id)), 3000);
  }, []);

  return (
    <Ctx.Provider value={{ show }}>
      {children}
      <div className="fixed top-20 right-4 z-[100] flex flex-col gap-2 pointer-events-none">
        {toasts.map((t) => (
          <div
            key={t.id}
            className="animate-slide-in-right bg-white border border-[var(--color-line)] rounded-xl shadow-lg px-4 py-3 flex items-center gap-3 min-w-[280px] pointer-events-auto"
          >
            <span
              className={`w-8 h-8 rounded-full flex items-center justify-center ${
                t.kind === "success"
                  ? "bg-[var(--color-success)]/15 text-[var(--color-success)]"
                  : t.kind === "error"
                    ? "bg-[var(--color-danger)]/15 text-[var(--color-danger)]"
                    : "bg-[var(--color-info)]/15 text-[var(--color-info)]"
              }`}
            >
              {t.kind === "success" && <Check className="w-4 h-4" />}
              {t.kind === "error" && <AlertCircle className="w-4 h-4" />}
              {t.kind === "info" && <Info className="w-4 h-4" />}
            </span>
            <p className="flex-1 text-sm text-[var(--color-ink)]">{t.message}</p>
            <button
              onClick={() =>
                setToasts((arr) => arr.filter((x) => x.id !== t.id))
              }
              className="text-[var(--color-ink-muted)] hover:text-[var(--color-ink)]"
            >
              <X className="w-4 h-4" />
            </button>
          </div>
        ))}
      </div>
    </Ctx.Provider>
  );
}

export function useToast() {
  const ctx = useContext(Ctx);
  if (!ctx) throw new Error("useToast must be used inside ToastProvider");
  return ctx;
}
