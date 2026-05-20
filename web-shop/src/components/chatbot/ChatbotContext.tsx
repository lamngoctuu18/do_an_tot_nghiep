import {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useMemo,
  useRef,
  useState,
  type ReactNode,
} from "react";
import { useAuth } from "../../context/AuthContext";
import { chatbotApi, tokenStore } from "../../lib/api";
import type { ChatMessage, QuickActionKey } from "./types";

type ChatbotContextValue = {
  open: boolean;
  minimized: boolean;
  messages: ChatMessage[];
  typing: boolean;
  streamingText: string;
  unread: number;
  sessionId: string;
  toggle: () => void;
  setOpen: (v: boolean) => void;
  minimize: () => void;
  restore: () => void;
  sendUser: (text: string) => void;
  triggerQuickAction: (k: QuickActionKey) => void;
  clearHistory: () => Promise<void>;
};

const ChatbotContext = createContext<ChatbotContextValue | null>(null);
const SESSION_KEY = "vton.chatbot.sessionId";
const HISTORY_KEY = "vton.chatbot.history";
const MAX_LOCAL_MSGS = 100;

const uid = () => Math.random().toString(36).slice(2, 10) + Date.now().toString(36);

const WELCOME: ChatMessage = {
  id: "welcome",
  role: "bot",
  createdAt: Date.now(),
  text:
    "Xin chào! Tôi là Trợ lý thời trang AI. Tôi có thể tư vấn size, gợi ý sản phẩm, hướng dẫn thử đồ ảo và kiểm tra đơn hàng giúp bạn.",
  quickReplies: [
    { label: "Tư vấn size", value: "qa:size" },
    { label: "Gợi ý sản phẩm", value: "qa:suggest" },
    { label: "Hướng dẫn thử đồ", value: "qa:tryon" },
    { label: "Kiểm tra đơn hàng", value: "qa:order" },
    { label: "Chính sách đổi trả", value: "qa:policy" },
  ],
};

function loadOrCreateSession(): string {
  let s = localStorage.getItem(SESSION_KEY);
  if (!s) {
    s = `web-${uid()}`;
    localStorage.setItem(SESSION_KEY, s);
  }
  return s;
}

function loadCachedMessages(): ChatMessage[] {
  try {
    const raw = localStorage.getItem(HISTORY_KEY);
    if (!raw) return [WELCOME];
    const arr = JSON.parse(raw) as ChatMessage[];
    return Array.isArray(arr) && arr.length ? arr : [WELCOME];
  } catch {
    return [WELCOME];
  }
}

function saveMessages(msgs: ChatMessage[]) {
  try {
    const trimmed = msgs.slice(-MAX_LOCAL_MSGS);
    localStorage.setItem(HISTORY_KEY, JSON.stringify(trimmed));
  } catch {
    /* ignore quota */
  }
}

function attachmentsToMessage(payload: any, baseId: string): ChatMessage {
  return {
    id: baseId,
    role: "bot",
    createdAt: Date.now(),
    text: payload.text || undefined,
    products: payload.products || undefined,
    order: payload.order || undefined,
    tryOnSteps: payload.tryOnSteps || undefined,
    quickReplies: payload.quickReplies || undefined,
    requireLogin: payload.requireLogin || undefined,
  };
}

export function ChatbotProvider({ children }: { children: ReactNode }) {
  const { user } = useAuth();
  const [open, setOpenState] = useState(false);
  const [minimized, setMinimized] = useState(false);
  const [messages, setMessages] = useState<ChatMessage[]>(loadCachedMessages);
  const [typing, setTyping] = useState(false);
  const [streamingText, setStreamingText] = useState("");
  const [unread, setUnread] = useState(0);
  const [sessionId] = useState<string>(loadOrCreateSession);
  const esRef = useRef<EventSource | null>(null);

  useEffect(() => {
    saveMessages(messages);
  }, [messages]);

  const setOpen = useCallback((v: boolean) => {
    setOpenState(v);
    if (v) {
      setUnread(0);
      setMinimized(false);
    }
  }, []);

  const toggle = useCallback(() => setOpen(!open), [open, setOpen]);

  const minimize = useCallback(() => {
    setMinimized(true);
  }, []);

  const restore = useCallback(() => {
    setMinimized(false);
    setUnread(0);
  }, []);

  const appendBot = useCallback((msg: ChatMessage) => {
    setMessages((prev) => [...prev, msg]);
    setOpenState((isOpen) => {
      if (!isOpen || minimized) setUnread((u) => u + 1);
      return isOpen;
    });
  }, [minimized]);

  const runStream = useCallback(
    (text: string) => {
      if (esRef.current) {
        esRef.current.close();
        esRef.current = null;
      }
      setTyping(true);
      setStreamingText("");
      const accumulated = { value: "" };
      const access = tokenStore.access;
      const baseUrl = chatbotApi.streamUrl(text, sessionId);
      const url = access ? `${baseUrl}&token=${encodeURIComponent(access)}` : baseUrl;

      let es: EventSource;
      try {
        es = new EventSource(url, { withCredentials: false });
      } catch {
        setTyping(false);
        appendBot({
          id: uid(),
          role: "bot",
          createdAt: Date.now(),
          text: "Không thể kết nối tới máy chủ.",
        });
        return;
      }
      esRef.current = es;
      const botId = uid();

      es.onmessage = (evt) => {
        try {
          const data = JSON.parse(evt.data);
          if (data.type === "delta") {
            accumulated.value += data.content || "";
            setStreamingText(accumulated.value);
          } else if (data.type === "attachments") {
            const msg = attachmentsToMessage({ ...data.payload, text: accumulated.value }, botId);
            appendBot(msg);
            setStreamingText("");
          } else if (data.type === "done") {
            setTyping(false);
            es.close();
            esRef.current = null;
          } else if (data.type === "error") {
            setTyping(false);
            setStreamingText("");
            appendBot({
              id: uid(),
              role: "bot",
              createdAt: Date.now(),
              text: "Đã có lỗi xảy ra. Vui lòng thử lại.",
            });
            es.close();
            esRef.current = null;
          }
        } catch {
          /* ignore malformed */
        }
      };
      es.onerror = () => {
        // Fallback to non-stream POST if SSE fails
        es.close();
        esRef.current = null;
        chatbotApi
          .send(text, sessionId)
          .then((r) => {
            setTyping(false);
            setStreamingText("");
            appendBot(attachmentsToMessage(r.reply || {}, uid()));
          })
          .catch(() => {
            setTyping(false);
            setStreamingText("");
            appendBot({
              id: uid(),
              role: "bot",
              createdAt: Date.now(),
              text: "Mất kết nối tới trợ lý. Vui lòng thử lại sau.",
            });
          });
      };
    },
    [appendBot, sessionId],
  );

  const sendUser = useCallback(
    (text: string) => {
      const clean = text.trim();
      if (!clean) return;
      const userMsg: ChatMessage = {
        id: uid(),
        role: "user",
        createdAt: Date.now(),
        text: clean,
      };
      setMessages((prev) => [...prev, userMsg]);
      runStream(clean);
    },
    [runStream],
  );

  const triggerQuickAction = useCallback(
    (k: QuickActionKey) => {
      const label =
        k === "size"
          ? "Tư vấn size"
          : k === "suggest"
          ? "Gợi ý sản phẩm"
          : k === "tryon"
          ? "Hướng dẫn thử đồ"
          : k === "order"
          ? "Kiểm tra đơn hàng"
          : "Chính sách đổi trả";
      const userMsg: ChatMessage = {
        id: uid(),
        role: "user",
        createdAt: Date.now(),
        text: label,
      };
      setMessages((prev) => [...prev, userMsg]);
      runStream(`qa:${k}`);
    },
    [runStream],
  );

  const clearHistory = useCallback(async () => {
    try {
      await chatbotApi.clear(sessionId);
    } catch {
      /* ignore */
    }
    setMessages([WELCOME]);
    localStorage.removeItem(HISTORY_KEY);
  }, [sessionId]);

  // hide unused warning + reactive to user role changes
  void user;

  const value = useMemo<ChatbotContextValue>(
    () => ({
      open,
      minimized,
      messages,
      typing,
      streamingText,
      unread,
      sessionId,
      toggle,
      setOpen,
      minimize,
      restore,
      sendUser,
      triggerQuickAction,
      clearHistory,
    }),
    [open, minimized, messages, typing, streamingText, unread, sessionId, toggle, setOpen, minimize, restore, sendUser, triggerQuickAction, clearHistory],
  );

  return <ChatbotContext.Provider value={value}>{children}</ChatbotContext.Provider>;
}

export function useChatbot() {
  const ctx = useContext(ChatbotContext);
  if (!ctx) throw new Error("useChatbot must be used inside ChatbotProvider");
  return ctx;
}
