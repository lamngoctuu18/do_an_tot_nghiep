export type ChatRole = "bot" | "user";

export type ChatProductSuggestion = {
  id: string | number;
  name: string;
  price: number;
  image: string;
  reason?: string;
};

export type ChatOrderInfo = {
  code: string;
  status: string;
  total: number;
  createdAt: string;
  itemsCount: number;
};

export type ChatTryOnStep = {
  title: string;
  description: string;
};

export type ChatMessage = {
  id: string;
  role: ChatRole;
  createdAt: number;
  // content variants
  text?: string;
  products?: ChatProductSuggestion[];
  order?: ChatOrderInfo;
  tryOnSteps?: ChatTryOnStep[];
  quickReplies?: { label: string; value: string }[];
  requireLogin?: boolean;
};

export type QuickActionKey =
  | "size"
  | "suggest"
  | "tryon"
  | "order"
  | "policy";
