// Lightweight API client with JWT support + auto refresh.
// Backend prefix: /api (set in NestJS main.ts).

const BASE = '/api';
const ACCESS_KEY = 'vton.accessToken';
const REFRESH_KEY = 'vton.refreshToken';
const USER_KEY = 'vton.user';

export type ApiUser = {
  id: number;
  email: string;
  fullName: string;
  role: 'CUSTOMER' | 'SELLER' | 'ADMIN';
  avatarUrl?: string | null;
};

export class ApiError extends Error {
  status: number;
  errors?: any;
  constructor(status: number, message: string, errors?: any) {
    super(message);
    this.status = status;
    this.errors = errors;
  }
}

export const tokenStore = {
  get access() {
    return localStorage.getItem(ACCESS_KEY);
  },
  get refresh() {
    return localStorage.getItem(REFRESH_KEY);
  },
  get user(): ApiUser | null {
    const raw = localStorage.getItem(USER_KEY);
    return raw ? (JSON.parse(raw) as ApiUser) : null;
  },
  set(access: string, refresh: string, user: ApiUser) {
    localStorage.setItem(ACCESS_KEY, access);
    localStorage.setItem(REFRESH_KEY, refresh);
    localStorage.setItem(USER_KEY, JSON.stringify(user));
  },
  setUser(user: ApiUser) {
    localStorage.setItem(USER_KEY, JSON.stringify(user));
  },
  clear() {
    localStorage.removeItem(ACCESS_KEY);
    localStorage.removeItem(REFRESH_KEY);
    localStorage.removeItem(USER_KEY);
  },
};

let refreshing: Promise<string | null> | null = null;

async function doRefresh(): Promise<string | null> {
  const rt = tokenStore.refresh;
  if (!rt) return null;
  try {
    const res = await fetch(`${BASE}/auth/refresh`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ refreshToken: rt }),
    });
    if (!res.ok) {
      tokenStore.clear();
      return null;
    }
    const data = await res.json();
    const payload = data;
    tokenStore.set(payload.accessToken, payload.refreshToken, payload.user);
    return payload.accessToken as string;
  } catch {
    tokenStore.clear();
    return null;
  }
}

type RequestOpts = {
  method?: string;
  body?: any;
  headers?: Record<string, string>;
  auth?: boolean;
  raw?: boolean;
};

export async function api<T = any>(path: string, opts: RequestOpts = {}): Promise<T> {
  const { method = 'GET', body, headers = {}, auth = true, raw = false } = opts;
  const init: RequestInit = { method, headers: { ...headers } };

  if (body !== undefined) {
    if (body instanceof FormData) {
      init.body = body;
    } else {
      (init.headers as any)['Content-Type'] = 'application/json';
      init.body = JSON.stringify(body);
    }
  }

  if (auth) {
    const at = tokenStore.access;
    if (at) (init.headers as any).Authorization = `Bearer ${at}`;
  }

  let res = await fetch(`${BASE}${path}`, init);

  if (res.status === 401 && auth && tokenStore.refresh) {
    refreshing = refreshing ?? doRefresh();
    const newToken = await refreshing;
    refreshing = null;
    if (newToken) {
      (init.headers as any).Authorization = `Bearer ${newToken}`;
      res = await fetch(`${BASE}${path}`, init);
    }
  }

  if (raw) return res as any;

  let data: any = null;
  const text = await res.text();
  if (text) {
    try {
      data = JSON.parse(text);
    } catch {
      data = text;
    }
  }

  if (!res.ok) {
    const msg = data?.message || data?.error || res.statusText || 'Request failed';
    throw new ApiError(res.status, Array.isArray(msg) ? msg.join(', ') : msg, data?.errors);
  }
  return data as T;
}

// -------- Domain helpers --------
export const authApi = {
  login: (email: string, password: string) =>
    api<{ accessToken: string; refreshToken: string; user: ApiUser }>('/auth/login', {
      method: 'POST',
      body: { email, password },
      auth: false,
    }),
  register: (dto: { email: string; password: string; fullName: string; phone?: string }) =>
    api<{ accessToken: string; refreshToken: string; user: ApiUser }>('/auth/register', {
      method: 'POST',
      body: dto,
      auth: false,
    }),
  logout: () => api('/auth/logout', { method: 'POST', body: { refreshToken: tokenStore.refresh } }),
  me: () => api<ApiUser>('/users/me'),
};

export const productsApi = {
  list: (params: Record<string, any> = {}) => {
    const qs = new URLSearchParams(
      Object.entries(params).filter(([, v]) => v !== undefined && v !== '') as any,
    ).toString();
    return api<{ items: any[]; total: number; page: number; size: number }>(
      `/products${qs ? `?${qs}` : ''}`,
      { auth: false },
    );
  },
  detail: (slug: string) => api<any>(`/products/${slug}`, { auth: false }),
  categories: () => api<any[]>('/categories', { auth: false }),
};

export const cartApi = {
  get: () => api<any>('/cart'),
  add: (variantId: number, quantity: number) =>
    api('/cart/items', { method: 'POST', body: { variantId, quantity } }),
  update: (itemId: number, quantity: number) =>
    api(`/cart/items/${itemId}`, { method: 'PATCH', body: { quantity } }),
  remove: (itemId: number) => api(`/cart/items/${itemId}`, { method: 'DELETE' }),
};

export const ordersApi = {
  list: () => api<any[]>('/orders'),
  detail: (code: string) => api<any>(`/orders/${code}`),
  checkout: (dto: { addressId: number; paymentMethod: string; note?: string }) =>
    api<any>('/orders/checkout', { method: 'POST', body: dto }),
  cancel: (code: string, reason: string) =>
    api<any>(`/orders/${code}/cancel`, { method: 'PATCH', body: { reason } }),
};

export const addressesApi = {
  list: () => api<any[]>('/users/me/addresses'),
  create: (dto: any) => api('/users/me/addresses', { method: 'POST', body: dto }),
  update: (id: number, dto: any) => api(`/users/me/addresses/${id}`, { method: 'PUT', body: dto }),
  remove: (id: number) => api(`/users/me/addresses/${id}`, { method: 'DELETE' }),
  setDefault: (id: number) => api(`/users/me/addresses/${id}/default`, { method: 'PATCH' }),
};

export const notificationsApi = {
  list: (isRead?: boolean) =>
    api<any[]>(`/notifications${isRead !== undefined ? `?is_read=${isRead}` : ''}`),
  read: (id: number) => api(`/notifications/${id}/read`, { method: 'PATCH' }),
  readAll: () => api('/notifications/read-all', { method: 'PATCH' }),
};

export const wishlistApi = {
  list: () => api<any[]>('/wishlist'),
  add: (productId: number) => api(`/wishlist/${productId}`, { method: 'POST' }),
  remove: (productId: number) => api(`/wishlist/${productId}`, { method: 'DELETE' }),
};

export const sellerApi = {
  dashboard: () => api<any>('/seller/dashboard/summary'),
  orders: (params: { status?: string; page?: number; limit?: number } = {}) => {
    const qs = new URLSearchParams(
      Object.entries(params).filter(([, v]) => v !== undefined && v !== '') as any,
    ).toString();
    return api<{ items: any[]; total: number }>(`/seller/orders${qs ? `?${qs}` : ''}`);
  },
  orderDetail: (code: string) => api<any>(`/seller/orders/${code}`),
  confirm: (code: string) => api(`/seller/orders/${code}/confirm`, { method: 'PATCH' }),
  updateStatus: (code: string, orderStatus: string) =>
    api(`/seller/orders/${code}/status`, { method: 'PATCH', body: { orderStatus } }),
  tryonStats: () => api<any[]>('/seller/tryon/statistics'),
  myProducts: () => api<any[]>('/seller/products'),
  myProduct: (id: number) => api<any>(`/seller/products/${id}`),
  createProduct: (body: any) => api<any>('/seller/products', { method: 'POST', body }),
  updateProduct: (id: number, body: any) =>
    api<any>(`/seller/products/${id}`, { method: 'PUT', body }),
  deleteProduct: (id: number) => api(`/seller/products/${id}`, { method: 'DELETE' }),
  submitProduct: (id: number) =>
    api(`/seller/products/${id}/submit`, { method: 'PATCH' }),
  uploadImage: async (file: File): Promise<{ url: string }> => {
    const fd = new FormData();
    fd.append('file', file);
    const access = localStorage.getItem('vton.accessToken');
    const res = await fetch('/api/seller/upload', {
      method: 'POST',
      headers: access ? { Authorization: `Bearer ${access}` } : {},
      body: fd,
    });
    if (!res.ok) throw new ApiError(res.status, 'Upload failed');
    return res.json();
  },
};

export const reviewsApi = {
  list: (productId: number) =>
    api<any[]>(`/products/${productId}/reviews`, { auth: false }),
  create: (productId: number, body: { rating: number; comment?: string; images?: string[] }) =>
    api<any>(`/products/${productId}/reviews`, { method: 'POST', body }),
};

export const adminApi = {
  stats: () => api<any>('/admin/dashboard/summary'),
  users: (params: Record<string, any> = {}) => {
    const qs = new URLSearchParams(
      Object.entries(params).filter(([, v]) => v !== undefined && v !== '') as any,
    ).toString();
    return api<any>(`/admin/users${qs ? `?${qs}` : ''}`);
  },
  lockUser: (id: number) => api(`/admin/users/${id}/lock`, { method: 'PATCH' }),
  unlockUser: (id: number) => api(`/admin/users/${id}/unlock`, { method: 'PATCH' }),
  shops: () => api<any[]>('/admin/shops'),
  approveShop: (id: number) => api(`/admin/shops/${id}/approve`, { method: 'PATCH' }),
  rejectShop: (id: number, reason: string) =>
    api(`/admin/shops/${id}/reject`, { method: 'PATCH', body: { reason } }),
  productsPending: () => api<any[]>('/admin/products?status=PENDING'),
  approveProduct: (id: number) => api(`/admin/products/${id}/approve`, { method: 'PATCH' }),
  rejectProduct: (id: number, reason: string) =>
    api(`/admin/products/${id}/reject`, { method: 'PATCH', body: { reason } }),
  reportRevenue: (from?: string, to?: string) => {
    const qs = new URLSearchParams();
    if (from) qs.set('from_date', from);
    if (to) qs.set('to_date', to);
    return api<any[]>(`/admin/reports/revenue${qs.toString() ? `?${qs}` : ''}`);
  },
  topProducts: (limit = 10) => api<any[]>(`/admin/reports/top-products?limit=${limit}`),
  tryonReport: () => api<any>('/admin/reports/tryon'),
  reviews: (status?: string) =>
    api<any[]>(`/admin/reviews${status ? `?status=${status}` : ''}`),
  hideReview: (id: number) => api(`/admin/reviews/${id}/hide`, { method: 'PATCH' }),
  unhideReview: (id: number) => api(`/admin/reviews/${id}/unhide`, { method: 'PATCH' }),
};

export const usersApi = {
  updateProfile: (dto: any) => api<ApiUser>('/users/me', { method: 'PUT', body: dto }),
};

export const chatbotApi = {
  send: (message: string, sessionId: string) =>
    api<{ sessionId: string; reply: any }>('/chatbot/messages', {
      method: 'POST',
      body: { message, sessionId },
    }),
  history: (sessionId: string) =>
    api<{ messages: any[] }>(`/chatbot/history?sessionId=${encodeURIComponent(sessionId)}`),
  clear: (sessionId: string) =>
    api(`/chatbot/history?sessionId=${encodeURIComponent(sessionId)}`, { method: 'DELETE' }),
  streamUrl: (message: string, sessionId: string) =>
    `${BASE}/chatbot/messages/stream?message=${encodeURIComponent(message)}&sessionId=${encodeURIComponent(sessionId)}`,
};
