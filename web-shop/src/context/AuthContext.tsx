import { createContext, useContext, useEffect, useState } from 'react';
import type { ReactNode } from 'react';
import { authApi, tokenStore, type ApiUser } from '../lib/api';

interface AuthContextType {
  user: ApiUser | null;
  loading: boolean;
  login: (email: string, password: string) => Promise<void>;
  register: (dto: { email: string; password: string; fullName: string; phone?: string }) => Promise<void>;
  logout: () => Promise<void>;
  refreshUser: () => Promise<void>;
}

const AuthContext = createContext<AuthContextType | null>(null);

export function AuthProvider({ children }: { children: ReactNode }) {
  const [user, setUser] = useState<ApiUser | null>(tokenStore.user);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    if (tokenStore.access) {
      authApi
        .me()
        .then((u) => {
          tokenStore.setUser(u);
          setUser(u);
        })
        .catch(() => {
          tokenStore.clear();
          setUser(null);
        })
        .finally(() => setLoading(false));
    } else {
      setLoading(false);
    }
  }, []);

  const login = async (email: string, password: string) => {
    const data = await authApi.login(email, password);
    tokenStore.set(data.accessToken, data.refreshToken, data.user);
    setUser(data.user);
  };

  const register = async (dto: { email: string; password: string; fullName: string; phone?: string }) => {
    const data = await authApi.register(dto);
    tokenStore.set(data.accessToken, data.refreshToken, data.user);
    setUser(data.user);
  };

  const logout = async () => {
    try {
      await authApi.logout();
    } catch {
      /* ignore */
    }
    tokenStore.clear();
    setUser(null);
  };

  const refreshUser = async () => {
    const u = await authApi.me();
    tokenStore.setUser(u);
    setUser(u);
  };

  return (
    <AuthContext.Provider value={{ user, loading, login, register, logout, refreshUser }}>
      {children}
    </AuthContext.Provider>
  );
}

export function useAuth() {
  const ctx = useContext(AuthContext);
  if (!ctx) throw new Error('useAuth must be used within AuthProvider');
  return ctx;
}
