import { createContext, useContext, useEffect, useState, useCallback } from 'react';
import type { ReactNode } from 'react';
import { cartApi } from '../lib/api';
import { useAuth } from './AuthContext';

export type CartItem = {
  id: number;
  quantity: number;
  variant: {
    id: number;
    color?: string;
    size?: string;
    priceDelta: string;
    stock: number;
    product: {
      id: number;
      name: string;
      slug: string;
      price: string;
      images?: { url: string }[];
    };
  };
};

interface CartContextType {
  items: CartItem[];
  loading: boolean;
  addItem: (variantId: number, quantity?: number) => Promise<void>;
  removeItem: (itemId: number) => Promise<void>;
  updateQuantity: (itemId: number, qty: number) => Promise<void>;
  refresh: () => Promise<void>;
  clear: () => void;
  totalItems: number;
  totalPrice: number;
}

const CartContext = createContext<CartContextType | null>(null);

const itemPrice = (it: CartItem) => +it.variant.product.price + +it.variant.priceDelta;

export function CartProvider({ children }: { children: ReactNode }) {
  const { user } = useAuth();
  const [items, setItems] = useState<CartItem[]>([]);
  const [loading, setLoading] = useState(false);

  const refresh = useCallback(async () => {
    if (!user) {
      setItems([]);
      return;
    }
    setLoading(true);
    try {
      const cart = await cartApi.get();
      setItems(cart.items ?? []);
    } finally {
      setLoading(false);
    }
  }, [user]);

  useEffect(() => {
    refresh();
  }, [refresh]);

  const addItem = async (variantId: number, quantity = 1) => {
    if (!user) throw new Error('Cần đăng nhập');
    const cart = await cartApi.add(variantId, quantity);
    setItems(cart.items ?? []);
  };

  const removeItem = async (itemId: number) => {
    const cart = await cartApi.remove(itemId);
    setItems(cart.items ?? []);
  };

  const updateQuantity = async (itemId: number, qty: number) => {
    if (qty <= 0) {
      await removeItem(itemId);
      return;
    }
    const cart = await cartApi.update(itemId, qty);
    setItems(cart.items ?? []);
  };

  const clear = () => setItems([]);

  const totalItems = items.reduce((s, i) => s + i.quantity, 0);
  const totalPrice = items.reduce((s, i) => s + itemPrice(i) * i.quantity, 0);

  return (
    <CartContext.Provider
      value={{ items, loading, addItem, removeItem, updateQuantity, refresh, clear, totalItems, totalPrice }}
    >
      {children}
    </CartContext.Provider>
  );
}

export function useCart() {
  const ctx = useContext(CartContext);
  if (!ctx) throw new Error('useCart must be used within CartProvider');
  return ctx;
}
