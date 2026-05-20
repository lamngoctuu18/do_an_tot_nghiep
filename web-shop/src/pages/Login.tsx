import { useState } from 'react';
import { Link, useNavigate, useLocation } from 'react-router-dom';
import { useAuth } from '../context/AuthContext';
import { ApiError, tokenStore } from '../lib/api';

export default function Login() {
  const { login } = useAuth();
  const nav = useNavigate();
  const loc = useLocation();

  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [err, setErr] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);

  const submit = async (e: React.FormEvent) => {
    e.preventDefault();
    setErr(null);
    setLoading(true);
    try {
      await login(email, password);
      const stateRedirect = (loc.state as any)?.from;
      const role = tokenStore.user?.role;
      const target = stateRedirect ?? (role === 'ADMIN' ? '/admin' : role === 'SELLER' ? '/seller' : '/');
      nav(target, { replace: true });
    } catch (e: any) {
      setErr(e instanceof ApiError ? e.message : 'Đăng nhập thất bại');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="max-w-md mx-auto px-4 py-16">
      <h1 className="text-3xl font-semibold mb-2">Đăng nhập</h1>
      <p className="text-gray-500 mb-8">Chào mừng quay lại với VTON Fashion.</p>
      <form onSubmit={submit} className="space-y-4">
        <div>
          <label className="block text-sm mb-1.5">Email</label>
          <input
            type="email"
            required
            value={email}
            onChange={(e) => setEmail(e.target.value)}
            className="w-full border border-gray-200 rounded-md px-3 py-2.5 focus:outline-none focus:border-[var(--color-accent)]"
          />
        </div>
        <div>
          <label className="block text-sm mb-1.5">Mật khẩu</label>
          <input
            type="password"
            required
            value={password}
            onChange={(e) => setPassword(e.target.value)}
            className="w-full border border-gray-200 rounded-md px-3 py-2.5 focus:outline-none focus:border-[var(--color-accent)]"
          />
        </div>
        {err && <div className="text-red-600 text-sm">{err}</div>}
        <button
          type="submit"
          disabled={loading}
          className="w-full bg-[var(--color-ink)] text-white py-3 rounded-md font-medium disabled:opacity-50"
        >
          {loading ? 'Đang đăng nhập...' : 'Đăng nhập'}
        </button>
      </form>
      <p className="text-sm text-gray-600 mt-6">
        Chưa có tài khoản?{' '}
        <Link to="/register" className="text-[var(--color-accent)] font-medium">
          Đăng ký ngay
        </Link>
      </p>
    </div>
  );
}
