import { useState } from 'react';
import { Link, useNavigate } from 'react-router-dom';
import { useAuth } from '../context/AuthContext';
import { ApiError } from '../lib/api';

export default function Register() {
  const { register } = useAuth();
  const nav = useNavigate();
  const [form, setForm] = useState({ email: '', password: '', fullName: '', phone: '' });
  const [err, setErr] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);

  const upd = (k: keyof typeof form) => (e: React.ChangeEvent<HTMLInputElement>) =>
    setForm({ ...form, [k]: e.target.value });

  const submit = async (e: React.FormEvent) => {
    e.preventDefault();
    setErr(null);
    setLoading(true);
    try {
      await register({
        email: form.email,
        password: form.password,
        fullName: form.fullName,
        phone: form.phone || undefined,
      });
      nav('/', { replace: true });
    } catch (e: any) {
      setErr(e instanceof ApiError ? e.message : 'Đăng ký thất bại');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="max-w-md mx-auto px-4 py-16">
      <h1 className="text-3xl font-semibold mb-2">Đăng ký</h1>
      <p className="text-gray-500 mb-8">Tạo tài khoản để mua sắm và thử đồ ảo.</p>
      <form onSubmit={submit} className="space-y-4">
        {[
          { k: 'fullName', label: 'Họ và tên', type: 'text', required: true },
          { k: 'email', label: 'Email', type: 'email', required: true },
          { k: 'phone', label: 'Số điện thoại', type: 'tel', required: false },
          { k: 'password', label: 'Mật khẩu (≥ 6 ký tự)', type: 'password', required: true },
        ].map((f) => (
          <div key={f.k}>
            <label className="block text-sm mb-1.5">{f.label}</label>
            <input
              type={f.type}
              required={f.required}
              minLength={f.k === 'password' ? 6 : undefined}
              value={(form as any)[f.k]}
              onChange={upd(f.k as any)}
              className="w-full border border-gray-200 rounded-md px-3 py-2.5 focus:outline-none focus:border-[var(--color-accent)]"
            />
          </div>
        ))}
        {err && <div className="text-red-600 text-sm">{err}</div>}
        <button
          type="submit"
          disabled={loading}
          className="w-full bg-[var(--color-ink)] text-white py-3 rounded-md font-medium disabled:opacity-50"
        >
          {loading ? 'Đang tạo tài khoản...' : 'Đăng ký'}
        </button>
      </form>
      <p className="text-sm text-gray-600 mt-6">
        Đã có tài khoản?{' '}
        <Link to="/login" className="text-[var(--color-accent)] font-medium">
          Đăng nhập
        </Link>
      </p>
    </div>
  );
}
