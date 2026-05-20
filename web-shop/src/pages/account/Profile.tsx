import { useEffect, useState } from 'react';
import { useAuth } from '../../context/AuthContext';
import { usersApi, ApiError } from '../../lib/api';

export default function Profile() {
  const { user, refreshUser } = useAuth();
  const [form, setForm] = useState({
    fullName: '',
    phone: '',
    gender: '' as '' | 'MALE' | 'FEMALE' | 'OTHER',
    dateOfBirth: '',
    avatarUrl: '',
  });
  const [msg, setMsg] = useState<{ type: 'ok' | 'err'; text: string } | null>(null);
  const [saving, setSaving] = useState(false);

  useEffect(() => {
    if (!user) return;
    setForm({
      fullName: user.fullName ?? '',
      phone: (user as any).phone ?? '',
      gender: (user as any).gender ?? '',
      dateOfBirth: (user as any).dateOfBirth ? String((user as any).dateOfBirth).slice(0, 10) : '',
      avatarUrl: user.avatarUrl ?? '',
    });
  }, [user]);

  const submit = async (e: React.FormEvent) => {
    e.preventDefault();
    setSaving(true);
    setMsg(null);
    try {
      const payload: any = {
        fullName: form.fullName,
        phone: form.phone || undefined,
        avatarUrl: form.avatarUrl || undefined,
      };
      if (form.gender) payload.gender = form.gender;
      if (form.dateOfBirth) payload.dateOfBirth = form.dateOfBirth;
      await usersApi.updateProfile(payload);
      await refreshUser();
      setMsg({ type: 'ok', text: 'Đã cập nhật hồ sơ.' });
    } catch (e: any) {
      setMsg({ type: 'err', text: e instanceof ApiError ? e.message : 'Lỗi cập nhật' });
    } finally {
      setSaving(false);
    }
  };

  return (
    <div>
      <h2 className="text-xl font-semibold mb-4">Hồ sơ cá nhân</h2>
      <form onSubmit={submit} className="space-y-4 max-w-xl">
        <div>
          <label className="block text-sm mb-1.5">Email</label>
          <input
            disabled
            value={user?.email ?? ''}
            className="w-full border border-gray-200 rounded-md px-3 py-2.5 bg-gray-50 text-gray-500"
          />
        </div>
        <div>
          <label className="block text-sm mb-1.5">Họ và tên</label>
          <input
            value={form.fullName}
            onChange={(e) => setForm({ ...form, fullName: e.target.value })}
            className="w-full border border-gray-200 rounded-md px-3 py-2.5"
          />
        </div>
        <div className="grid grid-cols-2 gap-4">
          <div>
            <label className="block text-sm mb-1.5">Số điện thoại</label>
            <input
              value={form.phone}
              onChange={(e) => setForm({ ...form, phone: e.target.value })}
              className="w-full border border-gray-200 rounded-md px-3 py-2.5"
            />
          </div>
          <div>
            <label className="block text-sm mb-1.5">Giới tính</label>
            <select
              value={form.gender}
              onChange={(e) => setForm({ ...form, gender: e.target.value as any })}
              className="w-full border border-gray-200 rounded-md px-3 py-2.5 bg-white"
            >
              <option value="">--</option>
              <option value="MALE">Nam</option>
              <option value="FEMALE">Nữ</option>
              <option value="OTHER">Khác</option>
            </select>
          </div>
        </div>
        <div>
          <label className="block text-sm mb-1.5">Ngày sinh</label>
          <input
            type="date"
            value={form.dateOfBirth}
            onChange={(e) => setForm({ ...form, dateOfBirth: e.target.value })}
            className="w-full border border-gray-200 rounded-md px-3 py-2.5"
          />
        </div>
        <div>
          <label className="block text-sm mb-1.5">Avatar URL</label>
          <input
            value={form.avatarUrl}
            onChange={(e) => setForm({ ...form, avatarUrl: e.target.value })}
            className="w-full border border-gray-200 rounded-md px-3 py-2.5"
          />
        </div>
        {msg && (
          <div className={`text-sm ${msg.type === 'ok' ? 'text-green-600' : 'text-red-600'}`}>
            {msg.text}
          </div>
        )}
        <button
          disabled={saving}
          className="bg-[var(--color-ink)] text-white px-6 py-2.5 rounded-md font-medium disabled:opacity-50"
        >
          {saving ? 'Đang lưu...' : 'Lưu thay đổi'}
        </button>
      </form>
    </div>
  );
}
