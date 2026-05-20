import { NavLink, Outlet, Link, useNavigate } from 'react-router-dom';
import { LayoutDashboard, Users, Store, Package, BarChart3, MessageSquare, LogOut } from 'lucide-react';
import { useAuth } from '../context/AuthContext';
import { getInitials } from '../lib/initials';

const ITEMS = [
  { to: '/admin', label: 'Tổng quan', icon: LayoutDashboard, end: true },
  { to: '/admin/users', label: 'Người dùng', icon: Users },
  { to: '/admin/shops', label: 'Cửa hàng', icon: Store },
  { to: '/admin/products', label: 'Duyệt sản phẩm', icon: Package },
  { to: '/admin/reviews', label: 'Đánh giá', icon: MessageSquare },
  { to: '/admin/reports', label: 'Báo cáo', icon: BarChart3 },
];

export default function AdminLayout() {
  const { user, logout } = useAuth();
  const nav = useNavigate();
  const handleLogout = async () => {
    await logout();
    nav('/');
  };

  return (
    <div className="min-h-screen bg-[var(--color-bg-soft)] flex flex-col">
      <header className="bg-white border-b border-[var(--color-line)]">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 h-14 flex items-center justify-between">
          <div className="flex items-center gap-3">
            <Link to="/admin" className="flex items-center gap-2">
              <div className="w-8 h-8 rounded-lg bg-black text-white font-bold flex items-center justify-center">V</div>
              <span className="font-display font-semibold">Admin</span>
            </Link>
          </div>
          <div className="flex items-center gap-2">
            {user && (
              <div className="flex items-center gap-2">
                <div className="w-8 h-8 rounded-full bg-[var(--color-ink)] text-white text-[11px] font-bold flex items-center justify-center" title={user.fullName}>
                  {getInitials(user.fullName)}
                </div>
                <span className="hidden sm:block text-sm font-medium truncate max-w-[140px]">{user.fullName}</span>
                <button onClick={handleLogout} className="p-2 rounded-md text-gray-600 hover:bg-gray-50" title="Đăng xuất">
                  <LogOut className="w-4 h-4" />
                </button>
              </div>
            )}
          </div>
        </div>
      </header>
      <div className="flex-1 max-w-7xl w-full mx-auto px-4 sm:px-6 lg:px-8 py-6">
        <h1 className="text-xl md:text-2xl font-semibold mb-4 md:mb-6">Trang quản trị</h1>
        <div className="grid grid-cols-1 md:grid-cols-[220px_1fr] gap-4 md:gap-8">
          <aside className="md:space-y-1 -mx-4 px-4 md:mx-0 md:px-0 overflow-x-auto md:overflow-visible">
            <div className="flex md:flex-col gap-1 min-w-max md:min-w-0 pb-1 md:pb-0">
              {ITEMS.map((it) => (
                <NavLink
                  key={it.to}
                  to={it.to}
                  end={it.end}
                  className={({ isActive }) =>
                    `flex items-center gap-2 md:gap-3 px-3 py-2 md:py-2.5 rounded-md text-sm whitespace-nowrap ${
                      isActive ? 'bg-white font-semibold shadow-sm' : 'text-gray-600 hover:bg-white/60'
                    }`
                  }
                >
                  <it.icon className="w-4 h-4" /> {it.label}
                </NavLink>
              ))}
            </div>
          </aside>
          <section className="min-w-0 bg-white rounded-lg border border-[var(--color-line)] p-4 md:p-6">
            <Outlet />
          </section>
        </div>
      </div>
    </div>
  );
}
