import { NavLink, Outlet } from 'react-router-dom';
import { User, MapPin, Package, Heart, Bell } from 'lucide-react';

const ITEMS = [
  { to: '/account', label: 'Hồ sơ', icon: User, end: true },
  { to: '/account/addresses', label: 'Địa chỉ', icon: MapPin },
  { to: '/account/orders', label: 'Đơn hàng', icon: Package },
  { to: '/account/wishlist', label: 'Yêu thích', icon: Heart },
  { to: '/account/notifications', label: 'Thông báo', icon: Bell },
];

export default function AccountLayout() {
  return (
    <div className="max-w-6xl mx-auto px-4 py-6 md:py-10">
      <h1 className="text-xl md:text-2xl font-semibold mb-4 md:mb-6">Tài khoản của tôi</h1>
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
                    isActive
                      ? 'bg-[var(--color-bg-soft)] text-[var(--color-ink)] font-semibold'
                      : 'text-gray-600 hover:bg-gray-50'
                  }`
                }
              >
                <it.icon className="w-4 h-4" />
                {it.label}
              </NavLink>
            ))}
          </div>
        </aside>
        <section className="min-w-0">
          <Outlet />
        </section>
      </div>
    </div>
  );
}
