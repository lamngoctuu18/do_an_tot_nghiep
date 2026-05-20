import { Link, NavLink, useNavigate } from "react-router-dom";
import { ShoppingBag, Search, User, Menu, X, Sparkles, Heart, LogOut, Package, LayoutDashboard } from "lucide-react";
import { useCart } from "../context/CartContext";
import { useAuth } from "../context/AuthContext";
import { useEffect, useRef, useState } from "react";
import { getInitials } from "../lib/initials";

const NAV_LINKS = [
  { to: "/", label: "Trang chủ" },
  { to: "/?category=Nam", label: "Nam" },
  { to: "/?category=Nữ", label: "Nữ" },
  { to: "/?category=Áo+hoodie", label: "Hoodie" },
  { to: "/?category=sale", label: "Sale" },
];

export default function Navbar() {
  const { totalItems } = useCart();
  const { user, logout } = useAuth();
  const nav = useNavigate();
  const [open, setOpen] = useState(false);
  const [scrolled, setScrolled] = useState(false);
  const [userMenu, setUserMenu] = useState(false);
  const menuRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const onScroll = () => setScrolled(window.scrollY > 8);
    window.addEventListener("scroll", onScroll);
    return () => window.removeEventListener("scroll", onScroll);
  }, []);

  useEffect(() => {
    const onClick = (e: MouseEvent) => {
      if (menuRef.current && !menuRef.current.contains(e.target as Node)) setUserMenu(false);
    };
    document.addEventListener("mousedown", onClick);
    return () => document.removeEventListener("mousedown", onClick);
  }, []);

  const handleLogout = async () => {
    await logout();
    setUserMenu(false);
    nav("/");
  };

  return (
    <header
      className={`sticky top-0 z-50 bg-white transition-all duration-300 ${
        scrolled ? "shadow-[0_2px_8px_rgba(17,24,39,0.06)]" : ""
      } border-b border-[var(--color-line)]`}
    >
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex items-center justify-between h-[72px]">
          {/* Logo */}
          <Link to="/" className="flex items-center gap-2 group">
            <div className="w-9 h-9 rounded-xl bg-black flex items-center justify-center text-white font-bold tracking-tight">
              V
            </div>
            <span className="text-xl font-display font-semibold text-[var(--color-ink)]">
              VTON<span className="text-[var(--color-accent)]">.</span>
            </span>
          </Link>

          {/* Desktop nav */}
          <nav className="hidden lg:flex items-center gap-8">
            {NAV_LINKS.map((l) => (
              <NavLink
                key={l.to}
                to={l.to}
                end={l.to === "/"}
                className={({ isActive }) =>
                  `text-sm font-medium transition-colors ${
                    isActive
                      ? "text-[var(--color-ink)]"
                      : "text-[var(--color-ink-muted)] hover:text-[var(--color-ink)]"
                  }`
                }
              >
                {l.label}
              </NavLink>
            ))}
          </nav>

          {/* Right actions */}
          <div className="flex items-center gap-2">
            <button className="hidden sm:inline-flex p-2.5 rounded-full text-[var(--color-ink)] hover:bg-[var(--color-bg-soft)] transition-colors">
              <Search className="w-5 h-5" />
            </button>

            <Link
              to="/try-on"
              className="hidden md:inline-flex items-center gap-1.5 px-4 py-2 rounded-full text-sm font-semibold text-[var(--color-ink)] bg-[var(--color-accent)] hover:-translate-y-0.5 transition-transform"
              style={{ boxShadow: "0 4px 14px rgba(214,185,140,0.35)" }}
            >
              <Sparkles className="w-4 h-4" />
              Thử đồ ảo
            </Link>

            <button className="hidden sm:inline-flex p-2.5 rounded-full hover:bg-[var(--color-bg-soft)] transition-colors">
              <Heart className="w-5 h-5" />
            </button>

            {!user ? (
              <div className="hidden sm:flex items-center gap-2">
                <Link
                  to="/login"
                  className="px-3.5 py-2 rounded-full text-sm font-medium text-[var(--color-ink)] hover:bg-[var(--color-bg-soft)] transition-colors"
                >
                  Đăng nhập
                </Link>
                <Link
                  to="/register"
                  className="px-3.5 py-2 rounded-full text-sm font-semibold bg-[var(--color-ink)] text-white hover:opacity-90 transition-opacity"
                >
                  Đăng ký
                </Link>
              </div>
            ) : null}

            <div className="relative" ref={menuRef}>
              {user && (
                <button
                  onClick={() => setUserMenu((v) => !v)}
                  className="hidden sm:inline-flex items-center justify-center w-10 h-10 rounded-full bg-[var(--color-ink)] text-white text-xs font-bold tracking-wide hover:opacity-90 transition-opacity"
                  title={user.fullName}
                >
                  {getInitials(user.fullName)}
                </button>
              )}
              {user && userMenu && (
                <div className="absolute right-0 mt-2 w-64 max-w-[calc(100vw-2rem)] bg-white border border-[var(--color-line)] rounded-xl shadow-lg py-2 z-50">
                  <div className="px-4 py-2 border-b border-[var(--color-line)]">
                    <div className="font-semibold text-sm truncate">{user.fullName}</div>
                    <div className="text-xs text-gray-500 truncate">{user.email}</div>
                    <div className="text-[10px] uppercase tracking-wide text-[var(--color-accent)] font-bold mt-0.5">
                      {user.role === 'ADMIN' ? 'Quản trị viên' : user.role === 'SELLER' ? 'Người bán' : 'Khách hàng'}
                    </div>
                  </div>
                  {user.role === 'ADMIN' ? (
                    <Link to="/admin" onClick={() => setUserMenu(false)} className="flex items-center gap-2 px-4 py-2 hover:bg-[var(--color-bg-soft)] text-sm">
                      <LayoutDashboard className="w-4 h-4" /> Trang quản trị
                    </Link>
                  ) : user.role === 'SELLER' ? (
                    <Link to="/seller" onClick={() => setUserMenu(false)} className="flex items-center gap-2 px-4 py-2 hover:bg-[var(--color-bg-soft)] text-sm">
                      <LayoutDashboard className="w-4 h-4" /> Trang quản lý shop
                    </Link>
                  ) : (
                    <>
                      <Link to="/account" onClick={() => setUserMenu(false)} className="flex items-center gap-2 px-4 py-2 hover:bg-[var(--color-bg-soft)] text-sm">
                        <User className="w-4 h-4" /> Tài khoản
                      </Link>
                      <Link to="/account/orders" onClick={() => setUserMenu(false)} className="flex items-center gap-2 px-4 py-2 hover:bg-[var(--color-bg-soft)] text-sm">
                        <Package className="w-4 h-4" /> Đơn hàng
                      </Link>
                    </>
                  )}
                  <button
                    onClick={handleLogout}
                    className="flex items-center gap-2 w-full text-left px-4 py-2 hover:bg-[var(--color-bg-soft)] text-sm text-red-600 border-t border-[var(--color-line)] mt-1 pt-2"
                  >
                    <LogOut className="w-4 h-4" /> Đăng xuất
                  </button>
                </div>
              )}
            </div>

            <Link
              to="/cart"
              className="relative p-2.5 rounded-full hover:bg-[var(--color-bg-soft)] transition-colors"
            >
              <ShoppingBag className="w-5 h-5" />
              {totalItems > 0 && (
                <span className="absolute -top-0.5 -right-0.5 min-w-[20px] h-5 px-1 bg-[var(--color-primary)] text-white text-[11px] font-bold rounded-full flex items-center justify-center">
                  {totalItems}
                </span>
              )}
            </Link>

            <button
              onClick={() => setOpen(!open)}
              className="lg:hidden p-2.5 rounded-full hover:bg-[var(--color-bg-soft)]"
            >
              {open ? <X className="w-5 h-5" /> : <Menu className="w-5 h-5" />}
            </button>
          </div>
        </div>

        {/* Mobile menu */}
        {open && (
          <div className="lg:hidden pb-4 space-y-1 animate-fade-up">
            {NAV_LINKS.map((l) => (
              <Link
                key={l.to}
                to={l.to}
                onClick={() => setOpen(false)}
                className="block px-3 py-2.5 rounded-lg text-[var(--color-ink)] hover:bg-[var(--color-bg-soft)]"
              >
                {l.label}
              </Link>
            ))}
            <Link
              to="/try-on"
              onClick={() => setOpen(false)}
              className="flex items-center gap-2 px-3 py-2.5 rounded-lg bg-[var(--color-accent)] font-semibold"
            >
              <Sparkles className="w-4 h-4" /> Thử đồ ảo
            </Link>
            {!user ? (
              <div className="grid grid-cols-2 gap-2 pt-2">
                <Link
                  to="/login"
                  onClick={() => setOpen(false)}
                  className="px-3 py-2.5 rounded-lg border border-[var(--color-line)] text-center text-sm font-medium"
                >
                  Đăng nhập
                </Link>
                <Link
                  to="/register"
                  onClick={() => setOpen(false)}
                  className="px-3 py-2.5 rounded-lg bg-[var(--color-ink)] text-white text-center text-sm font-semibold"
                >
                  Đăng ký
                </Link>
              </div>
            ) : (
              <div className="flex items-center gap-3 pt-2 px-3">
                <div className="w-9 h-9 rounded-full bg-[var(--color-ink)] text-white text-xs font-bold flex items-center justify-center">
                  {getInitials(user.fullName)}
                </div>
                <div className="text-sm font-medium truncate">{user.fullName}</div>
              </div>
            )}
          </div>
        )}
      </div>
    </header>
  );
}
