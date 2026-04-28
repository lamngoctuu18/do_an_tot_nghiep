import { Link } from "react-router-dom";
import { ShoppingCart, Shirt, Menu, X } from "lucide-react";
import { useCart } from "../context/CartContext";
import { useState } from "react";

export default function Navbar() {
  const { totalItems } = useCart();
  const [open, setOpen] = useState(false);

  return (
    <nav className="sticky top-0 z-50 bg-surface/80 backdrop-blur-xl border-b border-white/10">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex items-center justify-between h-16">
          <Link to="/" className="flex items-center gap-2 group">
            <Shirt className="w-8 h-8 text-primary group-hover:text-accent transition-colors" />
            <span className="text-xl font-bold bg-gradient-to-r from-primary to-accent bg-clip-text text-transparent">
              VTON Shop
            </span>
          </Link>

          {/* Desktop nav */}
          <div className="hidden md:flex items-center gap-8">
            <Link to="/" className="text-sm text-gray-300 hover:text-white transition-colors">
              Sản phẩm
            </Link>
            <Link to="/try-on" className="text-sm text-gray-300 hover:text-white transition-colors flex items-center gap-1">
              <Shirt className="w-4 h-4" /> Thử đồ AI
            </Link>
            <Link to="/cart" className="relative p-2 hover:bg-white/5 rounded-lg transition-colors">
              <ShoppingCart className="w-5 h-5" />
              {totalItems > 0 && (
                <span className="absolute -top-1 -right-1 w-5 h-5 bg-accent text-xs rounded-full flex items-center justify-center text-black font-bold">
                  {totalItems}
                </span>
              )}
            </Link>
          </div>

          {/* Mobile toggle */}
          <button onClick={() => setOpen(!open)} className="md:hidden p-2">
            {open ? <X className="w-6 h-6" /> : <Menu className="w-6 h-6" />}
          </button>
        </div>

        {/* Mobile menu */}
        {open && (
          <div className="md:hidden pb-4 space-y-2">
            <Link to="/" onClick={() => setOpen(false)} className="block px-3 py-2 rounded-lg hover:bg-white/5">
              Sản phẩm
            </Link>
            <Link to="/try-on" onClick={() => setOpen(false)} className="block px-3 py-2 rounded-lg hover:bg-white/5">
              🤖 Thử đồ AI
            </Link>
            <Link to="/cart" onClick={() => setOpen(false)} className="block px-3 py-2 rounded-lg hover:bg-white/5">
              🛒 Giỏ hàng ({totalItems})
            </Link>
          </div>
        )}
      </div>
    </nav>
  );
}
