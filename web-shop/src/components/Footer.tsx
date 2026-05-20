import { Link } from "react-router-dom";
import { Instagram, Facebook, Youtube, Mail, Sparkles } from "lucide-react";

export default function Footer() {
  return (
    <footer className="bg-[var(--color-ink)] text-white mt-24">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-16">
        {/* Newsletter */}
        <div className="border-b border-white/10 pb-12 mb-12 grid grid-cols-1 md:grid-cols-2 gap-6 items-center">
          <div>
            <h3 className="font-display text-2xl md:text-3xl mb-2">
              Đăng ký nhận tin
            </h3>
            <p className="text-white/60 text-sm">
              Cập nhật bộ sưu tập mới và ưu đãi độc quyền sớm nhất.
            </p>
          </div>
          <form className="flex gap-2">
            <input
              type="email"
              placeholder="Email của bạn"
              className="flex-1 px-4 py-3 rounded-full bg-white/10 border border-white/15 text-white placeholder:text-white/40 outline-none focus:border-[var(--color-accent)]"
            />
            <button
              type="submit"
              className="px-6 py-3 rounded-full bg-[var(--color-accent)] text-[var(--color-ink)] font-semibold hover:-translate-y-0.5 transition-transform"
            >
              Đăng ký
            </button>
          </form>
        </div>

        {/* Columns */}
        <div className="grid grid-cols-2 md:grid-cols-4 gap-8 mb-12">
          <div className="col-span-2 md:col-span-1">
            <div className="flex items-center gap-2 mb-4">
              <div className="w-9 h-9 rounded-xl bg-[var(--color-accent)] flex items-center justify-center text-[var(--color-ink)] font-bold">
                V
              </div>
              <span className="text-xl font-display">
                VTON<span className="text-[var(--color-accent)]">.</span>
              </span>
            </div>
            <p className="text-sm text-white/60 leading-relaxed">
              Thời trang hiện đại tích hợp công nghệ AI thử đồ ảo. Mua sắm thông minh,
              trải nghiệm khác biệt.
            </p>
          </div>

          <div>
            <h4 className="font-semibold mb-4 text-sm uppercase tracking-wider">Mua sắm</h4>
            <ul className="space-y-2.5 text-sm text-white/60">
              <li><Link to="/" className="hover:text-[var(--color-accent)]">Nam</Link></li>
              <li><Link to="/" className="hover:text-[var(--color-accent)]">Nữ</Link></li>
              <li><Link to="/" className="hover:text-[var(--color-accent)]">Mới về</Link></li>
              <li><Link to="/" className="hover:text-[var(--color-accent)]">Sale</Link></li>
            </ul>
          </div>

          <div>
            <h4 className="font-semibold mb-4 text-sm uppercase tracking-wider">Hỗ trợ</h4>
            <ul className="space-y-2.5 text-sm text-white/60">
              <li><a href="#" className="hover:text-[var(--color-accent)]">Bảng size</a></li>
              <li><a href="#" className="hover:text-[var(--color-accent)]">Vận chuyển</a></li>
              <li><a href="#" className="hover:text-[var(--color-accent)]">Đổi trả</a></li>
              <li><a href="#" className="hover:text-[var(--color-accent)]">Liên hệ</a></li>
            </ul>
          </div>

          <div>
            <h4 className="font-semibold mb-4 text-sm uppercase tracking-wider">AI Try-On</h4>
            <ul className="space-y-2.5 text-sm text-white/60">
              <li className="flex items-center gap-1.5"><Sparkles className="w-3.5 h-3.5 text-[var(--color-accent)]" /> Thử đồ ảo</li>
              <li>TPS Warping</li>
              <li>SegFormer Parsing</li>
              <li>Diffusion Refine</li>
            </ul>
          </div>
        </div>

        {/* Bottom */}
        <div className="border-t border-white/10 pt-8 flex flex-col md:flex-row items-center justify-between gap-4">
          <p className="text-sm text-white/40">
            © 2026 VTON Shop — Đồ án tốt nghiệp. All rights reserved.
          </p>
          <div className="flex items-center gap-4 text-white/60">
            <a href="#" className="hover:text-[var(--color-accent)]"><Instagram className="w-5 h-5" /></a>
            <a href="#" className="hover:text-[var(--color-accent)]"><Facebook className="w-5 h-5" /></a>
            <a href="#" className="hover:text-[var(--color-accent)]"><Youtube className="w-5 h-5" /></a>
            <a href="#" className="hover:text-[var(--color-accent)]"><Mail className="w-5 h-5" /></a>
          </div>
        </div>
      </div>
    </footer>
  );
}
