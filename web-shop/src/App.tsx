import { BrowserRouter, Routes, Route, Navigate, useLocation } from "react-router-dom";
import Navbar from "./components/Navbar";
import Footer from "./components/Footer";
import ProtectedRoute from "./components/ProtectedRoute";
import AccountLayout from "./components/AccountLayout";
import SellerLayout from "./components/SellerLayout";
import AdminLayout from "./components/AdminLayout";
import Home from "./pages/Home";
import ProductDetail from "./pages/ProductDetail";
import TryOn from "./pages/TryOn";
import Cart from "./pages/Cart";
import Checkout from "./pages/Checkout";
import Login from "./pages/Login";
import Register from "./pages/Register";
import Profile from "./pages/account/Profile";
import Addresses from "./pages/account/Addresses";
import Orders from "./pages/account/Orders";
import OrderDetail from "./pages/account/OrderDetail";
import Wishlist from "./pages/account/Wishlist";
import Notifications from "./pages/account/Notifications";
import SellerDashboard from "./pages/seller/SellerDashboard";
import SellerOrders from "./pages/seller/SellerOrders";
import SellerOrderDetail from "./pages/seller/SellerOrderDetail";
import SellerTryonStats from "./pages/seller/SellerTryonStats";
import SellerProducts from "./pages/seller/SellerProducts";
import SellerProductForm from "./pages/seller/SellerProductForm";
import AdminDashboard from "./pages/admin/AdminDashboard";
import AdminUsers from "./pages/admin/AdminUsers";
import AdminShops from "./pages/admin/AdminShops";
import AdminProducts from "./pages/admin/AdminProducts";
import AdminReviews from "./pages/admin/AdminReviews";
import AdminReports from "./pages/admin/AdminReports";
import { CartProvider } from "./context/CartContext";
import { AuthProvider, useAuth } from "./context/AuthContext";
import { ToastProvider } from "./components/Toast";
import { ChatbotProvider, ChatbotFloatingButton, ChatbotWindow } from "./components/chatbot";

export default function App() {
  return (
    <AuthProvider>
      <ToastProvider>
        <CartProvider>
          <BrowserRouter>
            <ChatbotProvider>
              <AppShell />
            </ChatbotProvider>
          </BrowserRouter>
        </CartProvider>
      </ToastProvider>
    </AuthProvider>
  );
}

function AppShell() {
  const { pathname } = useLocation();
  const { user } = useAuth();
  const isDashboard = pathname.startsWith("/admin") || pathname.startsWith("/seller");

  if (user?.role === "ADMIN" && !pathname.startsWith("/admin")) {
    return <Navigate to="/admin" replace />;
  }
  if (user?.role === "SELLER" && !pathname.startsWith("/seller")) {
    return <Navigate to="/seller" replace />;
  }

  return (
    <div className="min-h-screen bg-white text-[var(--color-ink)] flex flex-col">
      {!isDashboard && <Navbar />}
      <main className="flex-1">
        <Routes>
                <Route path="/" element={<Home />} />
                <Route path="/product/:id" element={<ProductDetail />} />
                <Route path="/try-on" element={<TryOn />} />
                <Route path="/try-on/:id" element={<TryOn />} />
                <Route path="/cart" element={<Cart />} />
                <Route path="/login" element={<Login />} />
                <Route path="/register" element={<Register />} />
                <Route
                  path="/checkout"
                  element={
                    <ProtectedRoute>
                      <Checkout />
                    </ProtectedRoute>
                  }
                />
                <Route
                  path="/account"
                  element={
                    <ProtectedRoute>
                      <AccountLayout />
                    </ProtectedRoute>
                  }
                >
                  <Route index element={<Profile />} />
                  <Route path="addresses" element={<Addresses />} />
                  <Route path="orders" element={<Orders />} />
                  <Route path="orders/:code" element={<OrderDetail />} />
                  <Route path="wishlist" element={<Wishlist />} />
                  <Route path="notifications" element={<Notifications />} />
                </Route>
                <Route
                  path="/seller"
                  element={
                    <ProtectedRoute roles={["SELLER", "ADMIN"]}>
                      <SellerLayout />
                    </ProtectedRoute>
                  }
                >
                  <Route index element={<SellerDashboard />} />
                  <Route path="products" element={<SellerProducts />} />
                  <Route path="products/new" element={<SellerProductForm />} />
                  <Route path="products/:id" element={<SellerProductForm />} />
                  <Route path="orders" element={<SellerOrders />} />
                  <Route path="orders/:code" element={<SellerOrderDetail />} />
                  <Route path="tryon-stats" element={<SellerTryonStats />} />
                </Route>
                <Route
                  path="/admin"
                  element={
                    <ProtectedRoute roles={["ADMIN"]}>
                      <AdminLayout />
                    </ProtectedRoute>
                  }
                >
                  <Route index element={<AdminDashboard />} />
                  <Route path="users" element={<AdminUsers />} />
                  <Route path="shops" element={<AdminShops />} />
                  <Route path="products" element={<AdminProducts />} />
                  <Route path="reviews" element={<AdminReviews />} />
                  <Route path="reports" element={<AdminReports />} />
                </Route>
                <Route path="*" element={<Navigate to="/" replace />} />
              </Routes>
            </main>
      {!isDashboard && <Footer />}
      {!isDashboard && (
        <>
          <ChatbotFloatingButton />
          <ChatbotWindow />
        </>
      )}
    </div>
  );
}
