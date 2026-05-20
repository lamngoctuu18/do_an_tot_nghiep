import { Navigate, useLocation } from 'react-router-dom';
import { useAuth } from '../context/AuthContext';
import type { ReactNode } from 'react';

export default function ProtectedRoute({
  children,
  roles,
}: {
  children: ReactNode;
  roles?: ('CUSTOMER' | 'SELLER' | 'ADMIN')[];
}) {
  const { user, loading } = useAuth();
  const loc = useLocation();

  if (loading) {
    return <div className="py-20 text-center text-gray-500">Đang tải...</div>;
  }
  if (!user) {
    return <Navigate to="/login" state={{ from: loc.pathname }} replace />;
  }
  if (roles && !roles.includes(user.role)) {
    return <Navigate to="/" replace />;
  }
  return <>{children}</>;
}
