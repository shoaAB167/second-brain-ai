import React, { createContext, useContext, useState, useEffect } from "react";
import { loginUser, registerUser } from "../services/authApi";

const AuthContext = createContext(null);

export function AuthProvider({ children }) {
  const [token, setToken] = useState(() => localStorage.getItem("sb_auth_token") || null);
  const [userEmail, setUserEmail] = useState(() => localStorage.getItem("sb_auth_user_email") || null);
  const [isAuthModalOpen, setIsAuthModalOpen] = useState(false);
  const [authError, setAuthError] = useState(null);
  const [isLoading, setIsLoading] = useState(false);

  const login = async (email, password) => {
    setIsLoading(true);
    setAuthError(null);
    try {
      const res = await loginUser({ email, password });
      const authToken = res.access_token;
      setToken(authToken);
      setUserEmail(email);
      localStorage.setItem("sb_auth_token", authToken);
      localStorage.setItem("sb_auth_user_email", email);
      setIsAuthModalOpen(false);
      return res;
    } catch (err) {
      setAuthError(err.message);
      throw err;
    } finally {
      setIsLoading(false);
    }
  };

  const register = async (email, password) => {
    setIsLoading(true);
    setAuthError(null);
    try {
      await registerUser({ email, password });
      // Automatically log in after registration
      return await login(email, password);
    } catch (err) {
      setAuthError(err.message);
      throw err;
    } finally {
      setIsLoading(false);
    }
  };

  const logout = () => {
    setToken(null);
    setUserEmail(null);
    localStorage.removeItem("sb_auth_token");
    localStorage.removeItem("sb_auth_user_email");
    localStorage.removeItem("second_brain_conversation_id");
  };

  const openAuthModal = () => {
    setAuthError(null);
    setIsAuthModalOpen(true);
  };

  const closeAuthModal = () => {
    setAuthError(null);
    setIsAuthModalOpen(false);
  };

  return (
    <AuthContext.Provider
      value={{
        token,
        userEmail,
        isAuthenticated: !!token,
        isAuthModalOpen,
        authError,
        isLoading,
        login,
        register,
        logout,
        openAuthModal,
        closeAuthModal,
      }}
    >
      {children}
    </AuthContext.Provider>
  );
}

export { useAuth } from "../hooks/useAuth";
export { AuthContext };

