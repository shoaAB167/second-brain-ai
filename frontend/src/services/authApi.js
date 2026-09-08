const API_BASE_URL =
  import.meta.env.VITE_API_BASE_URL || "http://localhost:8000";

let refreshPromise = null;

/**
 * Register a new user with email and password.
 * @param {Object} credentials - { email, password }
 * @returns {Promise<Object>} Safe UserResponse object
 */
export async function registerUser({ email, password }) {
  let response;
  try {
    response = await fetch(`${API_BASE_URL}/api/v1/auth/register`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({ email, password }),
    });
  } catch (err) {
    throw new Error(
      "Unable to connect to Second Brain AI server. Please make sure the backend server is running."
    );
  }

  if (!response.ok) {
    let errorMsg = "Registration failed. Please try again.";
    try {
      const data = await response.json();
      if (data.error?.message) {
        errorMsg = data.error.message;
      } else if (data.detail) {
        errorMsg = typeof data.detail === "string" ? data.detail : JSON.stringify(data.detail);
      } else if (data.message) {
        errorMsg = data.message;
      }
    } catch {
      // Ignore JSON parse errors
    }
    throw new Error(errorMsg);
  }

  return await response.json();
}

/**
 * Authenticate user credentials and return JWT bearer token.
 * @param {Object} credentials - { email, password }
 * @returns {Promise<Object>} { access_token, token_type }
 */
export async function loginUser({ email, password }) {
  let response;
  try {
    response = await fetch(`${API_BASE_URL}/api/v1/auth/login`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({ email, password }),
    });
  } catch (err) {
    throw new Error(
      "Unable to connect to Second Brain AI server. Please make sure the backend server is running."
    );
  }

  if (!response.ok) {
    let errorMsg = "Invalid email or password.";
    try {
      const data = await response.json();
      if (data.error?.message) {
        errorMsg = data.error.message;
      } else if (data.detail) {
        errorMsg = typeof data.detail === "string" ? data.detail : JSON.stringify(data.detail);
      } else if (data.message) {
        errorMsg = data.message;
      }
    } catch {
      // Ignore JSON parse errors
    }
    throw new Error(errorMsg);
  }

  return await response.json();
}

/**
 * Refresh current access token using deduplicated in-flight request.
 * @param {string} [token] - Optional token to refresh, defaults to localStorage token.
 * @returns {Promise<string>} New access token string.
 */
export async function refreshToken(token = null) {
  const currentToken = token || localStorage.getItem("sb_auth_token");
  if (!currentToken) {
    throw new Error("No active token to refresh.");
  }

  // Deduplicate concurrent refresh calls
  if (refreshPromise) {
    return refreshPromise;
  }

  refreshPromise = (async () => {
    try {
      const response = await fetch(`${API_BASE_URL}/api/v1/auth/refresh`, {
        method: "POST",
        headers: {
          Authorization: `Bearer ${currentToken}`,
          "Content-Type": "application/json",
        },
      });

      if (!response.ok) {
        throw new Error("Failed to refresh session token.");
      }

      const data = await response.json();
      const newToken = data.access_token;
      if (newToken) {
        localStorage.setItem("sb_auth_token", newToken);
      }
      return newToken;
    } finally {
      refreshPromise = null;
    }
  })();

  return refreshPromise;
}
