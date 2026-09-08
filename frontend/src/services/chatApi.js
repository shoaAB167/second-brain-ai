import { refreshToken } from "./authApi";

const API_BASE_URL =
  import.meta.env.VITE_API_BASE_URL || "http://localhost:8000";

/**
 * Robust SSE Stream parser reading POST response streams in JavaScript.
 * Automatically handles 401 token refresh retry, split network chunks,
 * CRLF/LF event boundaries, multi-byte UTF-8 boundaries, and cancellation.
 */
export async function streamChatResponse(
  payload,
  onEvent,
  onError,
  signal,
  isRetry = false
) {
  const url = `${API_BASE_URL}/api/v1/chat/stream`;
  let token = payload.token || localStorage.getItem("sb_auth_token");

  const headers = {
    "Content-Type": "application/json",
  };

  if (token) {
    headers["Authorization"] = `Bearer ${token}`;
  }

  try {
    const response = await fetch(url, {
      method: "POST",
      headers,
      body: JSON.stringify({
        message: payload.message,
        conversation_id: payload.conversation_id || null,
        system_prompt: payload.system_prompt || null,
      }),
      signal,
    });

    if (!response.ok) {
      // If 401 Unauthorized and not already a retry, attempt automatic silent token refresh
      if (response.status === 401 && !isRetry && token) {
        try {
          const newToken = await refreshToken(token);
          if (newToken) {
            return await streamChatResponse(
              { ...payload, token: newToken },
              onEvent,
              onError,
              signal,
              true
            );
          }
        } catch {
          // Token refresh failed, fall through to reporting 401 error
        }
      }

      const errorText = await response.text().catch(() => "");
      let message = "Failed to communicate with Second Brain companion.";
      try {
        const parsed = JSON.parse(errorText);
        message = parsed.error?.message || parsed.detail || message;
      } catch {
        if (response.status === 401) {
          message = "Authentication session expired. Please log in again.";
        } else if (response.status === 404) {
          message = "Conversation thread not found.";
        }
      }
      onError(message, response.status);
      return;
    }

    if (!response.body) {
      onError("No response stream received from backend.");
      return;
    }

    const reader = response.body.getReader();
    const decoder = new TextDecoder("utf-8");
    let buffer = "";

    while (true) {
      const { done, value } = await reader.read();
      if (done) {
        break;
      }

      buffer += decoder.decode(value, { stream: true });

      while (true) {
        const match = buffer.match(/\r?\n\r?\n/);
        if (!match) break;

        const boundaryIndex = match.index;
        const delimiterLength = match[0].length;

        const eventBlock = buffer.slice(0, boundaryIndex).trim();
        buffer = buffer.slice(boundaryIndex + delimiterLength);

        if (eventBlock) {
          const success = parseAndEmitEvent(eventBlock, onEvent, onError);
          if (!success) {
            try {
              await reader.cancel();
            } catch {
              // Ignore cancellation errors
            }
            return;
          }
        }
      }
    }

    const remaining = buffer.trim();
    if (remaining) {
      parseAndEmitEvent(remaining, onEvent, onError);
    }
  } catch (err) {
    if (err.name === "AbortError") {
      return;
    }
    onError("Unable to connect to Second Brain. Please ensure the server is running.");
  }
}

function parseAndEmitEvent(rawEventText, onEvent, onError) {
  const lines = rawEventText.split(/\r?\n/);
  for (const line of lines) {
    const trimmed = line.trim();
    if (trimmed.startsWith("data:")) {
      const jsonStr = trimmed.slice(5).trim();
      if (jsonStr) {
        try {
          const parsed = JSON.parse(jsonStr);
          onEvent(parsed);
        } catch (err) {
          if (import.meta.env?.DEV) {
            console.warn("Failed to parse SSE event JSON payload:", jsonStr, err);
          }
          onError("Unable to process the response stream. Please try again.");
          return false;
        }
      }
    }
  }
  return true;
}
