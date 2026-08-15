const API_BASE_URL =
  import.meta.env.VITE_API_BASE_URL || "http://127.0.0.1:8000";

/**
 * Robust SSE Stream parser function reading POST response streams in JavaScript.
 * Handles split network chunks, CRLF/LF event boundaries (\r\n\r\n and \n\n),
 * multi-byte UTF-8 boundaries, multiple data events per chunk, and safe error handling.
 */
export async function streamChatResponse(
  payload,
  onEvent,
  onError,
  signal
) {
  const url = `${API_BASE_URL}/api/v1/chat/stream`;

  try {
    const response = await fetch(url, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        message: payload.message,
        conversation_id: payload.conversation_id || null,
        system_prompt: payload.system_prompt || null,
      }),
      signal,
    });

    if (!response.ok) {
      const errorText = await response.text().catch(() => "");
      let message = "Failed to communicate with Second Brain AI backend.";
      try {
        const parsed = JSON.parse(errorText);
        message = parsed.error?.message || parsed.detail || message;
      } catch {
        if (response.status === 404) {
          message = "Conversation thread not found.";
        }
      }
      onError(message);
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

      // Decode bytes using stream: true to preserve partial UTF-8 multi-byte characters
      buffer += decoder.decode(value, { stream: true });

      // Process complete SSE event blocks separated by \r\n\r\n or \n\n
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
              // Ignore reader cancellation errors
            }
            return;
          }
        }
      }
    }

    // Flush any remaining buffer text at end of stream
    const remaining = buffer.trim();
    if (remaining) {
      parseAndEmitEvent(remaining, onEvent, onError);
    }
  } catch (err) {
    if (err.name === "AbortError") {
      // Aborted by user / new chat - safe silent return (normal control flow)
      return;
    }
    onError("Unable to connect to Second Brain AI server.");
  }
}

function parseAndEmitEvent(rawEventText, onEvent, onError) {
  // Normalize line endings and split into individual lines
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
