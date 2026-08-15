import { describe, expect, it, vi, beforeEach, afterEach } from "vitest";
import React, { act } from "react";
import { createRoot } from "react-dom/client";
import { useChat } from "./useChat.js";
import * as chatApi from "../services/chatApi.js";

const LOCAL_STORAGE_KEY = "second_brain_conversation_id";

// Custom hook test runner helper
function renderUseChatHook() {
  let hookResult = {};

  function TestComponent() {
    hookResult.current = useChat();
    return null;
  }

  const container = document.createElement("div");
  document.body.appendChild(container);
  const root = createRoot(container);

  act(() => {
    root.render(<TestComponent />);
  });

  return {
    get current() {
      return hookResult.current;
    },
    unmount() {
      act(() => {
        root.unmount();
      });
      container.remove();
    },
  };
}

describe("useChat hook behavior", () => {
  beforeEach(() => {
    vi.restoreAllMocks();
    localStorage.clear();
    if (!globalThis.crypto) {
      globalThis.crypto = {};
    }
    if (!globalThis.crypto.randomUUID) {
      let idCounter = 0;
      globalThis.crypto.randomUUID = () => `mock-uuid-${++idCounter}`;
    }
  });

  afterEach(() => {
    localStorage.clear();
  });

  it("TEST 1: Empty or whitespace message cannot be submitted", async () => {
    const streamSpy = vi.spyOn(chatApi, "streamChatResponse");
    const hook = renderUseChatHook();

    await act(async () => {
      await hook.current.sendMessage("   ");
    });

    expect(hook.current.messages).toEqual([]);
    expect(hook.current.isStreaming).toBe(false);
    expect(streamSpy).not.toHaveBeenCalled();

    hook.unmount();
  });

  it("TEST 2 & 3: User message appears immediately and exactly ONE assistant message is created", async () => {
    vi.spyOn(chatApi, "streamChatResponse").mockImplementation(() => new Promise(() => {}));
    const hook = renderUseChatHook();

    await act(async () => {
      hook.current.sendMessage("Hello Shoaib");
    });

    expect(hook.current.messages.length).toBe(2);
    expect(hook.current.messages[0]).toEqual({
      id: expect.any(String),
      role: "user",
      content: "Hello Shoaib",
    });
    expect(hook.current.messages[1]).toEqual({
      id: expect.any(String),
      role: "assistant",
      content: "",
    });
    expect(hook.current.isStreaming).toBe(true);

    hook.unmount();
  });

  it("TEST 4: Multiple token events append to the SAME assistant message", async () => {
    vi.spyOn(chatApi, "streamChatResponse").mockImplementation(
      async (payload, onEvent) => {
        onEvent({ type: "token", content: "Hello" });
        onEvent({ type: "token", content: " Shoaib" });
        onEvent({ type: "token", content: "!" });
      }
    );

    const hook = renderUseChatHook();

    await act(async () => {
      await hook.current.sendMessage("Hi");
    });

    expect(hook.current.messages.length).toBe(2);
    expect(hook.current.messages[0].role).toBe("user");
    expect(hook.current.messages[1].role).toBe("assistant");
    expect(hook.current.messages[1].content).toBe("Hello Shoaib!");

    hook.unmount();
  });

  it("TEST 5 & 6: DONE event completes streaming and stores conversation_id in localStorage", async () => {
    const mockConvId = "test-conv-123";
    vi.spyOn(chatApi, "streamChatResponse").mockImplementation(
      async (payload, onEvent) => {
        onEvent({ type: "token", content: "Response" });
        onEvent({ type: "done", conversation_id: mockConvId });
      }
    );

    const hook = renderUseChatHook();

    await act(async () => {
      await hook.current.sendMessage("Hi");
    });

    expect(hook.current.isStreaming).toBe(false);
    expect(hook.current.conversationId).toBe(mockConvId);
    expect(localStorage.getItem(LOCAL_STORAGE_KEY)).toBe(mockConvId);

    hook.unmount();
  });

  it("TEST 7: Existing conversation_id is sent with subsequent request", async () => {
    localStorage.setItem(LOCAL_STORAGE_KEY, "existing-conv-456");
    const streamSpy = vi.spyOn(chatApi, "streamChatResponse").mockImplementation(async () => {});

    const hook = renderUseChatHook();

    expect(hook.current.conversationId).toBe("existing-conv-456");

    await act(async () => {
      await hook.current.sendMessage("Second message");
    });

    expect(streamSpy).toHaveBeenCalledWith(
      {
        message: "Second message",
        conversation_id: "existing-conv-456",
      },
      expect.any(Function),
      expect.any(Function),
      expect.any(Object)
    );

    hook.unmount();
  });

  it("TEST 8, 9, 10 & 11: New Chat clears messages, conversationId, localStorage, and aborts stream", async () => {
    let capturedSignal;
    vi.spyOn(chatApi, "streamChatResponse").mockImplementation(
      async (payload, onEvent, onError, signal) => {
        capturedSignal = signal;
        return new Promise(() => {}); // unresolved hanging stream
      }
    );

    const hook = renderUseChatHook();

    await act(async () => {
      hook.current.sendMessage("Message before new chat");
    });

    expect(hook.current.messages.length).toBe(2);
    expect(hook.current.isStreaming).toBe(true);
    expect(capturedSignal.aborted).toBe(false);

    await act(async () => {
      hook.current.startNewChat();
    });

    expect(hook.current.messages).toEqual([]);
    expect(hook.current.conversationId).toBeNull();
    expect(hook.current.isStreaming).toBe(false);
    expect(hook.current.error).toBeNull();
    expect(localStorage.getItem(LOCAL_STORAGE_KEY)).toBeNull();
    expect(capturedSignal.aborted).toBe(true);

    hook.unmount();
  });

  it("TEST 12: AbortError does not show a user-facing error", async () => {
    vi.spyOn(chatApi, "streamChatResponse").mockImplementation(async () => {
      const err = new Error("The user aborted a request.");
      err.name = "AbortError";
      throw err;
    });

    const hook = renderUseChatHook();

    await act(async () => {
      await hook.current.sendMessage("Abort test");
    });

    expect(hook.current.isStreaming).toBe(false);
    expect(hook.current.error).toBeNull();

    hook.unmount();
  });

  it("TEST 13: SSE error event stops streaming and shows safe error", async () => {
    vi.spyOn(chatApi, "streamChatResponse").mockImplementation(
      async (payload, onEvent) => {
        onEvent({ type: "error", message: "Provider rate limit exceeded." });
      }
    );

    const hook = renderUseChatHook();

    await act(async () => {
      await hook.current.sendMessage("Rate limit test");
    });

    expect(hook.current.isStreaming).toBe(false);
    expect(hook.current.error).toBe("Provider rate limit exceeded.");

    hook.unmount();
  });

  it("TEST 14: Network failure shows safe error", async () => {
    vi.spyOn(chatApi, "streamChatResponse").mockImplementation(
      async (payload, onEvent, onError) => {
        onError("Unable to connect to Second Brain AI server.");
      }
    );

    const hook = renderUseChatHook();

    await act(async () => {
      await hook.current.sendMessage("Network error test");
    });

    expect(hook.current.isStreaming).toBe(false);
    expect(hook.current.error).toBe("Unable to connect to Second Brain AI server.");

    hook.unmount();
  });

  it("TEST 15: Late events from an aborted/old request cannot modify a new chat", async () => {
    let savedOnEvent;
    vi.spyOn(chatApi, "streamChatResponse").mockImplementation(
      async (payload, onEvent) => {
        savedOnEvent = onEvent;
      }
    );

    const hook = renderUseChatHook();

    await act(async () => {
      hook.current.sendMessage("Old Request");
    });

    // User starts a new chat mid-stream
    await act(async () => {
      hook.current.startNewChat();
    });

    expect(hook.current.messages).toEqual([]);

    // Late event arrives from the old request
    await act(async () => {
      if (savedOnEvent) {
        savedOnEvent({ type: "token", content: "LATE TOKEN FROM OLD STREAM" });
      }
    });

    // Verify late event was completely ignored and new chat state is untainted
    expect(hook.current.messages).toEqual([]);
    expect(hook.current.isStreaming).toBe(false);

    hook.unmount();
  });
});
