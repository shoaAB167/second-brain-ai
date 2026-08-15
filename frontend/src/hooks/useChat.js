import { useCallback, useEffect, useRef, useState } from "react";
import { streamChatResponse } from "../services/chatApi";

const LOCAL_STORAGE_KEY = "second_brain_conversation_id";

export function useChat() {
  const [messages, setMessages] = useState([]);
  const [conversationId, setConversationId] = useState(null);
  const [isStreaming, setIsStreaming] = useState(false);
  const [error, setError] = useState(null);

  const abortControllerRef = useRef(null);
  const requestIdRef = useRef(0);

  // Restore conversation_id from localStorage on mount
  useEffect(() => {
    try {
      const savedId = localStorage.getItem(LOCAL_STORAGE_KEY);
      if (savedId) {
        setConversationId(savedId);
      }
    } catch {
      // Ignore storage errors safely in restricted environments
    }
  }, []);

  const startNewChat = useCallback(() => {
    // Invalidate any ongoing or late request callbacks
    requestIdRef.current++;

    if (abortControllerRef.current) {
      abortControllerRef.current.abort();
      abortControllerRef.current = null;
    }

    setMessages([]);
    setConversationId(null);
    setIsStreaming(false);
    setError(null);
    try {
      localStorage.removeItem(LOCAL_STORAGE_KEY);
    } catch {
      // Ignore storage errors safely
    }
  }, []);

  const sendMessage = useCallback(
    async (text) => {
      const trimmed = text ? text.trim() : "";
      if (!trimmed || isStreaming) return;

      // Invalidate previous requests and track generation for current request
      const currentRequestId = ++requestIdRef.current;

      if (abortControllerRef.current) {
        abortControllerRef.current.abort();
      }

      const controller = new AbortController();
      abortControllerRef.current = controller;

      setError(null);
      setIsStreaming(true);

      const userMsgId = crypto.randomUUID();
      const assistantMsgId = crypto.randomUUID();

      const userMessage = {
        id: userMsgId,
        role: "user",
        content: trimmed,
      };

      const assistantMessage = {
        id: assistantMsgId,
        role: "assistant",
        content: "",
      };

      setMessages((prev) => [...prev, userMessage, assistantMessage]);

      const currentConvId = conversationId;

      try {
        await streamChatResponse(
          {
            message: trimmed,
            conversation_id: currentConvId,
          },
          (event) => {
            // Ignore events if a new request or new chat was started
            if (currentRequestId !== requestIdRef.current) return;

            if (event.type === "token" && event.content) {
              setMessages((prev) =>
                prev.map((msg) =>
                  msg.id === assistantMsgId
                    ? { ...msg, content: msg.content + event.content }
                    : msg
                )
              );
            } else if (event.type === "done") {
              setIsStreaming(false);
              if (event.conversation_id) {
                setConversationId(event.conversation_id);
                try {
                  localStorage.setItem(LOCAL_STORAGE_KEY, event.conversation_id);
                } catch {
                  // Ignore storage errors safely
                }
              }
            } else if (event.type === "error") {
              setIsStreaming(false);
              setError(
                event.message || "An error occurred while streaming response."
              );
            }
          },
          (errorMessage) => {
            // Ignore errors if a new request or new chat was started
            if (currentRequestId !== requestIdRef.current) return;

            setIsStreaming(false);
            setError(errorMessage);
          },
          controller.signal
        );
      } catch (err) {
        if (currentRequestId !== requestIdRef.current) return;
        if (err.name !== "AbortError") {
          setError("An unexpected network error occurred.");
        }
      } finally {
        if (currentRequestId === requestIdRef.current) {
          setIsStreaming(false);
        }
      }
    },
    [conversationId, isStreaming]
  );

  return {
    messages,
    conversationId,
    isStreaming,
    error,
    sendMessage,
    startNewChat,
  };
}
