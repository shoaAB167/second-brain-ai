import { useCallback, useEffect, useRef, useState } from "react";
import { streamChatResponse } from "../services/chatApi";
import { useAuth } from "../context/AuthContext";
import { MessageRole, StreamEventType } from "../types/chat";
import { buildCompanionSystemPrompt } from "../types/companion";

const LOCAL_STORAGE_KEY = "second_brain_conversation_id";

export function useChat() {
  const [messages, setMessages] = useState([]);
  const [conversationId, setConversationId] = useState(null);
  const [isStreaming, setIsStreaming] = useState(false);
  const [error, setError] = useState(null);
  const [lastUserPrompt, setLastUserPrompt] = useState("");

  const { token, openAuthModal } = useAuth();

  const abortControllerRef = useRef(null);
  const requestIdRef = useRef(0);

  // Restore conversation_id from localStorage on mount if authenticated
  useEffect(() => {
    if (!token) {
      setMessages([]);
      setConversationId(null);
      try {
        localStorage.removeItem(LOCAL_STORAGE_KEY);
      } catch {
        // Ignore storage errors
      }
      return;
    }

    try {
      const savedId = localStorage.getItem(LOCAL_STORAGE_KEY);
      if (savedId) {
        setConversationId(savedId);
      }
    } catch {
      // Ignore storage errors safely
    }
  }, [token]);

  const stopStreaming = useCallback(() => {
    if (abortControllerRef.current) {
      abortControllerRef.current.abort();
      abortControllerRef.current = null;
    }
    setIsStreaming(false);
  }, []);

  const startNewChat = useCallback(() => {
    requestIdRef.current++;

    if (abortControllerRef.current) {
      abortControllerRef.current.abort();
      abortControllerRef.current = null;
    }

    setMessages([]);
    setConversationId(null);
    setIsStreaming(false);
    setError(null);
    setLastUserPrompt("");
    try {
      localStorage.removeItem(LOCAL_STORAGE_KEY);
    } catch {
      // Ignore storage errors safely
    }
  }, []);

  const sendMessage = useCallback(
    async (text, customSystemPrompt = null) => {
      const trimmed = text ? text.trim() : "";
      if (!trimmed || isStreaming) return;

      if (!token) {
        openAuthModal();
        return;
      }

      const currentRequestId = ++requestIdRef.current;

      if (abortControllerRef.current) {
        abortControllerRef.current.abort();
      }

      const controller = new AbortController();
      abortControllerRef.current = controller;

      setError(null);
      setIsStreaming(true);
      setLastUserPrompt(trimmed);

      const timestamp = new Date().toLocaleTimeString([], {
        hour: "2-digit",
        minute: "2-digit",
      });

      const userMsgId = crypto.randomUUID();
      const assistantMsgId = crypto.randomUUID();

      const userMessage = {
        id: userMsgId,
        role: MessageRole.USER,
        content: trimmed,
        timestamp,
      };

      const assistantMessage = {
        id: assistantMsgId,
        role: MessageRole.ASSISTANT,
        content: "",
        timestamp,
        personalContext: null,
      };

      setMessages((prev) => [...prev, userMessage, assistantMessage]);

      const currentConvId = conversationId;
      const systemPrompt = customSystemPrompt || buildCompanionSystemPrompt();

      try {
        await streamChatResponse(
          {
            message: trimmed,
            conversation_id: currentConvId,
            system_prompt: systemPrompt,
            token,
          },
          (event) => {
            if (currentRequestId !== requestIdRef.current) return;

            if (event.type === StreamEventType.CONTEXT && event.context) {
              setMessages((prev) =>
                prev.map((msg) =>
                  msg.id === assistantMsgId
                    ? { ...msg, personalContext: event.context }
                    : msg
                )
              );
            } else if (event.type === StreamEventType.TOKEN && event.content) {
              setMessages((prev) =>
                prev.map((msg) =>
                  msg.id === assistantMsgId
                    ? { ...msg, content: msg.content + event.content }
                    : msg
                )
              );
            } else if (event.type === StreamEventType.DONE) {
              setIsStreaming(false);
              if (event.conversation_id) {
                setConversationId(event.conversation_id);
                try {
                  localStorage.setItem(LOCAL_STORAGE_KEY, event.conversation_id);
                } catch {
                  // Ignore storage errors safely
                }
              }
            } else if (event.type === StreamEventType.ERROR) {
              setIsStreaming(false);
              setError(
                event.message || "An error occurred while streaming response."
              );
            }
          },
          (errorMessage, status) => {
            if (currentRequestId !== requestIdRef.current) return;

            setIsStreaming(false);
            if (status === 401) {
              openAuthModal();
            }
            setError(errorMessage);
          },
          controller.signal
        );
      } catch (err) {
        if (currentRequestId !== requestIdRef.current) return;
        if (err.name !== "AbortError") {
          setError("An unexpected network error occurred. Please try again.");
        }
      } finally {
        if (currentRequestId === requestIdRef.current) {
          setIsStreaming(false);
        }
      }
    },
    [conversationId, isStreaming, token, openAuthModal]
  );

  const retryLastMessage = useCallback(() => {
    if (lastUserPrompt && !isStreaming) {
      setMessages((prev) => {
        if (
          prev.length > 0 &&
          prev[prev.length - 1].role === MessageRole.ASSISTANT &&
          !prev[prev.length - 1].content
        ) {
          return prev.slice(0, prev.length - 2);
        }
        return prev;
      });
      sendMessage(lastUserPrompt);
    }
  }, [lastUserPrompt, isStreaming, sendMessage]);

  return {
    messages,
    conversationId,
    isStreaming,
    error,
    lastUserPrompt,
    sendMessage,
    stopStreaming,
    startNewChat,
    retryLastMessage,
  };
}
