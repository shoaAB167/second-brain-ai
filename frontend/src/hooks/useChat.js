import { useCallback, useEffect, useRef, useState } from "react";
import { streamChatResponse } from "../services/chatApi";

const LOCAL_STORAGE_KEY = "second_brain_conversation_id";

export function useChat() {
  const [messages, setMessages] = useState([]);
  const [conversationId, setConversationId] = useState(null);
  const [isStreaming, setIsStreaming] = useState(false);
  const [error, setError] = useState(null);

  const abortControllerRef = useRef(null);

  // Restore conversation_id from localStorage on mount
  useEffect(() => {
    const savedId = localStorage.getItem(LOCAL_STORAGE_KEY);
    if (savedId) {
      setConversationId(savedId);
    }
  }, []);

  const startNewChat = useCallback(() => {
    if (abortControllerRef.current) {
      abortControllerRef.current.abort();
      abortControllerRef.current = null;
    }

    setMessages([]);
    setConversationId(null);
    setIsStreaming(false);
    setError(null);
    localStorage.removeItem(LOCAL_STORAGE_KEY);
  }, []);

  const sendMessage = useCallback(
    async (text) => {
      const trimmed = text.trim();
      if (!trimmed || isStreaming) return;

      // Abort any ongoing request before starting a new one
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

      await streamChatResponse(
        {
          message: trimmed,
          conversation_id: currentConvId,
        },
        (event) => {
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
              localStorage.setItem(LOCAL_STORAGE_KEY, event.conversation_id);
            }
          } else if (event.type === "error") {
            setIsStreaming(false);
            setError(
              event.message || "An error occurred while streaming response."
            );
          }
        },
        (errorMessage) => {
          setIsStreaming(false);
          setError(errorMessage);
        },
        controller.signal
      );
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
