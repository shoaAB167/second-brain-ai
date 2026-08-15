import { describe, expect, it, vi } from "vitest";
import { streamChatResponse } from "./chatApi.js";

describe("chatApi streamChatResponse", () => {
  it("parses SSE data events split across network chunks correctly", async () => {
    const events = [];
    const onEvent = (event) => events.push(event);
    const onError = vi.fn();

    // Mock readable stream yielding split network chunks
    const chunk1 = new TextEncoder().encode('data: {"type":"token","content":"Hel');
    const chunk2 = new TextEncoder().encode('lo"}\n\ndata: {"type":"token","content":" world"}\n\ndata: {"type":"done"}\n\n');

    let streamStep = 0;
    const mockStream = new ReadableStream({
      pull(controller) {
        if (streamStep === 0) {
          controller.enqueue(chunk1);
          streamStep++;
        } else if (streamStep === 1) {
          controller.enqueue(chunk2);
          streamStep++;
        } else {
          controller.close();
        }
      },
    });

    global.fetch = vi.fn().mockResolvedValue({
      ok: true,
      body: mockStream,
    });

    await streamChatResponse({ message: "Test" }, onEvent, onError);

    expect(events.length).toBe(3);
    expect(events[0]).toEqual({ type: "token", content: "Hello" });
    expect(events[1]).toEqual({ type: "token", content: " world" });
    expect(events[2]).toEqual({ type: "done" });
    expect(onError).not.toHaveBeenCalled();
  });
});
