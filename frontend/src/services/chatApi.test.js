import { describe, expect, it, vi, beforeEach } from "vitest";
import { streamChatResponse } from "./chatApi.js";

describe("chatApi streamChatResponse - SSE Parser Robustness", () => {
  beforeEach(() => {
    vi.restoreAllMocks();
  });

  it("parses SSE data events split across network chunks (LF)", async () => {
    const events = [];
    const onEvent = (event) => events.push(event);
    const onError = vi.fn();

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

  it("handles CRLF (\\r\\n\\r\\n) event delimiters", async () => {
    const events = [];
    const onEvent = (event) => events.push(event);
    const onError = vi.fn();

    const payload = 'data: {"type":"token","content":"CRLF Test"}\r\n\r\ndata: {"type":"done"}\r\n\r\n';
    const chunk = new TextEncoder().encode(payload);

    const mockStream = new ReadableStream({
      start(controller) {
        controller.enqueue(chunk);
        controller.close();
      },
    });

    global.fetch = vi.fn().mockResolvedValue({
      ok: true,
      body: mockStream,
    });

    await streamChatResponse({ message: "CRLF" }, onEvent, onError);

    expect(events.length).toBe(2);
    expect(events[0]).toEqual({ type: "token", content: "CRLF Test" });
    expect(events[1]).toEqual({ type: "done" });
    expect(onError).not.toHaveBeenCalled();
  });

  it("handles multiple events in a single network chunk", async () => {
    const events = [];
    const onEvent = (event) => events.push(event);
    const onError = vi.fn();

    const payload =
      'data: {"type":"token","content":"One"}\n\ndata: {"type":"token","content":" Two"}\n\ndata: {"type":"token","content":" Three"}\n\ndata: {"type":"done"}\n\n';
    const chunk = new TextEncoder().encode(payload);

    const mockStream = new ReadableStream({
      start(controller) {
        controller.enqueue(chunk);
        controller.close();
      },
    });

    global.fetch = vi.fn().mockResolvedValue({
      ok: true,
      body: mockStream,
    });

    await streamChatResponse({ message: "Multi" }, onEvent, onError);

    expect(events.length).toBe(4);
    expect(events[0].content).toBe("One");
    expect(events[1].content).toBe(" Two");
    expect(events[2].content).toBe(" Three");
    expect(events[3].type).toBe("done");
  });

  it("handles UTF-8 multi-byte characters split across network chunks", async () => {
    const events = [];
    const onEvent = (event) => events.push(event);
    const onError = vi.fn();

    // 🚀 emoji is 4 bytes in UTF-8: 0xF0 0x9F 0x9A 0x80
    // We construct SSE payload: 'data: {"type":"token","content":"🚀"}\n\n'
    const fullText = 'data: {"type":"token","content":"🚀"}\n\n';
    const fullBytes = new TextEncoder().encode(fullText);

    // Split the bytes right in the middle of the 4-byte 🚀 emoji sequence
    const splitIndex = fullText.indexOf("🚀") + 2; // Split 2 bytes into emoji
    const chunk1 = fullBytes.subarray(0, splitIndex);
    const chunk2 = fullBytes.subarray(splitIndex);

    let step = 0;
    const mockStream = new ReadableStream({
      pull(controller) {
        if (step === 0) {
          controller.enqueue(chunk1);
          step++;
        } else if (step === 1) {
          controller.enqueue(chunk2);
          step++;
        } else {
          controller.close();
        }
      },
    });

    global.fetch = vi.fn().mockResolvedValue({
      ok: true,
      body: mockStream,
    });

    await streamChatResponse({ message: "Emoji" }, onEvent, onError);

    expect(events.length).toBe(1);
    expect(events[0]).toEqual({ type: "token", content: "🚀" });
    expect(onError).not.toHaveBeenCalled();
  });

  it("handles malformed SSE JSON safely by calling onError without throwing", async () => {
    const events = [];
    const onEvent = (event) => events.push(event);
    const onError = vi.fn();

    const malformedPayload = 'data: {"type":"token", content: INVALID_JSON}\n\n';
    const chunk = new TextEncoder().encode(malformedPayload);

    const mockStream = new ReadableStream({
      start(controller) {
        controller.enqueue(chunk);
        controller.close();
      },
    });

    global.fetch = vi.fn().mockResolvedValue({
      ok: true,
      body: mockStream,
    });

    await streamChatResponse({ message: "Malformed" }, onEvent, onError);

    expect(events.length).toBe(0);
    expect(onError).toHaveBeenCalledWith(
      "Unable to process the response stream. Please try again."
    );
  });
});
