import React from "react";

/**
 * Safe, zero-dependency Markdown parser that converts LLM text into semantic React elements.
 * Handles code blocks, inline code, bold, italics, bullet points, headers, and links safely.
 */
export function renderMarkdown(content) {
  if (!content) return null;

  // Split by code blocks first
  const codeBlockRegex = /```([a-zA-Z0-9_-]*)\n([\s\S]*?)```/g;
  const parts = [];
  let lastIndex = 0;
  let match;

  while ((match = codeBlockRegex.exec(content)) !== null) {
    const textBefore = content.substring(lastIndex, match.index);
    if (textBefore) {
      parts.push({ type: "text", content: textBefore });
    }

    parts.push({
      type: "codeblock",
      language: match[1] || "text",
      code: match[2].replace(/\n$/, ""),
    });

    lastIndex = match.index + match[0].length;
  }

  const remainingText = content.substring(lastIndex);
  if (remainingText) {
    parts.push({ type: "text", content: remainingText });
  }

  return (
    <div className="markdown-body">
      {parts.map((part, index) => {
        if (part.type === "codeblock") {
          return (
            <div key={index} className="code-block-container">
              {part.language && part.language !== "text" && (
                <div className="code-block-header">{part.language}</div>
              )}
              <pre className="code-block">
                <code>{part.code}</code>
              </pre>
            </div>
          );
        }

        return <TextBlock key={index} text={part.content} />;
      })}
    </div>
  );
}

function TextBlock({ text }) {
  const lines = text.split("\n");
  const elements = [];
  let currentList = null;

  lines.forEach((line, lineIdx) => {
    const trimmed = line.trim();

    // Headers (# Header)
    if (/^#{1,3}\s+/.test(trimmed)) {
      if (currentList) {
        elements.push(renderList(currentList, elements.length));
        currentList = null;
      }
      const level = trimmed.match(/^#+/)[0].length;
      const headerText = trimmed.replace(/^#+\s+/, "");
      const HeaderTag = `h${Math.min(level + 1, 4)}`;
      elements.push(
        <HeaderTag key={`h-${lineIdx}`} className="markdown-header">
          {formatInline(headerText)}
        </HeaderTag>
      );
      return;
    }

    // Unordered lists (- or *)
    if (/^[-*]\s+/.test(trimmed)) {
      const itemText = trimmed.replace(/^[-*]\s+/, "");
      if (!currentList) {
        currentList = [];
      }
      currentList.push(itemText);
      return;
    }

    // If we were building a list and line is not a list item, flush the list
    if (currentList) {
      elements.push(renderList(currentList, elements.length));
      currentList = null;
    }

    // Empty lines act as spacing
    if (!trimmed) {
      elements.push(<div key={`sp-${lineIdx}`} className="markdown-spacer" />);
      return;
    }

    // Standard paragraph line
    elements.push(
      <p key={`p-${lineIdx}`} className="markdown-paragraph">
        {formatInline(trimmed)}
      </p>
    );
  });

  if (currentList) {
    elements.push(renderList(currentList, elements.length));
  }

  return <>{elements}</>;
}

function renderList(items, key) {
  return (
    <ul key={`ul-${key}`} className="markdown-list">
      {items.map((item, idx) => (
        <li key={idx} className="markdown-list-item">
          {formatInline(item)}
        </li>
      ))}
    </ul>
  );
}

function formatInline(text) {
  // Parses `inline code`, **bold**, *italic*, and [links](url)
  const tokens = [];
  const inlineRegex = /(`[^`]+`|\*\*[^*]+\*\*|\*[^*]+\*|\[[^\]]+\]\([^)]+\))/g;
  let lastIndex = 0;
  let match;

  while ((match = inlineRegex.exec(text)) !== null) {
    if (match.index > lastIndex) {
      tokens.push(text.substring(lastIndex, match.index));
    }

    const matchedStr = match[0];
    if (matchedStr.startsWith("`") && matchedStr.endsWith("`")) {
      tokens.push(
        <code key={match.index} className="inline-code">
          {matchedStr.slice(1, -1)}
        </code>
      );
    } else if (matchedStr.startsWith("**") && matchedStr.endsWith("**")) {
      tokens.push(
        <strong key={match.index} className="markdown-bold">
          {matchedStr.slice(2, -2)}
        </strong>
      );
    } else if (matchedStr.startsWith("*") && matchedStr.endsWith("*")) {
      tokens.push(
        <em key={match.index} className="markdown-italic">
          {matchedStr.slice(1, -1)}
        </em>
      );
    } else if (matchedStr.startsWith("[") && matchedStr.includes("](")) {
      const linkMatch = matchedStr.match(/\[([^\]]+)\]\(([^)]+)\)/);
      if (linkMatch) {
        const [, linkText, linkHref] = linkMatch;
        const isSafe = /^https?:\/\//i.test(linkHref) || linkHref.startsWith("/");
        tokens.push(
          <a
            key={match.index}
            href={isSafe ? linkHref : "#"}
            target="_blank"
            rel="noopener noreferrer"
            className="markdown-link"
          >
            {linkText}
          </a>
        );
      }
    }

    lastIndex = match.index + match[0].length;
  }

  if (lastIndex < text.length) {
    tokens.push(text.substring(lastIndex));
  }

  return tokens.length > 0 ? tokens : text;
}
