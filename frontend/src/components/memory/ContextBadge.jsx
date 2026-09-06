import React, { useState } from "react";

/**
 * Subtle Personal Context indicator for assistant responses.
 * Highlights that the response was informed by long-term memory without exposing
 * creepy internal data, raw IDs, or vector similarity scores.
 */
export function ContextBadge({ personalContext }) {
  const [isOpen, setIsOpen] = useState(false);

  if (!personalContext || !personalContext.memory_count) {
    return null;
  }

  const { memory_count, topics = [] } = personalContext;
  const memoryText =
    memory_count === 1 ? "1 memory" : `${memory_count} memories`;

  return (
    <div className="context-badge-container">
      <button
        type="button"
        className="context-badge-btn"
        onClick={() => setIsOpen(!isOpen)}
        aria-expanded={isOpen}
        aria-label={`Personal context retrieved using ${memoryText}`}
        title="View context grounding"
      >
        <span className="context-sparkle-icon" aria-hidden="true">✨</span>
        <span className="context-badge-text">Using {memoryText}</span>
        {topics.length > 0 && (
          <span className="context-topics-preview">
            ({topics.slice(0, 2).join(", ")})
          </span>
        )}
      </button>

      {isOpen && (
        <div className="context-popover" role="tooltip">
          <div className="context-popover-header">
            <strong>Personal Context Grounding</strong>
          </div>
          <p className="context-popover-desc">
            Second Brain referenced {memoryText} from your past experiences to personalize this response.
          </p>
          {topics.length > 0 && (
            <div className="context-topics-list">
              <span className="context-topics-label">Relevant topics:</span>
              <div className="context-tags">
                {topics.map((topic, i) => (
                  <span key={i} className="context-tag">
                    {topic}
                  </span>
                ))}
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );
}
