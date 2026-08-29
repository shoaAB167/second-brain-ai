from typing import List, Optional

from personal_ai.application.memory.retrieval_service import MemorySearchResult


class MemoryContextBuilder:
    """Formats retrieved long-term Experience memories into structured prompt context for LLM augmentation."""

    HEADER_INSTRUCTION = (
        "These are potentially relevant memories previously recorded about the user. "
        "Use them when relevant. Do not assume a memory is relevant just because it is present."
    )

    def build_context(self, memories: List[MemorySearchResult]) -> Optional[str]:
        """Convert a ranked list of MemorySearchResult objects into a structured XML-tagged prompt context.

        Args:
            memories: List of retrieved MemorySearchResult entities ranked by semantic similarity.

        Returns:
            Optional[str]: Formatted <user_memory> context block, or None if memories is empty.
        """
        if not memories:
            return None

        entries: List[str] = []
        for idx, mem in enumerate(memories, start=1):
            lines: List[str] = [f"{idx}."]
            if mem.type:
                lines.append(f"Type: {mem.type}")
            if mem.domain:
                lines.append(f"Domain: {mem.domain}")
            lines.append(f"Content: {mem.content.strip()}")
            entries.append("\n".join(lines))

        formatted_body = "\n\n".join(entries)

        return f"<user_memory>\n{self.HEADER_INSTRUCTION}\n\n{formatted_body}\n</user_memory>"
