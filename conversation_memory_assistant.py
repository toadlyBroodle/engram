#!/usr/bin/env python3
"""
Conversation Memory Assistant

Real-time memory integration for live conversations in Cursor IDE.
Provides instant access to relevant memories during chat sessions.

Usage:
    python conversation_memory_assistant.py --live
    # Then paste conversation text to get memory suggestions
"""

import sys
import json
import argparse
from typing import List, Dict, Any
from memory_integration import MemoryIntegration
from bootstrap_memory import MemoryBootstrap
from datetime import datetime


class ConversationMemoryAssistant:
    """
    Live memory assistant for Cursor IDE conversations.

    Monitors conversation flow and provides real-time memory integration
    to enhance AI responses with relevant historical knowledge.
    """

    def __init__(self, memory_path: str = "vector_memory"):
        self.integration = MemoryIntegration(memory_path=memory_path)
        self.bootstrap_handler = MemoryBootstrap(memory_path, self.integration)
        self.conversation_buffer = []
        self.session_start = datetime.now()

        print("🧠 Conversation Memory Assistant initialized")
        print("💡 Ready to enhance conversations with persistent memory")
        print("🔄 Automatic insight capture enabled")
        print("=" * 60)

    def process_conversation_turn(self, speaker: str, message: str) -> Dict[str, Any]:
        """
        Process a conversation turn and return relevant memories

        Args:
            speaker: 'user' or 'assistant'
            message: The message content

        Returns:
            Dictionary with memory integration results
        """
        # Update conversation context
        self.integration.update_conversation(speaker, message)
        self.conversation_buffer.append({
            "speaker": speaker,
            "message": message,
            "timestamp": datetime.now()
        })

        # Automatically capture important insights
        if len(message) > 20:  # Only check substantial messages
            try:
                self.bootstrap_handler.capture_important_insight(
                    text=message,
                    context=f"live_assistant_{speaker}",
                    importance=None
                )
            except Exception as e:
                # Don't let auto-capture failures break the conversation
                pass

        # Get relevant memories for this context
        memory_result = self.integration.get_context_memories()

        return {
            "message_processed": message[:100] + "..." if len(message) > 100 else message,
            "memories_found": memory_result["memory_count"],
            "tokens_used": memory_result["tokens_used"],
            "formatted_memories": memory_result["formatted_context"],
            "context_stats": memory_result["context_stats"]
        }

    def get_conversation_summary(self) -> Dict[str, Any]:
        """Get summary of current conversation with memory integration stats"""
        total_messages = len(self.conversation_buffer)
        conversation_duration = (datetime.now() - self.session_start).total_seconds()

        return {
            "session_duration_seconds": conversation_duration,
            "total_messages": total_messages,
            "messages_per_minute": (total_messages / max(conversation_duration, 1)) * 60,
            "system_status": self.integration.get_system_status(),
            "recent_conversation": self.conversation_buffer[-3:] if self.conversation_buffer else []
        }

    def suggest_memory_enhancement(self, current_topic: str = None) -> str:
        """
        Suggest how to enhance the current conversation with memories

        Args:
            current_topic: Optional specific topic to focus on

        Returns:
            Formatted suggestion with relevant memories
        """
        if current_topic:
            self.integration.update_conversation("system", f"Current topic: {current_topic}")

        memory_result = self.integration.get_context_memories()

        if memory_result["memory_count"] == 0:
            return "💭 No highly relevant memories found for current context. Consider adding some memories about this topic for future conversations."

        suggestion = f"🎯 **Memory Enhancement Suggestion**\n\n"
        suggestion += f"Found {memory_result['memory_count']} relevant memories ({memory_result['tokens_used']} tokens):\n\n"
        suggestion += memory_result["formatted_context"]
        suggestion += f"\n\n💡 **Integration Tip:** These memories could enhance your response by providing:\n"
        suggestion += f"- Historical context and lessons learned\n"
        suggestion += f"- Technical details and best practices\n"
        suggestion += f"- Related experiences and solutions\n"

        return suggestion

    def live_conversation_mode(self):
        """Interactive mode for live conversation memory assistance"""
        print("🎯 LIVE CONVERSATION MEMORY ASSISTANT")
        print("Instructions:")
        print("- Paste user messages and press Enter")
        print("- Type 'assistant: <message>' for AI responses")
        print("- Type 'suggest' for memory enhancement suggestions")
        print("- Type 'summary' for conversation stats")
        print("- Type 'quit' to exit")
        print("=" * 60)

        while True:
            try:
                user_input = input("\n> ").strip()

                if user_input.lower() in ['quit', 'exit', 'q']:
                    print("\n👋 Session ended. Conversation memories saved.")
                    break

                elif user_input.lower() == 'summary':
                    summary = self.get_conversation_summary()
                    print(f"\n📊 CONVERSATION SUMMARY:")
                    print(f"Duration: {summary['session_duration_seconds']:.1f}s")
                    print(f"Messages: {summary['total_messages']}")
                    print(f"Rate: {summary['messages_per_minute']:.1f} msg/min")
                    print(f"Memory system: {summary['system_status']['memory_system']['total_memories']} memories")

                elif user_input.lower() == 'suggest':
                    suggestion = self.suggest_memory_enhancement()
                    print(f"\n{suggestion}")

                elif user_input.startswith('assistant: '):
                    assistant_message = user_input[11:].strip()
                    result = self.process_conversation_turn("assistant", assistant_message)
                    print(f"\n🤖 Assistant message processed: {result['message_processed']}")
                    if result['memories_found'] > 0:
                        print(f"📚 {result['memories_found']} relevant memories available ({result['tokens_used']} tokens)")

                else:
                    # Treat as user message
                    result = self.process_conversation_turn("user", user_input)
                    print(f"\n👤 User message processed: {result['message_processed']}")
                    if result['memories_found'] > 0:
                        print(f"📚 {result['memories_found']} relevant memories available ({result['tokens_used']} tokens)")
                        print("\n💡 Type 'suggest' to see memory enhancement suggestions!")

            except KeyboardInterrupt:
                print("\n👋 Session interrupted. Conversation memories saved.")
                break
            except Exception as e:
                print(f"❌ Error: {e}")
                continue

    def export_conversation_memories(self, filepath: str = None) -> str:
        """Export conversation with integrated memories for analysis"""
        if not filepath:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filepath = f"conversation_memories_{timestamp}.json"

        conversation_data = {
            "session_info": {
                "start_time": self.session_start.isoformat(),
                "end_time": datetime.now().isoformat(),
                "total_messages": len(self.conversation_buffer)
            },
            "conversation": self.conversation_buffer,
            "memory_system_status": self.integration.get_system_status(),
            "integrated_memories": []
        }

        # Add memory integration data for each turn
        for turn in self.conversation_buffer:
            if turn["speaker"] in ["user", "assistant"]:
                # Recreate the context at this point
                temp_integration = MemoryIntegration()
                for prev_turn in self.conversation_buffer:
                    if prev_turn["timestamp"] <= turn["timestamp"]:
                        temp_integration.update_conversation(prev_turn["speaker"], prev_turn["message"])
                    if prev_turn == turn:
                        break

                memory_result = temp_integration.get_context_memories()
                conversation_data["integrated_memories"].append({
                    "turn": turn,
                    "memories": memory_result
                })

        with open(filepath, 'w') as f:
            json.dump(conversation_data, f, indent=2, default=str)

        return filepath


def create_memory_assistant(memory_path: str = "vector_memory") -> ConversationMemoryAssistant:
    """Factory function to create conversation memory assistant"""
    return ConversationMemoryAssistant(memory_path=memory_path)


def main():
    parser = argparse.ArgumentParser(description="Conversation Memory Assistant")
    parser.add_argument("--memory-path", default="vector_memory",
                       help="Path to memory storage")
    parser.add_argument("--live", action="store_true",
                       help="Start live conversation mode")
    parser.add_argument("--suggest", type=str,
                       help="Get memory suggestions for a topic")
    parser.add_argument("--export", type=str,
                       help="Export conversation memories to file")

    args = parser.parse_args()

    assistant = create_memory_assistant(args.memory_path)

    if args.live:
        assistant.live_conversation_mode()
    elif args.suggest:
        suggestion = assistant.suggest_memory_enhancement(args.suggest)
        print(suggestion)
    elif args.export:
        filepath = assistant.export_conversation_memories(args.export)
        print(f"✅ Conversation memories exported to: {filepath}")
    else:
        print("🧠 Conversation Memory Assistant")
        print("Use --live for interactive mode, --suggest for topic suggestions")
        print("Example: python conversation_memory_assistant.py --live")


if __name__ == "__main__":
    main()
