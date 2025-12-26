#!/usr/bin/env python3
"""
Memory Visualizer - Real-time CLI visualization of the Engram memory system.

Displays memory contents ranked by various metrics in a live-updating TUI.
Run this in a separate terminal during conversations to monitor memory state.

Usage:
    python memory_visualizer.py                    # Default view, sorted by combined score
    python memory_visualizer.py --sort importance  # Sort by importance only
    python memory_visualizer.py --sort recency     # Sort by recency
    python memory_visualizer.py --sort access      # Sort by access count
    python memory_visualizer.py --query "topic"    # Show relevance to a query
    python memory_visualizer.py --refresh 2        # Update every 2 seconds
"""

import argparse
import json
import os
import pickle
import time
import sys
import select
import tty
import termios
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field


# Custom unpickler that can handle MemoryEntry without requiring full engram import
class MemoryEntryStub:
    """Stub class for unpickling MemoryEntry objects"""
    def __init__(self):
        self.id = ""
        self.content = ""
        self.timestamp = datetime.now()
        self.importance = 0.5
        self.tags = []
        self.context = {}
        self.access_count = 0
        self.last_accessed = None
        self.related_memories = []
        self.embedding = []
        self.faiss_index = -1

    def __setstate__(self, state):
        self.__dict__.update(state)


class CustomUnpickler(pickle.Unpickler):
    """Custom unpickler that can load MemoryEntry without full engram_pkg"""
    def find_class(self, module, name):
        if name == 'MemoryEntry':
            return MemoryEntryStub
        return super().find_class(module, name)

try:
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    from rich.layout import Layout
    from rich.live import Live
    from rich.text import Text
    from rich.style import Style
    from rich.progress import SpinnerColumn, TextColumn, Progress
    from rich import box
except ImportError:
    print("❌ rich library required. Install with: pip install rich")
    sys.exit(1)


@dataclass
class MemoryDisplay:
    """Memory entry formatted for display"""
    id: str
    content: str
    importance: float
    access_count: int
    timestamp: datetime
    last_accessed: Optional[datetime]
    tags: List[str]
    age_days: float
    recency_score: float
    relevance_score: float = 0.0


class MemoryVisualizer:
    """Real-time CLI visualization of the Engram memory system"""

    def __init__(self, memory_path: str = "vector_memory", refresh_interval: float = 1.0):
        self.memory_path = Path(memory_path)
        self.metadata_file = self.memory_path / "metadata.pkl"
        self.activity_file = self.memory_path / "activity.jsonl"
        self.refresh_interval = refresh_interval
        self.console = Console()
        self.last_modified = None
        self.last_activity_modified = None
        self.memories: Dict[str, Any] = {}
        self.recent_activity: List[Dict] = []
        self.current_query: Optional[str] = None
        self.sort_mode: str = "combined"  # importance, recency, access, combined, relevance
        self.encoder = None  # Lazy-loaded for relevance scoring
        self.memory_scroll_offset: int = 0  # Scroll position for memory table
        self.max_visible_memories: int = 30  # Max memories visible at once

    def load_memories(self) -> bool:
        """Load memories from the pickle file. Returns True if data changed."""
        if not self.metadata_file.exists():
            return False

        try:
            current_mtime = self.metadata_file.stat().st_mtime
            if self.last_modified == current_mtime:
                return False  # No changes

            with open(self.metadata_file, 'rb') as f:
                metadata = CustomUnpickler(f).load()
                self.memories = metadata.get('memories', {})
                self.last_modified = current_mtime
                return True
        except Exception as e:
            self.console.print(f"[red]Error loading memories: {e}[/red]")
            return False

    def load_activity(self) -> bool:
        """Load recent activity from the log file."""
        if not self.activity_file.exists():
            return False

        try:
            current_mtime = self.activity_file.stat().st_mtime
            if self.last_activity_modified == current_mtime:
                return False

            with open(self.activity_file, 'r') as f:
                lines = f.readlines()
                self.recent_activity = []
                for line in lines[-10:]:  # Last 10 activities
                    try:
                        self.recent_activity.append(json.loads(line.strip()))
                    except json.JSONDecodeError:
                        pass
                self.recent_activity.reverse()  # Most recent first
                self.last_activity_modified = current_mtime
                return True
        except Exception:
            return False

    def _load_encoder(self):
        """Lazy-load the sentence transformer for relevance scoring"""
        if self.encoder is None:
            try:
                from sentence_transformers import SentenceTransformer
                self.encoder = SentenceTransformer('all-MiniLM-L6-v2', device='cpu')
            except ImportError:
                pass  # Will not show relevance scores

    def _calculate_relevance(self, memory, query_embedding) -> float:
        """Calculate relevance score for a memory given a query embedding"""
        if not hasattr(memory, 'embedding') or not memory.embedding:
            return 0.0
        
        import numpy as np
        
        mem_embedding = np.array(memory.embedding, dtype=np.float32)
        mem_embedding = mem_embedding / np.linalg.norm(mem_embedding)
        
        query_norm = query_embedding / np.linalg.norm(query_embedding)
        
        return float(np.dot(mem_embedding, query_norm))

    def _prepare_display_memories(self) -> List[MemoryDisplay]:
        """Prepare memories for display with calculated scores"""
        now = datetime.now()
        display_memories = []

        # Calculate query embedding if needed
        query_embedding = None
        if self.current_query and self.sort_mode == "relevance":
            self._load_encoder()
            if self.encoder:
                query_embedding = self.encoder.encode(self.current_query)

        for memory_id, memory in self.memories.items():
            age_days = (now - memory.timestamp).total_seconds() / 86400

            # Recency score (exponential decay with 30-day half-life)
            recency_score = 0.5 ** (age_days / 30)

            # Calculate relevance if we have a query
            relevance_score = 0.0
            if query_embedding is not None:
                relevance_score = self._calculate_relevance(memory, query_embedding)

            display_mem = MemoryDisplay(
                id=memory_id,
                content=memory.content,
                importance=memory.importance,
                access_count=memory.access_count,
                timestamp=memory.timestamp,
                last_accessed=memory.last_accessed,
                tags=memory.tags,
                age_days=age_days,
                recency_score=recency_score,
                relevance_score=relevance_score
            )
            display_memories.append(display_mem)

        # Sort based on mode
        if self.sort_mode == "importance":
            display_memories.sort(key=lambda m: m.importance, reverse=True)
        elif self.sort_mode == "recency":
            display_memories.sort(key=lambda m: m.timestamp, reverse=True)
        elif self.sort_mode == "access":
            display_memories.sort(key=lambda m: m.access_count, reverse=True)
        elif self.sort_mode == "combined":
            # Combined score: importance * 0.4 + recency * 0.3 + access_normalized * 0.3
            max_access = max((m.access_count for m in display_memories), default=1) or 1
            display_memories.sort(
                key=lambda m: m.importance * 0.4 + m.recency_score * 0.3 + (m.access_count / max_access) * 0.3,
                reverse=True
            )
        elif self.sort_mode == "relevance":
            display_memories.sort(key=lambda m: m.relevance_score, reverse=True)

        return display_memories

    def _format_age(self, age_days: float) -> str:
        """Format age in a human-readable way"""
        if age_days < 1:
            hours = int(age_days * 24)
            if hours < 1:
                return f"{int(age_days * 1440)}m"
            return f"{hours}h"
        elif age_days < 30:
            return f"{int(age_days)}d"
        elif age_days < 365:
            return f"{int(age_days / 30)}mo"
        else:
            return f"{int(age_days / 365)}y"

    def _get_importance_style(self, importance: float) -> str:
        """Get color style based on importance level"""
        if importance >= 0.9:
            return "bold red"
        elif importance >= 0.7:
            return "bold yellow"
        elif importance >= 0.5:
            return "green"
        elif importance >= 0.3:
            return "cyan"
        else:
            return "dim white"

    def _truncate(self, text: str, max_len: int) -> str:
        """Truncate text with ellipsis"""
        if len(text) <= max_len:
            return text
        return text[:max_len - 3] + "..."

    def create_memory_table(self) -> Table:
        """Create a rich table displaying memories"""
        display_memories = self._prepare_display_memories()
        total_memories = len(display_memories)

        # Clamp scroll offset
        max_offset = max(0, total_memories - self.max_visible_memories)
        self.memory_scroll_offset = max(0, min(self.memory_scroll_offset, max_offset))

        # Slice memories based on scroll position
        start_idx = self.memory_scroll_offset
        end_idx = start_idx + self.max_visible_memories
        visible_memories = display_memories[start_idx:end_idx]

        # Build title with scroll indicator
        if total_memories > self.max_visible_memories:
            scroll_info = f" [{start_idx + 1}-{min(end_idx, total_memories)}/{total_memories}]"
        else:
            scroll_info = f" [{total_memories}]"

        # Create table
        table = Table(
            title=f"🧠 Engram Memory Viewer{scroll_info}",
            box=box.ROUNDED,
            show_lines=True,
            title_style="bold magenta",
            header_style="bold cyan",
            expand=True
        )

        # Add columns based on sort mode
        table.add_column("#", style="dim", width=3, justify="right")
        table.add_column("ID", style="dim cyan", width=8)
        
        if self.sort_mode == "relevance":
            table.add_column("Rel", style="bold green", width=5, justify="right")
        
        table.add_column("Imp", style="yellow", width=5, justify="right")
        table.add_column("Acc", style="blue", width=4, justify="right")
        table.add_column("Age", style="magenta", width=5, justify="right")
        table.add_column("Tags", style="cyan", width=18)
        table.add_column("Content", style="white", ratio=1)

        # Add rows
        for idx, mem in enumerate(visible_memories, start_idx + 1):
            importance_style = self._get_importance_style(mem.importance)
            tags_str = self._truncate(", ".join(mem.tags) if mem.tags else "-", 18)
            content_str = self._truncate(mem.content.replace("\n", " "), 60)

            row = [
                str(idx),
                mem.id[:8],
            ]
            
            if self.sort_mode == "relevance":
                row.append(f"{mem.relevance_score:.2f}")
            
            row.extend([
                Text(f"{mem.importance:.2f}", style=importance_style),
                str(mem.access_count),
                self._format_age(mem.age_days),
                tags_str,
                content_str,
            ])

            table.add_row(*row)

        return table

    def create_stats_panel(self) -> Panel:
        """Create a panel with memory statistics"""
        if not self.memories:
            return Panel("No memories loaded", title="📊 Statistics")

        total = len(self.memories)
        avg_importance = sum(m.importance for m in self.memories.values()) / total
        total_access = sum(m.access_count for m in self.memories.values())
        avg_access = total_access / total

        # Age distribution
        now = datetime.now()
        ages = [(now - m.timestamp).days for m in self.memories.values()]
        avg_age = sum(ages) / len(ages) if ages else 0

        # Tag stats
        all_tags = []
        for m in self.memories.values():
            all_tags.extend(m.tags)
        unique_tags = len(set(all_tags))

        stats_text = Text()
        stats_text.append(f"📦 Total: ", style="dim")
        stats_text.append(f"{total}", style="bold cyan")
        stats_text.append(f"  │  ", style="dim")
        stats_text.append(f"⭐ Avg Imp: ", style="dim")
        stats_text.append(f"{avg_importance:.2f}", style="bold yellow")
        stats_text.append(f"  │  ", style="dim")
        stats_text.append(f"👁 Accesses: ", style="dim")
        stats_text.append(f"{total_access}", style="bold blue")
        stats_text.append(f"  │  ", style="dim")
        stats_text.append(f"📅 Avg Age: ", style="dim")
        stats_text.append(f"{avg_age:.0f}d", style="bold magenta")
        stats_text.append(f"  │  ", style="dim")
        stats_text.append(f"🏷 Tags: ", style="dim")
        stats_text.append(f"{unique_tags}", style="bold cyan")

        return Panel(stats_text, box=box.ROUNDED)

    def create_header(self) -> Panel:
        """Create the header panel"""
        now = datetime.now().strftime("%H:%M:%S")
        
        header_text = Text()
        header_text.append("🧠 ENGRAM MEMORY VISUALIZER", style="bold magenta")
        header_text.append(f"  │  ", style="dim")
        header_text.append(f"Sort: ", style="dim")
        header_text.append(f"{self.sort_mode.upper()}", style="bold green")
        
        if self.current_query:
            header_text.append(f"  │  ", style="dim")
            header_text.append(f"Query: ", style="dim")
            header_text.append(f'"{self.current_query}"', style="bold yellow")
        
        header_text.append(f"  │  ", style="dim")
        header_text.append(f"Updated: ", style="dim")
        header_text.append(now, style="cyan")
        
        return Panel(header_text, box=box.DOUBLE, style="blue")

    def create_activity_panel(self) -> Panel:
        """Create the activity log panel (compact version)"""
        if not self.recent_activity:
            return Panel("[dim]No recent activity[/dim]", title="🔄 MemMan Activity", box=box.ROUNDED)

        activity_text = Text()
        for i, entry in enumerate(self.recent_activity[:3]):  # Show last 3 (compact)
            action = entry.get("action", "unknown")
            
            if action == "memory_stored":
                content = entry.get("content", "")[:50]
                activity_text.append(f"💾 ", style="green")
                activity_text.append(f"{content}...", style="white")
            elif action == "memory_updated":
                content = entry.get("content", "")[:50]
                activity_text.append(f"🔄 ", style="yellow")
                activity_text.append(f"{content}...", style="white")
            elif action == "extraction_skipped":
                activity_text.append(f"⏭ skipped", style="dim")
            elif action == "store_error":
                activity_text.append(f"❌ error", style="red")
            else:
                activity_text.append(f"• {action}", style="dim")
            
            if i < 2:  # Add separator between entries
                activity_text.append("  │  ", style="dim")

        return Panel(activity_text, title="🔄 MemMan Activity", box=box.ROUNDED)

    def create_controls_panel(self) -> Panel:
        """Create the controls/help panel"""
        controls = Text()
        controls.append("q", style="bold red")
        controls.append("=quit ", style="dim")
        controls.append("↑/↓", style="bold cyan")
        controls.append("=scroll ", style="dim")
        controls.append("+/-", style="bold green")
        controls.append(f"=rows({self.max_visible_memories}) ", style="dim")
        controls.append("1", style="bold yellow")
        controls.append("=comb ", style="dim")
        controls.append("2", style="bold yellow")
        controls.append("=imp ", style="dim")
        controls.append("3", style="bold yellow")
        controls.append("=rec ", style="dim")
        controls.append("4", style="bold yellow")
        controls.append("=acc ", style="dim")
        controls.append("5", style="bold yellow")
        controls.append("=rel", style="dim")

        return Panel(controls, box=box.ROUNDED, style="dim")

    def create_display(self) -> Layout:
        """Create the full display layout"""
        layout = Layout()
        
        layout.split(
            Layout(name="header", size=3),
            Layout(name="stats", size=3),
            Layout(name="main", ratio=1),
            Layout(name="activity", size=3),
            Layout(name="controls", size=3),
        )

        layout["header"].update(self.create_header())
        layout["stats"].update(self.create_stats_panel())
        layout["main"].update(self.create_memory_table())
        layout["activity"].update(self.create_activity_panel())
        layout["controls"].update(self.create_controls_panel())

        return layout

    def _get_key_nonblocking(self) -> Optional[str]:
        """Get a keypress without blocking, returns None if no key pressed.
        Handles escape sequences for arrow keys."""
        try:
            if select.select([sys.stdin], [], [], 0)[0]:
                # Read all available input at once (up to 8 bytes for escape sequences)
                data = os.read(sys.stdin.fileno(), 8)
                if not data:
                    return None
                
                # Check for escape sequence (arrow keys: ESC [ A/B)
                if data == b'\x1b[A':
                    return 'UP'
                elif data == b'\x1b[B':
                    return 'DOWN'
                elif data.startswith(b'\x1b'):
                    return None  # Ignore other escape sequences
                
                # Return single character
                return data.decode('utf-8', errors='ignore')[0] if data else None
        except Exception:
            pass
        return None

    def run_live(self):
        """Run the live visualization with keyboard controls"""
        self.console.clear()

        # Initial load
        self.load_memories()
        self.load_activity()

        # Check if we have a proper TTY for keyboard input
        is_tty = sys.stdin.isatty()
        old_settings = None
        
        if is_tty:
            try:
                old_settings = termios.tcgetattr(sys.stdin)
                tty.setcbreak(sys.stdin.fileno())
            except Exception:
                is_tty = False  # Fall back to no keyboard control
        
        try:
            with Live(self.create_display(), console=self.console, refresh_per_second=4) as live:
                running = True
                last_update = 0
                poll_interval = 0.05  # Check keyboard every 50ms
                
                while running:
                    # Check for keyboard input (only if TTY)
                    if is_tty:
                        key = self._get_key_nonblocking()
                        if key:
                            if key.lower() == 'q':
                                running = False
                                continue
                            elif key == '1':
                                self.sort_mode = "combined"
                                self.memory_scroll_offset = 0  # Reset scroll on sort change
                            elif key == '2':
                                self.sort_mode = "importance"
                                self.memory_scroll_offset = 0
                            elif key == '3':
                                self.sort_mode = "recency"
                                self.memory_scroll_offset = 0
                            elif key == '4':
                                self.sort_mode = "access"
                                self.memory_scroll_offset = 0
                            elif key == '5':
                                self.sort_mode = "relevance"
                                self.memory_scroll_offset = 0
                            elif key == 'UP':
                                self.memory_scroll_offset = max(0, self.memory_scroll_offset - 1)
                            elif key == 'DOWN':
                                self.memory_scroll_offset += 1  # Will be clamped in create_memory_table
                            elif key in ('+', '='):  # = is unshifted + on most keyboards
                                self.max_visible_memories = min(100, self.max_visible_memories + 5)
                            elif key in ('-', '_'):
                                self.max_visible_memories = max(5, self.max_visible_memories - 5)
                            # Force immediate display update on key press
                            live.update(self.create_display())
                    
                    # Check for updated data and refresh display at configured interval
                    current_time = time.time()
                    if current_time - last_update >= self.refresh_interval:
                        self.load_memories()
                        self.load_activity()
                        live.update(self.create_display())
                        last_update = current_time
                    
                    time.sleep(poll_interval)

        except KeyboardInterrupt:
            pass
        finally:
            # Restore terminal settings if we changed them
            if old_settings is not None:
                try:
                    termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)
                except Exception:
                    pass
            self.console.print("\n[yellow]Visualization stopped.[/yellow]")

    def run_once(self):
        """Run a single snapshot display"""
        self.load_memories()
        self.console.print(self.create_display())


def main(args: argparse.Namespace):
    """Main entry point"""
    # Resolve memory path
    memory_path = args.memory_path
    if not Path(memory_path).is_absolute():
        # Try relative to script location
        script_dir = Path(__file__).parent
        memory_path = script_dir / memory_path

    visualizer = MemoryVisualizer(
        memory_path=str(memory_path),
        refresh_interval=args.refresh
    )
    
    visualizer.sort_mode = args.sort
    visualizer.current_query = args.query
    visualizer.max_visible_memories = args.max_rows

    if args.once:
        visualizer.run_once()
    else:
        visualizer.run_live()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="🧠 Real-time CLI visualization of the Engram memory system",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python memory_visualizer.py                      # Default view (combined ranking), live updates
  python memory_visualizer.py --sort importance    # Sort by importance only
  python memory_visualizer.py --sort recency       # Sort by most recent
  python memory_visualizer.py --sort access        # Sort by access count
  python memory_visualizer.py --query "python"     # Show relevance to "python"
  python memory_visualizer.py --refresh 2          # Update every 2 seconds
  python memory_visualizer.py --once               # Single snapshot, no live updates
        """
    )
    
    parser.add_argument("--memory-path", default="vector_memory", help="Path to vector memory storage (default: vector_memory)")
    parser.add_argument("--sort", choices=["importance", "recency", "access", "combined", "relevance"], default="combined", help="Sort mode (default: combined)")
    parser.add_argument("--query", type=str, default=None, help="Query string for relevance ranking")
    parser.add_argument("--refresh", type=float, default=1.0, help="Refresh interval in seconds (default: 1.0)")
    parser.add_argument("--max-rows", type=int, default=30, help="Max memories to display (default: 30, use +/- to adjust live)")
    parser.add_argument("--once", action="store_true", help="Run once instead of live updates")
    
    main(parser.parse_args())

