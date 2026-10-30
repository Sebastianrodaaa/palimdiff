"""palimdiff: evidence-delta in-place memory editing for long-horizon agents."""

__version__ = "0.1.0"

from palimdiff.agent_loop import MemoryAgentConfig, run_agent_loop
from palimdiff.canvas import MemoryCanvas

__all__ = ["MemoryCanvas", "MemoryAgentConfig", "run_agent_loop", "__version__"]
