"""
Event Bus
=========

Lightweight publish/subscribe event bus for decoupled
inter-component communication within the Hermes Agent.

Components can emit events (e.g. "task.completed", "circuit.opened")
and other components can subscribe to react without direct coupling.
"""

import logging
import threading
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class Event:
    """A single event on the bus."""
    topic: str
    data: Any = None
    timestamp: float = field(default_factory=time.time)
    source: str = ""


# Type alias for subscriber callbacks
Subscriber = Callable[[Event], None]


class EventBus:
    """
    Thread-safe pub/sub event bus.

    Usage:
        bus = EventBus()
        bus.subscribe("task.completed", lambda e: print(e.data))
        bus.publish("task.completed", {"latency_ms": 42})
    """

    def __init__(self, max_history: int = 500):
        self._subscribers: Dict[str, List[Subscriber]] = defaultdict(list)
        self._wildcard_subscribers: List[Subscriber] = []
        self._history: List[Event] = []
        self._max_history = max_history
        self._lock = threading.RLock()
        self._event_count = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def subscribe(self, topic: str, callback: Subscriber) -> Callable:
        """
        Subscribe to a topic. Returns an unsubscribe callable.

        Use topic="*" to receive all events.
        """
        with self._lock:
            if topic == "*":
                self._wildcard_subscribers.append(callback)
            else:
                self._subscribers[topic].append(callback)

        def _unsubscribe():
            with self._lock:
                if topic == "*":
                    self._wildcard_subscribers.remove(callback)
                else:
                    try:
                        self._subscribers[topic].remove(callback)
                    except ValueError:
                        pass

        return _unsubscribe

    def publish(self, topic: str, data: Any = None, source: str = ""):
        """Publish an event to all subscribers of the topic."""
        event = Event(topic=topic, data=data, source=source)

        with self._lock:
            self._event_count += 1
            self._history.append(event)
            if len(self._history) > self._max_history:
                self._history = self._history[-self._max_history:]

            # Exact-match subscribers
            subs = list(self._subscribers.get(topic, []))
            # Wildcard subscribers
            wilds = list(self._wildcard_subscribers)

        # Dispatch outside lock to avoid deadlocks
        for cb in subs + wilds:
            try:
                cb(event)
            except Exception as exc:
                logger.error(f"Event subscriber error on '{topic}': {exc}")

    def publish_async(self, topic: str, data: Any = None, source: str = ""):
        """Fire-and-forget publish in a daemon thread."""
        t = threading.Thread(
            target=self.publish, args=(topic, data, source), daemon=True
        )
        t.start()

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    def get_recent(self, topic: Optional[str] = None, limit: int = 20) -> List[Event]:
        """Return recent events, optionally filtered by topic."""
        with self._lock:
            events = self._history if topic is None else [
                e for e in self._history if e.topic == topic
            ]
            return list(events[-limit:])

    def get_stats(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "total_events": self._event_count,
                "topics": list(self._subscribers.keys()),
                "subscriber_counts": {
                    t: len(s) for t, s in self._subscribers.items()
                },
                "wildcard_subscribers": len(self._wildcard_subscribers),
                "history_size": len(self._history),
            }

    def clear(self):
        """Remove all subscribers and history."""
        with self._lock:
            self._subscribers.clear()
            self._wildcard_subscribers.clear()
            self._history.clear()


# Module-level singleton for convenience
_default_bus: Optional[EventBus] = None


def get_event_bus() -> EventBus:
    """Get or create the default EventBus singleton."""
    global _default_bus
    if _default_bus is None:
        _default_bus = EventBus()
    return _default_bus
