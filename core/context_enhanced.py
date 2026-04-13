"""
Enhanced Context Intelligence - Phase 2 UPGRADED
===============================================

Enhanced with:
- Semantic task classification
- Context-aware token allocation
- Hierarchical context management
- Conversation summarization
- Intent tracking
"""

import re
import time
import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Any, Tuple, Set
from collections import deque, Counter
import threading
import hashlib

logger = logging.getLogger(__name__)


class TaskType(Enum):
    """Task type classification."""
    CODE = "code"
    ANALYSIS = "analysis"
    CREATIVE = "creative"
    COMMAND = "command"
    CONVERSATION = "conversation"
    RESEARCH = "research"
    DEBUGGING = "debugging"
    UNKNOWN = "unknown"


class TaskComplexity(Enum):
    """Task complexity levels."""
    SIMPLE = 1
    MODERATE = 2
    COMPLEX = 3


@dataclass
class TaskProfile:
    """Complete task profile for allocation decisions."""
    task_type: TaskType = TaskType.UNKNOWN
    complexity: TaskComplexity = TaskComplexity.SIMPLE
    confidence: float = 0.0
    estimated_tokens: int = 1000
    keywords: List[str] = field(default_factory=list)
    requires_history: bool = False
    requires_files: bool = False
    urgency: int = 1
    intent: str = ""
    entities: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "task_type": self.task_type.value,
            "complexity": self.complexity.name,
            "confidence": round(self.confidence, 2),
            "estimated_tokens": self.estimated_tokens,
            "keywords": self.keywords,
            "requires_history": self.requires_history,
            "requires_files": self.requires_files,
            "urgency": self.urgency,
            "intent": self.intent,
            "entities": self.entities
        }


@dataclass
class AllocationResult:
    """Result of token allocation."""
    allocated: int
    total_available: int
    allocation_map: Dict[str, int]
    compression_needed: bool
    compression_ratio: float
    strategy: str = "standard"
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "allocated": self.allocated,
            "total_available": self.total_available,
            "allocation_map": self.allocation_map,
            "compression_needed": self.compression_needed,
            "compression_ratio": round(self.compression_ratio, 2),
            "strategy": self.strategy
        }


class EnhancedTaskClassifier:
    """
    Enhanced task classifier with semantic understanding.
    
    Features:
    - Multi-pattern matching
    - Context-aware classification
    - Intent extraction
    - Entity recognition
    - Confidence scoring
    """
    
    # Enhanced patterns with weights
    PATTERNS = {
        TaskType.CODE: {
            "patterns": [
                (r'\b(code|program|function|class|method|bug|fix|debug|error|exception|compile|build|import|from)\b', 2),
                (r'\b(write|create|implement|refactor|optimize).{0,30}(code|function|class|script)\b', 3),
                (r'\b(python|javascript|java|rust|go|typescript|sql|bash|shell|c\+\+|c#|ruby|php)\b', 2),
                (r'[{};:()=<>!&|]', 1),
                (r'`[^`]+`', 1),
                (r'```[\s\S]*?```', 3),
            ],
            "keywords": ["function", "class", "variable", "loop", "if", "else", "return"]
        },
        TaskType.DEBUGGING: {
            "patterns": [
                (r'\b(error|bug|issue|problem|fail|crash|exception|traceback|stack trace)\b', 3),
                (r'\b(debug|fix|solve|resolve|troubleshoot|diagnose)\b', 2),
                (r'\b(not working|broken|fails|throws|logs show)\b', 2),
            ],
            "keywords": ["error", "bug", "debug", "fix", "traceback"]
        },
        TaskType.ANALYSIS: {
            "patterns": [
                (r'\b(analyze|analysis|research|compare|contrast|evaluate|assess|examine|study)\b', 2),
                (r'\b(data|statistics|trend|pattern|correlation|metrics|performance|benchmark)\b', 2),
                (r'\b(summary|summarize|extract|identify|find)\s+(?:the\s+)?(?:key\s+)?(?:insights?|patterns?|themes?)\b', 2),
                (r'\b(why|how|what|when|where|who).{0,20}(difference|cause|effect|reason|impact|compare)\b', 1),
            ],
            "keywords": ["analyze", "compare", "data", "metrics", "trend"]
        },
        TaskType.RESEARCH: {
            "patterns": [
                (r'\b(research|search|find|look up|investigate|explore|learn about)\b', 2),
                (r'\b(information|details|facts|sources|references|documentation)\b', 1),
                (r'\b(what is|how does|explain|tell me about)\b', 1),
            ],
            "keywords": ["research", "find", "information", "source"]
        },
        TaskType.CREATIVE: {
            "patterns": [
                (r'\b(write|draft|create|compose|generate|design|make).{0,20}(story|essay|post|article|content|copy|text|email|letter|poem|song)\b', 3),
                (r'\b(brainstorm|ideate|imagine|creative|innovative|unique|original|inspiration)\b', 2),
                (r'\b(slogan|tagline|headline|title|name|brand|logo)\b', 2),
            ],
            "keywords": ["create", "write", "design", "brainstorm"]
        },
        TaskType.COMMAND: {
            "patterns": [
                (r'^(run|execute|start|stop|restart|kill|find|search|show|list|get|set|update|delete|create|make|build|install)\b', 2),
                (r'\b(run|execute).{0,15}(command|script|program|tool|function)\b', 2),
            ],
            "keywords": ["run", "execute", "command", "start", "stop"]
        },
        TaskType.CONVERSATION: {
            "patterns": [
                (r'^(hi|hello|hey|good morning|good afternoon|good evening|thanks?|thank you|please)\b', 2),
                (r'\b(how are you|what do you think|do you understand|can you|could you|would you)\b', 1),
                (r'\?(\s*$|\s+(?:about|regarding|concerning))', 1),
                (r'\b(clarify|explain|help|assist|support)\b', 1),
            ],
            "keywords": ["hello", "thanks", "please", "help"]
        }
    }
    
    # Complexity indicators
    COMPLEXITY_PATTERNS = {
        TaskComplexity.COMPLEX: [
            (r'\b(complex|complicated|difficult|challenging|sophisticated|comprehensive|thorough|extensive)\b', 2),
            (r'\b(multiple|many|several|various|diverse).{0,20}(steps|parts|aspects|factors|components)\b', 2),
            (r'\b(deep|detailed|in-depth|exhaustive|complete|full)\b', 1),
            (r'\b(and.{0,10}and.{0,10}and)\b', 2),
        ],
        TaskComplexity.SIMPLE: [
            (r'\b(simple|quick|easy|brief|short|concise|minimal|basic|straightforward)\b', 2),
            (r'^(just|only)\s+', 1),
            (r'\b(one|single|a couple of|few)\b', 1),
        ]
    }
    
    # Token estimates by type and complexity
    TOKEN_ESTIMATES = {
        (TaskType.CODE, TaskComplexity.SIMPLE): 2000,
        (TaskType.CODE, TaskComplexity.MODERATE): 4000,
        (TaskType.CODE, TaskComplexity.COMPLEX): 8000,
        (TaskType.DEBUGGING, TaskComplexity.SIMPLE): 2500,
        (TaskType.DEBUGGING, TaskComplexity.MODERATE): 4500,
        (TaskType.DEBUGGING, TaskComplexity.COMPLEX): 7000,
        (TaskType.ANALYSIS, TaskComplexity.SIMPLE): 1500,
        (TaskType.ANALYSIS, TaskComplexity.MODERATE): 3500,
        (TaskType.ANALYSIS, TaskComplexity.COMPLEX): 6000,
        (TaskType.RESEARCH, TaskComplexity.SIMPLE): 1500,
        (TaskType.RESEARCH, TaskComplexity.MODERATE): 3000,
        (TaskType.RESEARCH, TaskComplexity.COMPLEX): 5000,
        (TaskType.CREATIVE, TaskComplexity.SIMPLE): 1000,
        (TaskType.CREATIVE, TaskComplexity.MODERATE): 2500,
        (TaskType.CREATIVE, TaskComplexity.COMPLEX): 4500,
        (TaskType.COMMAND, TaskComplexity.SIMPLE): 500,
        (TaskType.COMMAND, TaskComplexity.MODERATE): 1500,
        (TaskType.COMMAND, TaskComplexity.COMPLEX): 3000,
        (TaskType.CONVERSATION, TaskComplexity.SIMPLE): 500,
        (TaskType.CONVERSATION, TaskComplexity.MODERATE): 1000,
        (TaskType.CONVERSATION, TaskComplexity.COMPLEX): 2000,
    }
    
    def __init__(self, min_confidence: float = 0.6):
        self.min_confidence = min_confidence
        self._compile_patterns()
        self._classification_history: deque = deque(maxlen=100)
        self._intent_history: Dict[str, int] = {}
        self._lock = threading.Lock()
    
    def _compile_patterns(self):
        """Compile regex patterns for performance."""
        import re
        self._compiled_patterns = {}
        for task_type, config in self.PATTERNS.items():
            self._compiled_patterns[task_type] = [
                (re.compile(p, re.IGNORECASE), weight)
                for p, weight in config["patterns"]
            ]
        
        self._compiled_complexity = {}
        for level, patterns in self.COMPLEXITY_PATTERNS.items():
            self._compiled_complexity[level] = [
                (re.compile(p, re.IGNORECASE), weight)
                for p, weight in patterns
            ]
    
    def classify(self, message: str, context: Optional[Dict] = None) -> TaskProfile:
        """
        Classify message with enhanced semantic understanding.
        
        Args:
            message: User message
            context: Optional conversation context
            
        Returns:
            TaskProfile with classification results
        """
        message_lower = message.lower()
        
        # Score each task type with weighted patterns
        scores = Counter()
        matched_keywords = []
        
        for task_type, patterns in self._compiled_patterns.items():
            for pattern, weight in patterns:
                matches = pattern.findall(message)
                if matches:
                    score = len(matches) * weight if isinstance(matches, list) else weight
                    scores[task_type] += score
                    if isinstance(matches, list):
                        matched_keywords.extend([str(m) for m in matches[:3]])
        
        # Determine task type with confidence
        if scores:
            top_task, top_score = scores.most_common(1)[0]
            total_score = sum(scores.values())
            confidence = top_score / total_score if total_score > 0 else 1.0
            
            # If debugging and code both score high, prefer debugging (more specific)
            if (TaskType.CODE in scores and TaskType.DEBUGGING in scores
                    and scores[TaskType.DEBUGGING] >= scores[TaskType.CODE] * 0.7):
                top_task = TaskType.DEBUGGING
                top_score = scores[TaskType.DEBUGGING]
                confidence = top_score / total_score if total_score > 0 else 1.0
            
            # Boost confidence for clear matches
            if top_score >= 5:
                confidence = min(1.0, confidence * 1.2)
            
            if confidence >= self.min_confidence:
                task_type = top_task
            else:
                task_type = TaskType.UNKNOWN
                confidence = confidence * 0.5
        else:
            task_type = TaskType.UNKNOWN
            confidence = 0.0
        
        # Determine complexity
        complexity = self._classify_complexity(message)
        
        # Estimate tokens
        estimated_tokens = self.TOKEN_ESTIMATES.get(
            (task_type, complexity),
            2000
        )
        
        # Extract intent
        intent = self._extract_intent(message, task_type)
        
        # Extract entities
        entities = self._extract_entities(message)
        
        # Check requirements
        requires_history = self._requires_history(message, task_type)
        requires_files = self._requires_files(message)
        
        # Create profile
        profile = TaskProfile(
            task_type=task_type,
            complexity=complexity,
            confidence=confidence,
            estimated_tokens=estimated_tokens,
            keywords=list(set(matched_keywords))[:5],
            requires_history=requires_history,
            requires_files=requires_files,
            urgency=self._detect_urgency(message),
            intent=intent,
            entities=entities
        )
        
        # Store in history
        with self._lock:
            self._classification_history.append(profile)
            self._intent_history[intent] = self._intent_history.get(intent, 0) + 1
        
        logger.debug(f"Classified: {task_type.value} ({complexity.name}), confidence={confidence:.2f}")
        return profile
    
    def _classify_complexity(self, message: str) -> TaskComplexity:
        """Classify message complexity with weighted scoring."""
        scores = Counter()
        
        for level, patterns in self._compiled_complexity.items():
            for pattern, weight in patterns:
                matches = pattern.findall(message)
                if matches:
                    scores[level] += len(matches) * weight if isinstance(matches, list) else weight
        
        if scores:
            return scores.most_common(1)[0][0]
        
        # Default based on message characteristics
        word_count = len(message.split())
        sentence_count = message.count('.') + message.count('!') + message.count('?') + 1
        
        if word_count < 15 and sentence_count <= 2:
            return TaskComplexity.SIMPLE
        elif word_count < 60 and sentence_count <= 5:
            return TaskComplexity.MODERATE
        else:
            return TaskComplexity.COMPLEX
    
    def _extract_intent(self, message: str, task_type: TaskType) -> str:
        """Extract primary intent from message."""
        intent_patterns = {
            "create": r'\b(create|make|build|write|generate|design)\b',
            "modify": r'\b(modify|change|update|edit|fix|improve|refactor)\b',
            "explain": r'\b(explain|describe|clarify|tell me about|what is|how does)\b',
            "compare": r'\b(compare|contrast|difference|versus|vs|better|worse)\b',
            "find": r'\b(find|search|look for|locate|get|fetch)\b',
            "debug": r'\b(debug|fix|solve|resolve|troubleshoot|error|bug)\b',
            "optimize": r'\b(optimize|improve|enhance|speed up|faster|better)\b',
        }
        
        for intent, pattern in intent_patterns.items():
            if re.search(pattern, message, re.IGNORECASE):
                return intent
        
        return task_type.value
    
    def _extract_entities(self, message: str) -> List[str]:
        """Extract named entities from message."""
        entities = []
        
        # File paths
        file_pattern = r'[\w\-./]+\.(py|js|ts|java|rs|go|cpp|c|h|json|yaml|yml|xml|csv|md|txt)'
        files = re.findall(file_pattern, message)
        entities.extend(files)
        
        # URLs
        url_pattern = r'https?://[^\s]+'
        urls = re.findall(url_pattern, message)
        entities.extend(urls)
        
        # Code blocks (simplified)
        code_pattern = r'`([^`]+)`'
        code = re.findall(code_pattern, message)
        entities.extend([c[:30] for c in code])
        
        return list(set(entities))[:5]
    
    def _requires_history(self, message: str, task_type: TaskType) -> bool:
        """Check if task requires conversation history."""
        history_indicators = [
            r'\b(as (?:we|i) discussed|as mentioned|referring to|regarding (?:our|the))\b',
            r'\b(previous|earlier|before|last time|follow up|continue|going back)\b',
            r'\b(the (?:file|code|topic|subject).{0,20}(?:showed|mentioned|said|had))\b',
            r'\b(what about|how about|and also|additionally|moreover)\b',
        ]
        
        for indicator in history_indicators:
            if re.search(indicator, message, re.IGNORECASE):
                return True
        
        return task_type in (TaskType.ANALYSIS, TaskType.CREATIVE, TaskType.DEBUGGING)
    
    def _requires_files(self, message: str) -> bool:
        """Check if task references files."""
        file_indicators = [
            r'\b(in|at|file|path|directory|folder|repo|repository)\s+[\w\-/]+\.\w+',
            r'\b(read|open|parse|process|load).{0,20}(file|csv|json|xml|yaml|txt)\b',
            r'`[^`]+\.(?:py|js|ts|java|rs|go|cpp|c|h|json|yaml|yml|xml|csv|md|txt)`',
        ]
        
        for indicator in file_indicators:
            if re.search(indicator, message, re.IGNORECASE):
                return True
        return False
    
    def _detect_urgency(self, message: str) -> int:
        """Detect urgency level (1-5)."""
        urgency = 1
        
        urgent_patterns = [
            (r'\b(urgent|asap|immediately|critical|blocking|broken|down|outage)\b', 3),
            (r'\b(hurry|quick|fast|now|soon|deadline|due)\b', 2),
            (r'(!{2,})', 2),
            (r'\b(help|emergency|panic|stuck)\b', 2),
        ]
        
        for pattern, boost in urgent_patterns:
            if re.search(pattern, message, re.IGNORECASE):
                urgency += boost
        
        return min(urgency, 5)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get classification statistics."""
        with self._lock:
            if not self._classification_history:
                return {"total": 0}
            
            type_counts = Counter(p.task_type.value for p in self._classification_history)
            complexity_counts = Counter(p.complexity.name for p in self._classification_history)
            avg_confidence = sum(p.confidence for p in self._classification_history) / len(self._classification_history)
            
            # Top intents
            top_intents = sorted(self._intent_history.items(), key=lambda x: x[1], reverse=True)[:5]
            
            return {
                "total": len(self._classification_history),
                "type_distribution": dict(type_counts),
                "complexity_distribution": dict(complexity_counts),
                "average_confidence": round(avg_confidence, 2),
                "top_intents": top_intents
            }


class EnhancedTokenAllocator:
    """
    Enhanced token allocator with dynamic optimization.
    
    Features:
    - Context-aware allocation
    - Dynamic ratio adjustment
    - Compression strategy selection
    - Historical optimization
    """
    
    # Base allocation ratios [system, history, working]
    BASE_RATIOS = {
        TaskType.CODE: [0.15, 0.35, 0.50],
        TaskType.DEBUGGING: [0.15, 0.40, 0.45],
        TaskType.ANALYSIS: [0.10, 0.40, 0.50],
        TaskType.RESEARCH: [0.10, 0.30, 0.60],
        TaskType.CREATIVE: [0.10, 0.20, 0.70],
        TaskType.COMMAND: [0.15, 0.15, 0.70],
        TaskType.CONVERSATION: [0.20, 0.50, 0.30],
        TaskType.UNKNOWN: [0.20, 0.40, 0.40],
    }
    
    # Complexity multipliers
    COMPLEXITY_MULTIPLIERS = {
        TaskComplexity.SIMPLE: 0.7,
        TaskComplexity.MODERATE: 1.0,
        TaskComplexity.COMPLEX: 1.4,
    }
    
    # Minimum allocations
    MIN_ALLOCATIONS = {
        "system": 500,
        "history": 800,
        "working": 500,
        "reserved": 500,  # For output
    }
    
    def __init__(
        self,
        max_context: int = 16000,
        enable_compression: bool = True,
        adaptive_ratios: bool = True
    ):
        self.max_context = max_context
        self.enable_compression = enable_compression
        self.adaptive_ratios = adaptive_ratios
        
        self._allocation_history: deque = deque(maxlen=50)
        self._efficiency_history: deque = deque(maxlen=100)
        self._lock = threading.Lock()
        
        # Adaptive ratio adjustments
        self._ratio_adjustments: Dict[TaskType, List[float]] = {}
    
    def allocate(
        self,
        message: str,
        profile: TaskProfile,
        current_history_size: int = 0,
        system_prompt_size: int = 2000
    ) -> AllocationResult:
        """
        Allocate tokens based on task profile.
        
        Args:
            message: User message
            profile: Task profile
            current_history_size: Current history token count
            system_prompt_size: Current system prompt size
            
        Returns:
            AllocationResult
        """
        # Calculate available space (reserve for output)
        available = self.max_context - self.MIN_ALLOCATIONS["reserved"]
        
        # Get base ratios
        base_ratios = self.BASE_RATIOS.get(profile.task_type, self.BASE_RATIOS[TaskType.UNKNOWN])
        
        # Apply adaptive adjustments
        if self.adaptive_ratios:
            ratios = self._apply_adaptive_ratios(profile.task_type, base_ratios, profile)
        else:
            ratios = base_ratios
        
        # Apply complexity multiplier
        complexity_mult = self.COMPLEXITY_MULTIPLIERS.get(profile.complexity, 1.0)
        
        # Calculate allocations
        system_alloc = max(
            self.MIN_ALLOCATIONS["system"],
            int(available * ratios[0])
        )
        
        # History allocation depends on whether history is required
        if profile.requires_history:
            history_alloc = max(
                self.MIN_ALLOCATIONS["history"],
                int(available * ratios[1] * complexity_mult)
            )
        else:
            history_alloc = self.MIN_ALLOCATIONS["history"]
        
        working_alloc = max(
            self.MIN_ALLOCATIONS["working"],
            int(available * ratios[2] * complexity_mult)
        )
        
        # Adjust for actual system prompt size
        system_alloc = max(system_alloc, system_prompt_size)
        
        # Ensure we don't exceed available
        total = system_alloc + history_alloc + working_alloc
        if total > available:
            scale = available / total
            system_alloc = int(system_alloc * scale)
            history_alloc = int(history_alloc * scale)
            working_alloc = available - system_alloc - history_alloc
        
        # Determine compression strategy
        compression_needed = total > available * 0.85
        compression_ratio = available / total if total > 0 else 1.0
        
        strategy = self._select_compression_strategy(profile, compression_ratio)
        
        result = AllocationResult(
            allocated=total,
            total_available=available,
            allocation_map={
                "system": system_alloc,
                "history": history_alloc,
                "working": working_alloc,
                "reserved": self.MIN_ALLOCATIONS["reserved"]
            },
            compression_needed=compression_needed,
            compression_ratio=min(1.0, compression_ratio),
            strategy=strategy
        )
        
        with self._lock:
            self._allocation_history.append(result)
        
        return result
    
    def _apply_adaptive_ratios(
        self,
        task_type: TaskType,
        base_ratios: List[float],
        profile: TaskProfile
    ) -> List[float]:
        """Apply adaptive adjustments to base ratios."""
        adjustments = self._ratio_adjustments.get(task_type, [0.0, 0.0, 0.0])
        
        # Adjust based on task characteristics
        if profile.requires_files:
            # Increase working memory for file operations
            adjustments = [a + (0.02 if i == 2 else -0.01) for i, a in enumerate(adjustments)]
        
        if profile.urgency >= 4:
            # For urgent tasks, prioritize working memory
            adjustments = [a + (0.01 if i == 2 else -0.005) for i, a in enumerate(adjustments)]
        
        # Apply adjustments with limits
        adjusted = []
        for base, adj in zip(base_ratios, adjustments):
            adjusted.append(max(0.05, min(0.8, base + adj)))
        
        # Normalize to sum to 1
        total = sum(adjusted)
        return [r / total for r in adjusted]
    
    def _select_compression_strategy(self, profile: TaskProfile, ratio: float) -> str:
        """Select compression strategy based on profile."""
        if ratio >= 0.9:
            return "none"
        
        strategies = {
            TaskType.CODE: "code_focused",
            TaskType.DEBUGGING: "preserve_recent",
            TaskType.ANALYSIS: "summarize",
            TaskType.RESEARCH: "semantic",
            TaskType.CREATIVE: "preserve_recent",
            TaskType.COMMAND: "aggressive",
            TaskType.CONVERSATION: "sample",
        }
        
        return strategies.get(profile.task_type, "standard")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get allocator statistics."""
        with self._lock:
            if not self._allocation_history:
                return {"allocations": 0}
            
            avg_usage = sum(
                a.allocated / a.total_available
                for a in self._allocation_history
            ) / len(self._allocation_history)
            
            compression_rate = sum(
                1 for a in self._allocation_history if a.compression_needed
            ) / len(self._allocation_history)
            
            return {
                "allocations": len(self._allocation_history),
                "average_usage": round(avg_usage, 2),
                "compression_rate": round(compression_rate, 2),
                "max_context": self.max_context
            }


class HierarchicalContextV2:
    """
    Enhanced hierarchical context management.
    
    Organizes conversation into prioritized levels:
    - L1: Immediate context (current turn)
    - L2: Recent context (last N turns)
    - L3: Summarized history
    - L4: Semantic clusters
    """
    
    def __init__(self, max_tokens: int = 16000):
        self.max_tokens = max_tokens
        self.levels = {
            "l1_immediate": [],
            "l2_recent": [],
            "l3_summary": [],
            "l4_clusters": []
        }
        self._token_counts = {level: 0 for level in self.levels}
        self._message_metadata: Dict[str, Dict] = {}
        self._lock = threading.Lock()
    
    def add_message(self, message: Dict, level: str = "l1"):
        """Add message at specified level."""
        if level not in self.levels:
            raise ValueError(f"Invalid level: {level}")
        
        with self._lock:
            msg_id = self._generate_msg_id(message)
            self.levels[level].append(message)
            self._token_counts[level] += self._estimate_tokens(message)
            self._message_metadata[msg_id] = {
                "added_at": time.time(),
                "level": level,
                "access_count": 0
            }
    
    def get_context(self, budget: int, priority_order: Optional[List[str]] = None) -> List[Dict]:
        """Get context within token budget."""
        if priority_order is None:
            priority_order = ["l1_immediate", "l2_recent", "l3_summary", "l4_clusters"]
        
        with self._lock:
            context = []
            remaining = budget
            
            for level in priority_order:
                for msg in self.levels[level]:
                    tokens = self._estimate_tokens(msg)
                    if remaining - tokens >= 0:
                        context.append(msg)
                        remaining -= tokens
                        
                        # Update access count
                        msg_id = self._generate_msg_id(msg)
                        if msg_id in self._message_metadata:
                            self._message_metadata[msg_id]["access_count"] += 1
                    else:
                        break
            
            return context
    
    def promote_message(self, message: Dict, to_level: str):
        """Promote a message to a higher priority level."""
        msg_id = self._generate_msg_id(message)
        
        with self._lock:
            # Find and remove from current level
            for level, messages in self.levels.items():
                if message in messages:
                    messages.remove(message)
                    self._token_counts[level] -= self._estimate_tokens(message)
                    break
            
            # Add to new level
            self.levels[to_level].append(message)
            self._token_counts[to_level] += self._estimate_tokens(message)
            
            if msg_id in self._message_metadata:
                self._message_metadata[msg_id]["level"] = to_level
    
    def summarize_level(self, level: str, max_summary_tokens: int = 500) -> Dict:
        """Create summary of a context level."""
        with self._lock:
            messages = self.levels.get(level, [])
            if not messages:
                return {"content": "", "metadata": {"level": level, "message_count": 0}}
            
            # Simple summarization (in practice, use LLM)
            combined = " ".join([m.get("content", "") for m in messages[-5:]])
            summary = combined[:max_summary_tokens * 4]  # Rough char estimate
            
            return {
                "content": f"[Summary of {len(messages)} messages]: {summary}...",
                "metadata": {
                    "level": level,
                    "message_count": len(messages),
                    "summarized": True
                }
            }
    
    def _estimate_tokens(self, message: Dict) -> int:
        """Estimate token count for message."""
        content = message.get("content", "")
        return int(len(content.split()) * 1.3) + 10
    
    def _generate_msg_id(self, message: Dict) -> str:
        """Generate unique ID for message."""
        content = message.get("content", "")
        return hashlib.md5(content.encode()).hexdigest()[:16]
    
    def get_stats(self) -> Dict[str, Any]:
        """Get context statistics."""
        with self._lock:
            return {
                "levels": {
                    level: {
                        "message_count": len(messages),
                        "token_count": self._token_counts[level]
                    }
                    for level, messages in self.levels.items()
                },
                "total_tokens": sum(self._token_counts.values()),
                "max_tokens": self.max_tokens
            }
