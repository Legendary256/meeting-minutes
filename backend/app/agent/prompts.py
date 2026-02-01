"""
Prompts for Meeting Agent.
"""
from typing import List, Optional


class AgentPrompts:
    """Managing prompts for Meeting Agent."""
    
    def generate_todos_prompt(
        self,
        agenda: str,
        goals: Optional[List[str]] = None,
        context: Optional[str] = None
    ) -> str:
        """Prompt for generating TODO list from agenda."""
        
        goals_text = ""
        if goals:
            goals_text = f"""
## Goals to achieve:
{chr(10).join(f"- {g}" for g in goals)}
"""
        
        context_text = ""
        if context:
            context_text = f"""
## Context (previous meetings, background):
{context}
"""
        
        return f"""You are a meeting assistant. Based on the agenda, create a list of specific topics/questions to discuss.

## Meeting agenda:
{agenda}
{goals_text}
{context_text}

## Task:
Analyze the agenda and generate a TODO list - specific items to discuss during the meeting.

For each item, specify:
- topic: Short topic name (max 10 words)
- description: What exactly needs to be discussed/decided
- priority: Priority 1-5 (1 = highest)

## Response format (JSON):
```json
[
  {{
    "topic": "Topic name",
    "description": "What to discuss, what questions to ask, what to decide",
    "priority": 1
  }},
  ...
]
```

Generate 5-15 specific TODO items. Be practical and specific.
Respond ONLY with JSON, no additional text."""
    
    def analyze_transcript_prompt(
        self,
        transcript: str,
        pending_topics: List[str],
        discussed_topics: List[str],
        goals: List[str]
    ) -> str:
        """Prompt for analyzing transcript fragment."""
        
        return f"""You are an assistant analyzing meeting progress in real-time.

## New transcript fragment:
{transcript}

## Topics NOT YET discussed:
{chr(10).join(f"- {t}" for t in pending_topics) if pending_topics else "- All topics discussed"}

## Topics already discussed:
{chr(10).join(f"- {t}" for t in discussed_topics) if discussed_topics else "- None"}

## Meeting goals:
{chr(10).join(f"- {g}" for g in goals) if goals else "- Not specified"}

## Task:
Analyze the fragment and determine:
1. Which pending topics were addressed in this fragment
2. What questions are worth asking (follow-up to the discussion)
3. Are there any warnings (e.g., going off-topic, skipping important issues)

## Response format (JSON):
```json
{{
  "topics_discussed": [
    {{
      "topic": "Topic name from pending list",
      "key_points": ["Main point 1", "Main point 2"],
      "completion": "full|partial"
    }}
  ],
  "suggestions": [
    {{
      "type": "question|followup|reminder",
      "content": "Suggestion/question content",
      "priority": "low|normal|high",
      "related_todo": "topic_name or null"
    }}
  ],
  "warnings": [
    "Warning if something needs attention"
  ]
}}
```

Respond ONLY with JSON. Be concise and practical."""
    
    def final_summary_prompt(
        self,
        transcript: str,
        todos: List,
        goals: List[str]
    ) -> str:
        """Prompt for final meeting summary."""
        
        todos_status = []
        for todo in todos:
            status_emoji = "✅" if todo.status.value == "discussed" else "❌"
            todos_status.append(f"{status_emoji} {todo.topic}")
        
        return f"""You are an assistant creating professional meeting summaries.

## Full meeting transcript:
{transcript[:15000]}

## Agenda topics status:
{chr(10).join(todos_status)}

## Meeting goals:
{chr(10).join(f"- {g}" for g in goals) if goals else "- Not specified"}

## Task:
Create a professional meeting summary containing:

1. **Summary** (3-5 sentences)
2. **Key decisions** (bullet points)
3. **Action items** (who, what, by when)
4. **Open issues** (what needs further discussion)
5. **Next steps**

Be specific and practical. Use Markdown formatting."""
    
    def suggest_questions_prompt(
        self,
        current_topic: str,
        transcript_excerpt: str,
        goals: List[str]
    ) -> str:
        """Prompt for suggesting questions during discussion."""
        
        return f"""Context: Discussion is ongoing about "{current_topic}".

Conversation excerpt:
{transcript_excerpt}

Meeting goals:
{chr(10).join(f"- {g}" for g in goals) if goals else "- General discussion"}

Suggest 2-3 specific questions worth asking to:
- Deepen the topic
- Ensure nothing is missed
- Establish specific next steps

Format: List of questions, each on a new line."""
