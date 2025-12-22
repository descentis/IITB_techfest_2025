import os
import json
from typing import Annotated, Literal
from datetime import datetime


from pydantic import BaseModel
from langchain_ollama import ChatOllama  
from langchain.tools import tool
from langchain_core.messages import SystemMessage, HumanMessage, ToolMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph import MessagesState


# ============================================================================
# STEP 1: Define Tools for the Agent
# ============================================================================


@tool
def brainstorm_ideas(topic: str, audience: str, num_ideas: int = 5) -> str:
    """
    Brainstorm content ideas on a given topic for a specific audience.
    
    Args:
        topic: The main topic/niche (e.g., "Machine Learning", "Productivity")
        audience: Target audience (e.g., "beginners", "professionals")
        num_ideas: Number of ideas to generate (default 5)
    
    Returns:
        List of creative content ideas
    """
    return f"""
    Brainstorm for Topic: {topic}
    Target Audience: {audience}
    Number of Ideas: {num_ideas}
    
    TOOL OUTPUT:
    The LLM will generate {num_ideas} unique content ideas tailored to {audience} 
    interested in {topic}. Ideas will include working titles, hooks, and brief descriptions.
    """


@tool
def create_outline(idea_title: str, content_type: str, depth: str = "detailed") -> str:
    """
    Create a detailed outline for a content piece.
    
    Args:
        idea_title: Title of the content idea
        content_type: Type of content (blog, video, podcast, tutorial, etc.)
        depth: Level of detail (brief, standard, detailed)
    
    Returns:
        Structured outline with sections and subsections
    """
    return f"""
    Create Outline for: {idea_title}
    Content Type: {content_type}
    Depth Level: {depth}
    
    TOOL OUTPUT:
    Generates a structured outline including:
    - Main sections with key points
    - Subsections and talking points
    - Estimated time/word count per section
    - Key takeaways
    - Call-to-action suggestions
    """


@tool
def generate_hook(topic: str, content_type: str, style: str = "engaging") -> str:
    """
    Generate an attention-grabbing opening hook for content.
    
    Args:
        topic: Topic of the content
        content_type: Type of content (blog, video, social media)
        style: Style of hook (engaging, educational, controversial, storytelling)
    
    Returns:
        Multiple hook options to choose from
    """
    return f"""
    Generate Hook for: {topic}
    Content Type: {content_type}
    Style: {style}
    
    TOOL OUTPUT:
    Provides 3-5 different hook variations including:
    - Questions to intrigue readers
    - Statistics or surprising facts
    - Story-based openers
    - Direct benefit statements
    """


@tool
def research_keywords(topic: str, audience: str) -> str:
    """
    Research relevant keywords and search terms for SEO optimization.
    
    Args:
        topic: Main content topic
        audience: Target audience for keyword context
    
    Returns:
        List of high-value keywords and search terms
    """
    return f"""
    Research Keywords for: {topic}
    Target Audience: {audience}
    
    TOOL OUTPUT:
    Returns keyword analysis including:
    - High-volume, low-competition keywords
    - Long-tail keyword suggestions
    - Related search terms
    - User intent mapping
    - Content gaps to address
    """


@tool
def personalize_content(base_idea: str, personalization_params: str) -> str:
    """
    Adapt content ideas based on personal brand, voice, and preferences.
    
    Args:
        base_idea: The original content idea
        personalization_params: JSON with preferences (brand_voice, niche, values, etc.)
    
    Returns:
        Personalized content approach aligned with brand
    """
    return f"""
    Personalize Content: {base_idea}
    Parameters: {personalization_params}
    
    TOOL OUTPUT:
    Tailors the content to match:
    - Brand voice and tone
    - Personal expertise areas
    - Audience connection points
    - Unique perspective/angle
    - Platform best practices
    """


@tool
def create_content_calendar(ideas_list: str, time_frame: str = "4 weeks") -> str:
    """
    Create a content distribution calendar from generated ideas.
    
    Args:
        ideas_list: List of content ideas (comma-separated or JSON)
        time_frame: Planning period (1 week, 2 weeks, 4 weeks, 3 months, 1 year)
    
    Returns:
        Publishing schedule with optimal posting times
    """
    return f"""
    Create Content Calendar
    Ideas to Schedule: {ideas_list}
    Time Frame: {time_frame}
    
    TOOL OUTPUT:
    Generates a calendar showing:
    - Publishing dates and frequency
    - Optimal posting times by platform
    - Content variety distribution
    - Seasonal opportunities
    - Engagement timing recommendations
    """


@tool
def suggest_formats(topic: str, audience: str) -> str:
    """
    Suggest diverse content formats for better reach and engagement.
    
    Args:
        topic: Content topic
        audience: Target audience
    
    Returns:
        Recommended content formats with platform suggestions
    """
    return f"""
    Suggest Formats for: {topic}
    Target Audience: {audience}
    
    TOOL OUTPUT:
    Recommends multiple formats:
    - Blog posts
    - Video tutorials
    - Podcasts/audio
    - Infographics
    - Social media threads
    - Interactive tools
    - Case studies
    With platform recommendations and audience preferences
    """


# ============================================================================
# STEP 1.5: LLM-based Tool Selection & Ambiguity Detection (NEW - NO HARDCODING)
# ============================================================================

def get_llm_for_analysis():
    """Create a dedicated LLM instance for analysis tasks"""
    return ChatOllama(
        model="mistral", 
        base_url="http://localhost:11434",
        temperature=0.3,  # Lower temperature for consistent analysis
    )


def analyze_request_with_llm(user_input: str) -> dict:
    """
    Use LLM to intelligently analyze user request.
    Returns: {
        'tools_needed': list of tool names,
        'is_ambiguous': bool,
        'missing_info': list of what's missing,
        'confidence_score': float (0-1),
        'reasoning': str (why the agent made these decisions)
    }
    
    No hardcoding! Let the LLM understand natural language.
    """
    
    analysis_llm = get_llm_for_analysis()
    
    analysis_prompt = """
    Analyze this user request and provide structured analysis.
    
    Available tools:
    1. brainstorm_ideas - Generate creative content ideas for a topic and audience
    2. create_outline - Create detailed outlines for content pieces
    3. generate_hook - Create attention-grabbing opening hooks
    4. research_keywords - Research SEO keywords and search terms
    5. personalize_content - Adapt content to match brand/voice
    6. create_content_calendar - Schedule content distribution over time
    7. suggest_formats - Recommend content formats and platforms
    
    User request: "{user_input}"
    
    Respond in this exact JSON format (no markdown, just pure JSON):
    {{
        "tools_needed": ["tool_name1", "tool_name2"],
        "is_ambiguous": true/false,
        "missing_info": ["what's missing", "other missing thing"],
        "confidence_score": 0.85,
        "reasoning": "Your explanation of why these tools are needed and what's missing"
    }}
    
    Guidelines:
    - Be thorough in understanding natural language
    - If user says "don't want X" or "skip X", don't include that tool
    - Identify all genuinely missing information (not just minor details)
    - Confidence = 1.0 if all critical info present, lower if significant info missing
    - Consider context: a request for "ideas" doesn't need calendar tool
    - A request for "schedule" definitely needs calendar tool
    - Think about what the user actually needs, not just keywords
    """.format(user_input=user_input)
    
    try:
        response = analysis_llm.invoke([HumanMessage(content=analysis_prompt)])
        
        # Extract JSON from response
        response_text = response.content
        
        # Try to parse JSON
        import json
        analysis = json.loads(response_text)
        
        return analysis
    
    except json.JSONDecodeError:
        # If JSON parsing fails, return a safe default
        print("⚠️  Warning: Could not parse LLM analysis response, using safe defaults")
        return {
            "tools_needed": ["brainstorm_ideas", "suggest_formats"],
            "is_ambiguous": True,
            "missing_info": ["Please provide more details about your request"],
            "confidence_score": 0.3,
            "reasoning": "Could not fully parse request - asking for clarification"
        }
    except Exception as e:
        print(f"⚠️  Warning: Error during LLM analysis: {e}")
        return {
            "tools_needed": ["brainstorm_ideas", "suggest_formats"],
            "is_ambiguous": True,
            "missing_info": ["Please try again"],
            "confidence_score": 0.2,
            "reasoning": f"Analysis error: {str(e)}"
        }


def generate_clarifying_questions_with_llm(user_input: str, analysis: dict) -> str:
    """
    Use LLM to generate natural, contextual clarifying questions.
    No templates! Just natural language.
    """
    
    question_llm = get_llm_for_analysis()
    
    question_prompt = """
    A user made this request: "{user_input}"
    
    Analysis of missing information:
    - Tools needed: {tools}
    - Is ambiguous: {ambiguous}
    - Missing info: {missing}
    - Reasoning: {reasoning}
    
    Generate 3-5 natural, conversational clarifying questions to help the user 
    provide the missing information. Make them friendly and specific to their request.
    
    Do NOT use a template format like:
    "1. Question here?
     2. Question here?"
    
    Instead, write it naturally, like you're having a conversation.
    Keep it concise but helpful.
    """.format(
        user_input=user_input,
        tools=', '.join(analysis['tools_needed']),
        ambiguous=analysis['is_ambiguous'],
        missing=', '.join(analysis['missing_info']),
        reasoning=analysis['reasoning']
    )
    
    try:
        response = question_llm.invoke([HumanMessage(content=question_prompt)])
        return response.content
    except Exception as e:
        # Fallback to simple message
        missing_str = ', '.join(analysis['missing_info'])
        return f"To better help you, could you provide more details about: {missing_str}"


# ============================================================================
# STEP 2: Initialize LLM and Define Agent Nodes
# ============================================================================


def create_content_agent():
    """
    Create and return the Content Idea Generator & Planner Agent using LangGraph.
    
    This version uses LLM for intelligent analysis instead of hardcoded rules.
    """
    

    llm = ChatOllama(
        model="mistral", 
        base_url="http://localhost:11434",
        temperature=0.7,
    )
    
    # Define all tools
    all_tools = [
        brainstorm_ideas,
        create_outline,
        generate_hook,
        research_keywords,
        personalize_content,
        create_content_calendar,
        suggest_formats,
    ]
    
    # Create tools dictionary for lookup
    tools_by_name = {tool.name: tool for tool in all_tools}
    
    # ============================================================================
    # STEP 3: Define Node Functions
    # ============================================================================
    
    def llm_call(state: MessagesState):
        """LLM node - analyzes request, checks ambiguity, then proceeds or asks questions"""
        messages = state["messages"]
        
        # Extract user's latest request
        user_message = None
        for msg in reversed(messages):
            if isinstance(msg, HumanMessage):
                user_message = msg.content
                break
        
        # ============================================================================
        # LLM-BASED ANALYSIS (NO HARDCODING!)
        # ============================================================================
        print("\n🧠 Analyzing request with LLM...")
        analysis = analyze_request_with_llm(user_message)
        
        selected_tool_names = analysis['tools_needed']
        is_ambiguous = analysis['is_ambiguous']
        confidence_score = analysis['confidence_score']
        reasoning = analysis['reasoning']
        
        # Display analysis to user
        print(f"\n📌 Tools selected: {', '.join(selected_tool_names)}")
        print(f"📊 Confidence: {confidence_score*100:.0f}%")
        print(f"💭 Reasoning: {reasoning}")
        
        # ============================================================================
        # CHECK AMBIGUITY (LLM-determined)
        # ============================================================================
        if is_ambiguous:
            missing_info = ', '.join(analysis['missing_info'])
            print(f"⚠️  Request is ambiguous. Missing: {missing_info}\n")
            
            # Generate natural questions using LLM
            clarifying_questions = generate_clarifying_questions_with_llm(user_message, analysis)
            
            # Return questions
            response = HumanMessage(content=clarifying_questions)
            response.__class__.__name__ = "AIMessage"
            return {"messages": [response]}
        
        # ============================================================================
        # NOT AMBIGUOUS - PROCEED WITH TOOLS
        # ============================================================================
        print("✅ Request is clear, proceeding with tool execution\n")
        
        selected_tools = [tool for tool in all_tools if tool.name in selected_tool_names]
        llm_with_selected_tools = llm.bind_tools(selected_tools)
        
        # Define system prompt
        system_prompt = f"""
You are an expert Content Strategy & Planning AI Agent helping creators, marketers, 
and writers develop engaging, strategic content.

Your reasoning process:
1. You understand natural language and what users really want
2. You select appropriate tools based on their needs
3. You ask clarifying questions when needed
4. You provide specific, actionable advice

For this request, use these tools:
{', '.join(selected_tool_names)}

Reasoning: {reasoning}

When responding:
- Use ONLY the available tools for this request
- Provide specific, actionable advice
- Balance creativity with strategy
- Consider platform-specific best practices
- Think about audience psychology and engagement

Format your responses clearly with:
- Key recommendations
- Actionable next steps
- Reasoning behind suggestions
- Examples when helpful
"""
        
        # Invoke LLM with selected tools
        response = llm_with_selected_tools.invoke(
            [SystemMessage(content=system_prompt)] + messages
        )
        
        return {"messages": [response]}
    
    def tool_node(state: MessagesState):
        """Tool node - executes tool calls"""
        messages = state["messages"]
        last_message = messages[-1]
        
        results = []
        
        # Execute each tool call
        for tool_call in last_message.tool_calls:
            tool = tools_by_name[tool_call["name"]]
            observation = tool.invoke(tool_call["args"])
            
            # Create ToolMessage with observation
            result = ToolMessage(
                content=observation,
                tool_call_id=tool_call["id"],
                name=tool_call["name"]
            )
            results.append(result)
        
        return {"messages": results}
    
    # ============================================================================
    # STEP 4: Define Conditional Edge Function
    # ============================================================================
    
    def should_continue(state: MessagesState) -> Literal["tool_node", END]:
        """
        Decide if we should continue the loop by calling a tool,
        or stop and return the final response
        """
        messages = state["messages"]
        last_message = messages[-1]
        
        # If the LLM makes a tool call, go to tool_node
        if hasattr(last_message, "tool_calls") and last_message.tool_calls:
            return "tool_node"
        
        # Otherwise, we stop (reply to the user)
        return END
    
    # ============================================================================
    # STEP 5: Build the Agent Graph
    # ============================================================================
    
    agent_builder = StateGraph(MessagesState)
    
    # Add nodes
    agent_builder.add_node("llm_call", llm_call)
    agent_builder.add_node("tool_node", tool_node)
    
    # Add edges
    agent_builder.add_edge(START, "llm_call")
    
    # Conditional edge
    agent_builder.add_conditional_edges(
        "llm_call",
        should_continue,
        {
            "tool_node": "tool_node",
            END: END
        }
    )
    
    # After tool execution, loop back to llm_call
    agent_builder.add_edge("tool_node", "llm_call")
    
    # Compile the agent
    agent = agent_builder.compile()
    
    return agent


# ============================================================================
# STEP 6: Interaction Functions
# ============================================================================


def run_agent_conversation(agent, user_input: str):
    """
    Send a message to the agent and get a response.
    """
    try:
        # Prepare input messages
        messages = [HumanMessage(content=user_input)]
        
        # Invoke the agent
        response = agent.invoke({"messages": messages})
        
        # Extract final response from messages
        final_messages = response["messages"]
        
        # Find the last non-ToolMessage
        for message in reversed(final_messages):
            if hasattr(message, 'content') and not message.__class__.__name__ == 'ToolMessage':
                return message.content
        
        # Fallback
        return str(final_messages[-1].content) if final_messages else "No response generated"
    
    except Exception as e:
        return f"Error: {str(e)}\n\nTroubleshooting:\n1. Make sure Ollama is running\n2. Run: ollama serve\n3. Verify the model: ollama list"


def interactive_session():
    """
    Start an interactive session with the Content Agent.
    """
    print("\n" + "="*70)
    print("🎬 Content Idea Generator & Planner Agent")
    print("   (LLM-Powered Natural Language Understanding)")
    print("="*70)
    print("\nWelcome! I'm your AI content strategy partner.")
    print("I can help you:")
    print("  • Generate creative content ideas")
    print("  • Create detailed outlines")
    print("  • Develop compelling hooks")
    print("  • Research keywords")
    print("  • Plan content calendars")
    print("  • Suggest content formats")
    print("\n✨ NEW: I use natural language understanding to analyze your requests!")
    print("   No hardcoded keyword matching - pure LLM intelligence.")
    print("\nType 'exit' or 'quit' to end the conversation.")
    print("Type 'help' for example prompts.")
    print("-"*70 + "\n")
    
    # Initialize agent
    print("🔄 Initializing agent (this may take a moment)...")
    try:
        agent = create_content_agent()
        print("✅ Agent ready!\n")
    except Exception as e:
        print(f"❌ Error initializing agent: {e}")
        print("\nMake sure:")
        print("1. Ollama is installed (https://ollama.ai)")
        print("2. Run 'ollama serve' in another terminal")
        print("3. Model is downloaded: 'ollama pull mistral'")
        print("4. Install langchain-ollama: 'pip install langchain-ollama'")
        return
    
    # Conversation loop
    while True:
        try:
            user_input = input("You: ").strip()
            
            if not user_input:
                continue
            
            if user_input.lower() in ['exit', 'quit']:
                print("\n👋 Thank you for using the Content Agent. Happy creating!")
                break
            
            if user_input.lower() == 'help':
                print_example_prompts()
                continue
            
            # Process user input
            print("\n🤔 Analyzing request...\n")
            response = run_agent_conversation(agent, user_input)
            print(f"Agent: {response}\n")
        
        except KeyboardInterrupt:
            print("\n\n👋 Session ended. Goodbye!")
            break
        except Exception as e:
            print(f"Error: {e}\n")


def print_example_prompts():
    """Print example prompts for the user."""
    examples = """
    📝 Example Prompts (LLM will understand context):
    
    1. Vague Request (LLM will ask for clarification):
       "Create a calendar"
       "I need content ideas"
       "Help me with my social media"
    
    2. Partial Information (LLM will ask what's missing):
       "Blog ideas for beginners"
       "Content for productivity"
       "Schedule my posts"
    
    3. Clear Requests (LLM will proceed immediately):
       "Generate 5 YouTube video ideas about Python for beginners"
       "Create a 4-week AI tutorial calendar for developers on YouTube"
       "Research keywords for machine learning tutorials"
    
    4. Specific Exclusions (LLM will understand):
       "Ideas about AI but don't create a calendar"
       "Give me formats without keyword research"
       "Blog ideas - skip the scheduling part"
    
    5. Complex Requests (LLM will parse context):
       "I'm a data scientist targeting professionals on LinkedIn. 
        Create a content strategy for machine learning posts"
       "I want weekly content about productivity for remote workers
        on Instagram and TikTok with short video formats"
    
    6. Conversational Style:
       "Can you help me plan my next quarter of content about AI?"
       "What should I write about for my tech blog targeting beginners?"
       "I need ideas for a YouTube series on Python programming"
    """
    print(examples)


# ============================================================================
# STEP 7: Main Entry Point
# ============================================================================


if __name__ == "__main__":
    print("\n" + "="*70)
    print("Content Idea Generator & Planner Agent - Setup Guide")
    print("="*70)
    
    print("\n📋 PREREQUISITES:")
    print("1. Install Ollama (free, open-source):")
    print("   • Download from https://ollama.ai")
    print("   • Install for your OS (Mac, Windows, Linux)")
    print("\n2. Pull a model:")
    print("   • Open terminal/command prompt")
    print("   • Run: ollama pull mistral")
    print("   • (Or any model: ollama pull neural-chat, ollama pull llama2)")
    print("\n3. Install langchain-ollama:")
    print("   • Run: pip install langchain-ollama")
    print("   • This provides ChatOllama with bind_tools() support")
    print("\n4. Start Ollama server:")
    print("   • Run: ollama serve")
    print("   • Keep this terminal open while using the agent")
    print("\n" + "="*70)
    
    # Check if Ollama is running
    import socket
    
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    ollama_running = sock.connect_ex(('localhost', 11434)) == 0
    sock.close()
    
    if not ollama_running:
        print("\n❌ Ollama is not running!")
        print("\n🚀 Please:")
        print("1. Make sure Ollama is installed")
        print("2. Run 'ollama serve' in another terminal")
        print("3. Then run this script again")
        print("\nAfter starting Ollama, the agent will launch automatically.")
    else:
        print("\n✅ Ollama is running!")
        print("Starting the Content Agent...\n")
        interactive_session()
