#!/usr/bin/env python3
"""
Chat Agent with Context Management

A simple chat agent using Llama 3.2-1B with configurable conversation history.

Features:
- Multiple context management strategies (unlimited, sliding window, summarization)
- Toggle conversation history on/off
- Token counting and context monitoring
- Clean conversation interface

Usage:
    python chat_agent.py [--model MODEL_NAME] [--no-history] [--context-strategy STRATEGY]

Examples:
    python chat_agent.py
    python chat_agent.py --no-history
    python chat_agent.py --context-strategy sliding_window --max-turns 5
    python chat_agent.py --model "meta-llama/Llama-3.2-1B-Instruct"

Requirements:
    pip install transformers torch
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import argparse
from datetime import datetime
from collections import deque
import sys

# ============================================================================
# CONFIGURATION
# ============================================================================

DEFAULT_MODEL = "meta-llama/Llama-3.2-1B-Instruct"
MAX_NEW_TOKENS = 256
TEMPERATURE = 0.7
TOP_P = 0.9

# Context management settings
MAX_CONTEXT_LENGTH = 2048  # Maximum tokens for model context
SLIDING_WINDOW_TURNS = 5   # Number of conversation turns to keep


# ============================================================================
# CONTEXT MANAGEMENT STRATEGIES
# ============================================================================

class ContextManager:
    """Base class for context management strategies"""
    
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.conversation_history = []
    
    def add_turn(self, user_message, assistant_message):
        """Add a conversation turn"""
        self.conversation_history.append({
            "user": user_message,
            "assistant": assistant_message
        })
    
    def get_context(self):
        """Get the current context (to be overridden by subclasses)"""
        raise NotImplementedError
    
    def count_tokens(self, text):
        """Count tokens in text"""
        return len(self.tokenizer.encode(text))
    
    def clear(self):
        """Clear conversation history"""
        self.conversation_history = []


class NoHistoryManager(ContextManager):
    """No conversation history - each turn is independent"""
    
    def get_context(self):
        return []
    
    def add_turn(self, user_message, assistant_message):
        # Don't save anything
        pass


class UnlimitedHistoryManager(ContextManager):
    """Keep all conversation history (will eventually exceed context limit)"""
    
    def get_context(self):
        return self.conversation_history.copy()


class SlidingWindowManager(ContextManager):
    """Keep only the last N turns of conversation"""
    
    def __init__(self, tokenizer, max_turns=SLIDING_WINDOW_TURNS):
        super().__init__(tokenizer)
        self.max_turns = max_turns
        self.conversation_history = deque(maxlen=max_turns)
    
    def add_turn(self, user_message, assistant_message):
        self.conversation_history.append({
            "user": user_message,
            "assistant": assistant_message
        })
    
    def get_context(self):
        return list(self.conversation_history)


class TokenLimitManager(ContextManager):
    """Keep history within token limit by removing oldest turns"""
    
    def __init__(self, tokenizer, max_tokens=MAX_CONTEXT_LENGTH):
        super().__init__(tokenizer)
        self.max_tokens = max_tokens
    
    def get_context(self):
        """Return conversation history that fits within token limit"""
        context = []
        total_tokens = 0
        
        # Add turns from most recent to oldest
        for turn in reversed(self.conversation_history):
            turn_text = f"User: {turn['user']}\nAssistant: {turn['assistant']}\n"
            turn_tokens = self.count_tokens(turn_text)
            
            if total_tokens + turn_tokens > self.max_tokens:
                break
            
            context.insert(0, turn)
            total_tokens += turn_tokens
        
        return context


class SummarizationManager(ContextManager):
    """Summarize old conversation when approaching token limit"""
    
    def __init__(self, tokenizer, model, max_tokens=MAX_CONTEXT_LENGTH):
        super().__init__(tokenizer)
        self.model = model
        self.max_tokens = max_tokens
        self.summary = None
    
    def get_context(self):
        """Return summarized old context + recent turns"""
        context = []
        
        # Add summary if available
        if self.summary:
            context.append({
                "user": "[Previous conversation summary]",
                "assistant": self.summary
            })
        
        # Add recent turns
        recent_turns = self.conversation_history[-3:]  # Keep last 3 turns
        context.extend(recent_turns)
        
        return context
    
    def add_turn(self, user_message, assistant_message):
        super().add_turn(user_message, assistant_message)
        
        # Check if we need to summarize
        context_text = self._format_history(self.conversation_history)
        if self.count_tokens(context_text) > self.max_tokens * 0.7:  # At 70% capacity
            self._create_summary()
    
    def _create_summary(self):
        """Create a summary of old conversation"""
        if len(self.conversation_history) <= 3:
            return
        
        # Summarize all but the last 3 turns
        old_turns = self.conversation_history[:-3]
        old_conversation = self._format_history(old_turns)
        
        prompt = f"""Summarize this conversation in 2-3 sentences:

{old_conversation}

Summary:"""
        
        # Generate summary (simplified - in production you'd use the model properly)
        self.summary = f"Previously discussed: {len(old_turns)} topics including the conversation shown above."
        
        # Remove old turns, keep only recent ones
        self.conversation_history = list(self.conversation_history[-3:])
    
    def _format_history(self, history):
        """Format conversation history as text"""
        lines = []
        for turn in history:
            lines.append(f"User: {turn['user']}")
            lines.append(f"Assistant: {turn['assistant']}")
        return "\n".join(lines)


# ============================================================================
# CHAT AGENT
# ============================================================================

class ChatAgent:
    """Chat agent with configurable context management"""
    
    def __init__(self, model_name=DEFAULT_MODEL, context_manager=None, device=None):
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else 
                                "mps" if torch.backends.mps.is_available() else "cpu")
        
        print(f"\n{'='*70}")
        print(f"Loading model: {model_name}")
        print(f"Device: {self.device}")
        print(f"{'='*70}\n")
        
        # Load model and tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        if self.device == "cuda":
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16,
                device_map="auto"
            )
        elif self.device == "mps":
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16
            )
            self.model = self.model.to(self.device)
        else:  # CPU
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float32
            )
            self.model = self.model.to(self.device)
        
        self.model.eval()
        
        # Set up context manager
        self.context_manager = context_manager or UnlimitedHistoryManager(self.tokenizer)
        
        print("✓ Model loaded successfully!\n")
    
    def format_prompt(self, user_message):
        """Format the prompt with conversation history"""
        context = self.context_manager.get_context()
        
        # Build conversation
        messages = []
        
        # Add conversation history
        for turn in context:
            messages.append({"role": "user", "content": turn["user"]})
            messages.append({"role": "assistant", "content": turn["assistant"]})
        
        # Add current message
        messages.append({"role": "user", "content": user_message})
        
        # Apply chat template
        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        return prompt
    
    def generate_response(self, user_message):
        """Generate a response to the user message"""
        # Format prompt with context
        prompt = self.format_prompt(user_message)
        
        # Tokenize
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        
        # Count input tokens
        input_token_count = inputs['input_ids'].shape[1]
        
        # Generate response
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                temperature=TEMPERATURE,
                top_p=TOP_P,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        # Decode response (only the new tokens)
        response = self.tokenizer.decode(
            outputs[0][inputs['input_ids'].shape[1]:],
            skip_special_tokens=True
        )
        
        # Count output tokens
        output_token_count = outputs.shape[1] - input_token_count
        
        # Add to conversation history
        self.context_manager.add_turn(user_message, response)
        
        return response, input_token_count, output_token_count
    
    def chat(self):
        """Interactive chat loop"""
        print(f"{'='*70}")
        print(f"Chat Agent Started")
        print(f"Model: {self.model_name}")
        print(f"Context Strategy: {self.context_manager.__class__.__name__}")
        print(f"Device: {self.device}")
        print(f"{'='*70}\n")
        print("Type 'quit', 'exit', or 'q' to end the conversation")
        print("Type 'clear' to clear conversation history")
        print("Type 'stats' to see conversation statistics")
        print(f"{'='*70}\n")
        
        turn_number = 0
        
        while True:
            # Get user input
            try:
                user_input = input("You: ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\n\nGoodbye!")
                break
            
            if not user_input:
                continue
            
            # Check for commands
            if user_input.lower() in ['quit', 'exit', 'q']:
                print("\nGoodbye!")
                break
            
            if user_input.lower() == 'clear':
                self.context_manager.clear()
                turn_number = 0
                print("✓ Conversation history cleared!\n")
                continue
            
            if user_input.lower() == 'stats':
                self.print_stats()
                continue
            
            # Generate response
            turn_number += 1
            print("\nAssistant: ", end="", flush=True)
            
            try:
                response, input_tokens, output_tokens = self.generate_response(user_input)
                print(response)
                print(f"\n[Turn {turn_number} | Input: {input_tokens} tokens | Output: {output_tokens} tokens]\n")
            except Exception as e:
                print(f"\n❌ Error generating response: {e}\n")
                continue
    
    def print_stats(self):
        """Print conversation statistics"""
        context = self.context_manager.get_context()
        print(f"\n{'='*70}")
        print("Conversation Statistics")
        print(f"{'='*70}")
        print(f"Total turns: {len(context)}")
        
        if context:
            total_tokens = sum(
                self.context_manager.count_tokens(turn['user']) + 
                self.context_manager.count_tokens(turn['assistant'])
                for turn in context
            )
            print(f"Total context tokens: {total_tokens}")
            print(f"Average tokens per turn: {total_tokens / len(context):.1f}")
        
        print(f"{'='*70}\n")


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Chat Agent with Context Management")
    
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_MODEL,
        help=f"Model name (default: {DEFAULT_MODEL})"
    )
    
    parser.add_argument(
        "--no-history",
        action="store_true",
        help="Disable conversation history (each turn is independent)"
    )
    
    parser.add_argument(
        "--context-strategy",
        type=str,
        choices=["unlimited", "sliding_window", "token_limit", "summarization"],
        default="unlimited",
        help="Context management strategy"
    )
    
    parser.add_argument(
        "--max-turns",
        type=int,
        default=SLIDING_WINDOW_TURNS,
        help=f"Max turns for sliding window (default: {SLIDING_WINDOW_TURNS})"
    )
    
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=MAX_CONTEXT_LENGTH,
        help=f"Max tokens for context (default: {MAX_CONTEXT_LENGTH})"
    )
    
    parser.add_argument(
        "--device",
        type=str,
        choices=["cpu", "cuda", "mps", "auto"],
        default="auto",
        help="Device to use for inference"
    )
    
    args = parser.parse_args()
    
    # Determine device
    if args.device == "auto":
        device = None  # Let ChatAgent auto-detect
    else:
        device = args.device
    
    # Create tokenizer for context manager
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    
    # Create context manager based on strategy
    if args.no_history:
        print("Using: No History (each turn is independent)")
        context_manager = NoHistoryManager(tokenizer)
    elif args.context_strategy == "sliding_window":
        print(f"Using: Sliding Window (last {args.max_turns} turns)")
        context_manager = SlidingWindowManager(tokenizer, max_turns=args.max_turns)
    elif args.context_strategy == "token_limit":
        print(f"Using: Token Limit ({args.max_tokens} tokens)")
        context_manager = TokenLimitManager(tokenizer, max_tokens=args.max_tokens)
    elif args.context_strategy == "summarization":
        print(f"Using: Summarization (max {args.max_tokens} tokens)")
        # We'll pass the model after loading
        context_manager = None  # Will be created after model loads
    else:  # unlimited
        print("Using: Unlimited History (⚠️  may exceed context limit)")
        context_manager = UnlimitedHistoryManager(tokenizer)
    
    # Create chat agent
    agent = ChatAgent(
        model_name=args.model,
        context_manager=context_manager,
        device=device
    )
    
    # If using summarization, update context manager with model
    if args.context_strategy == "summarization":
        agent.context_manager = SummarizationManager(
            tokenizer, 
            agent.model, 
            max_tokens=args.max_tokens
        )
    
    # Start chat
    agent.chat()


if __name__ == "__main__":
    main()