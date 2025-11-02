#!/usr/bin/env python3
"""
Gradio Chat Interface
Web-based chat interface for interacting with trained model
"""

import sys
import torch
from pathlib import Path
from typing import List, Tuple, Optional

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

try:
    import gradio as gr
except ImportError:
    print("ERROR: Gradio not installed.")
    print("Install with: pip install gradio")
    sys.exit(1)

from models import load_model_from_config
from tokenizers import Tokenizer


class ChatInterface:
    """
    Gradio-based chat interface for trained model
    """
    
    def __init__(self, checkpoint_path: Optional[str] = None):
        """
        Initialize chat interface
        
        Args:
            checkpoint_path: Path to model checkpoint (auto-detect if None)
        """
        self.checkpoint_path = checkpoint_path or self._find_best_checkpoint()
        self.model = None
        self.tokenizer = None
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model_info = {}
        
    def _find_best_checkpoint(self) -> str:
        """
        Find best checkpoint with priority:
        1. checkpoint-best.pt
        2. Most recent checkpoint-*.pt
        3. checkpoint-final.pt
        """
        checkpoint_dir = Path('checkpoints')
        
        if not checkpoint_dir.exists():
            raise FileNotFoundError(
                "No checkpoints directory found. Please train a model first:\n"
                "  python training/train.py"
            )
        
        # Priority 1: Best checkpoint
        best_checkpoint = checkpoint_dir / 'checkpoint-best.pt'
        if best_checkpoint.exists():
            return str(best_checkpoint)
        
        # Priority 2: Most recent numbered checkpoint
        checkpoints = list(checkpoint_dir.glob('checkpoint-*.pt'))
        if checkpoints:
            # Extract step numbers and find max
            numbered = []
            for cp in checkpoints:
                try:
                    # Handle both checkpoint-1000.pt and checkpoint-best.pt
                    parts = cp.stem.split('-')
                    if len(parts) >= 2 and parts[1].isdigit():
                        step = int(parts[1])
                        numbered.append((step, cp))
                except (ValueError, IndexError):
                    continue
            
            if numbered:
                numbered.sort(reverse=True)
                return str(numbered[0][1])
        
        # Priority 3: Final checkpoint
        final_checkpoint = checkpoint_dir / 'checkpoint-final.pt'
        if final_checkpoint.exists():
            return str(final_checkpoint)
        
        raise FileNotFoundError(
            "No checkpoints found in checkpoints/\n"
            "Please train a model first: python training/train.py"
        )
    
    def load_model(self):
        """Load model and tokenizer from checkpoint"""
        print(f"Loading model from: {self.checkpoint_path}")
        
        try:
            # Load checkpoint
            checkpoint = torch.load(self.checkpoint_path, map_location='cpu', weights_only=False)
            
            # Load model
            self.model = load_model_from_config()
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model.to(self.device)
            self.model.eval()
            
            # Store model info
            self.model_info = {
                'checkpoint': self.checkpoint_path,
                'step': checkpoint.get('step', 'unknown'),
                'loss': checkpoint.get('loss', 'unknown'),
                'parameters': sum(p.numel() for p in self.model.parameters()),
                'device': self.device
            }
            
            # Load tokenizer
            tokenizer_path = Path('tokenizer/tokenizer.json')
            if not tokenizer_path.exists():
                raise FileNotFoundError(
                    "Tokenizer not found at tokenizer/tokenizer.json\n"
                    "Please train tokenizer first: python tokenizer/train.py --data data/raw/"
                )
            
            self.tokenizer = Tokenizer.from_file(str(tokenizer_path))
            
            print(f"Model loaded successfully!")
            print(f"  Parameters: {self.model_info['parameters']:,}")
            print(f"  Device: {self.device}")
            print(f"  Step: {self.model_info['step']}")
            
        except FileNotFoundError as e:
            print(f"ERROR: {e}")
            sys.exit(1)
        except Exception as e:
            print(f"ERROR: Failed to load model: {e}")
            import traceback
            traceback.print_exc()
            sys.exit(1)
    
    def generate_response(
        self,
        message: str,
        history: List[Tuple[str, str]],
        temperature: float,
        top_k: int,
        top_p: float,
        max_length: int
    ) -> Tuple[str, List[Tuple[str, str]]]:
        """
        Generate response for user message
        
        Args:
            message: User's message
            history: Conversation history (list of [user_msg, bot_msg] pairs)
            temperature: Sampling temperature
            top_k: Top-k sampling
            top_p: Nucleus sampling
            max_length: Maximum tokens to generate
        
        Returns:
            Tuple of (empty string for input box, updated history)
        """
        if not message.strip():
            return "", history
        
        try:
            # Build context from history
            context_parts = []
            for user_msg, bot_msg in history[-5:]:  # Keep last 5 turns
                context_parts.append(f"User: {user_msg}")
                if bot_msg:
                    context_parts.append(f"Assistant: {bot_msg}")
            
            # Add current message
            context_parts.append(f"User: {message}")
            context_parts.append("Assistant:")
            
            context_text = "\n".join(context_parts)
            
            # Encode
            encoding = self.tokenizer.encode(context_text)
            input_ids = torch.tensor([encoding.ids], dtype=torch.long, device=self.device)
            
            # Trim if too long
            max_context = self.model.config.max_length - max_length
            if input_ids.size(1) > max_context:
                input_ids = input_ids[:, -max_context:]
            
            # Generate
            with torch.no_grad():
                generated_ids = self._generate(
                    input_ids,
                    max_length,
                    temperature,
                    top_k,
                    top_p
                )
            
            # Decode response (only new tokens)
            response_ids = generated_ids[0, input_ids.size(1):].tolist()
            response = self.tokenizer.decode(response_ids)
            
            # Clean up response
            response = response.split("User:")[0].strip()
            response = response.split("Assistant:")[0].strip()
            
            # Remove special tokens
            for stop_word in ["<|endoftext|>", "</s>", "<eos>", "<pad>"]:
                response = response.replace(stop_word, "")
            response = response.strip()
            
            # If response is empty, provide fallback
            if not response:
                response = "I'm not sure how to respond to that."
            
            # Update history
            history.append((message, response))
            
            return "", history
            
        except Exception as e:
            error_msg = f"Error generating response: {str(e)}"
            print(error_msg)
            history.append((message, f"ERROR: {error_msg}"))
            return "", history
    
    def _generate(
        self,
        input_ids: torch.Tensor,
        max_length: int,
        temperature: float,
        top_k: int,
        top_p: float
    ) -> torch.Tensor:
        """Generate tokens autoregressively"""
        for _ in range(max_length):
            # Crop to model's max length if needed
            input_ids_cond = input_ids
            if input_ids.size(1) > self.model.config.max_length:
                input_ids_cond = input_ids[:, -self.model.config.max_length:]
            
            # Forward pass
            outputs = self.model(input_ids_cond)
            logits = outputs['logits']
            
            # Get next token logits
            next_token_logits = logits[0, -1, :] / temperature
            
            # Apply top-k filtering
            if top_k > 0:
                top_k_values, _ = torch.topk(next_token_logits, min(top_k, next_token_logits.size(-1)))
                indices_to_remove = next_token_logits < top_k_values[-1]
                next_token_logits[indices_to_remove] = float('-inf')
            
            # Apply top-p filtering
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
                
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                
                indices_to_remove = sorted_indices[sorted_indices_to_remove]
                next_token_logits[indices_to_remove] = float('-inf')
            
            # Sample
            probs = torch.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            
            # Append to sequence
            input_ids = torch.cat([input_ids, next_token.unsqueeze(0)], dim=1)
            
            # Check for EOS token (assuming token 2 is EOS)
            if next_token.item() == 2:
                break
        
        return input_ids
    
    def create_interface(self):
        """Create Gradio interface with modern dark theme"""
        
        # Custom CSS for dark theme and modern styling
        custom_css = """
        .gradio-container {
            background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%) !important;
        }
        .contain {
            max-width: 900px !important;
            margin: auto !important;
        }
        #greeting-box {
            text-align: center;
            padding: 40px 20px;
            margin-bottom: 30px;
        }
        #greeting-title {
            font-size: 32px;
            font-weight: 600;
            color: #ffffff;
            margin-bottom: 10px;
        }
        #greeting-subtitle {
            font-size: 18px;
            color: #a0a0a0;
            margin-bottom: 20px;
        }
        #model-info {
            font-size: 14px;
            color: #808080;
            margin-top: 10px;
        }
        .prompt-card {
            background: rgba(255, 255, 255, 0.05) !important;
            border: 1px solid rgba(255, 255, 255, 0.1) !important;
            border-radius: 12px !important;
            padding: 16px !important;
            cursor: pointer !important;
            transition: all 0.3s ease !important;
        }
        .prompt-card:hover {
            background: rgba(255, 255, 255, 0.08) !important;
            border-color: rgba(76, 175, 80, 0.5) !important;
        }
        #chatbot {
            border: 1px solid rgba(255, 255, 255, 0.1) !important;
            border-radius: 16px !important;
            background: rgba(0, 0, 0, 0.2) !important;
        }
        #msg-input {
            border-radius: 24px !important;
            border: 1px solid rgba(255, 255, 255, 0.2) !important;
            background: rgba(255, 255, 255, 0.05) !important;
            color: #ffffff !important;
        }
        .settings-panel {
            background: rgba(255, 255, 255, 0.03) !important;
            border-radius: 12px !important;
            padding: 20px !important;
            margin-top: 20px !important;
        }
        """
        
        with gr.Blocks(
            title="Chat with Your Model",
            theme=gr.themes.Soft(
                primary_hue="green",
                secondary_hue="blue",
                neutral_hue="slate",
            ),
            css=custom_css
        ) as interface:
            
            # Greeting section (shown when no conversation)
            with gr.Column(elem_id="greeting-box"):
                gr.HTML("""
                    <div style="text-align: center;">
                        <div style="width: 80px; height: 80px; margin: 0 auto 20px; 
                                    background: linear-gradient(135deg, #4CAF50 0%, #45a049 100%); 
                                    border-radius: 50%; display: flex; align-items: center; 
                                    justify-content: center; box-shadow: 0 4px 20px rgba(76, 175, 80, 0.4);">
                            <span style="font-size: 40px;">🤖</span>
                        </div>
                        <h1 id="greeting-title">Good evening!</h1>
                        <p id="greeting-subtitle">How can I help you today?</p>
                    </div>
                """)
                
                gr.Markdown(
                    f'<p id="model-info">Model: {Path(self.checkpoint_path).name} | '
                    f'Step: {self.model_info.get("step", "unknown")} | '
                    f'Parameters: {self.model_info.get("parameters", 0):,} | '
                    f'Device: {self.device}</p>',
                    elem_id="model-info"
                )
            
            # Prompt suggestions
            with gr.Row():
                prompt1 = gr.Button(
                    "💡 Get creative ideas\nBrainstorm innovative solutions",
                    elem_classes="prompt-card",
                    scale=1
                )
                prompt2 = gr.Button(
                    "✍️ Write content\nGenerate articles or stories",
                    elem_classes="prompt-card",
                    scale=1
                )
                prompt3 = gr.Button(
                    "🔍 Analyze text\nSummarize or explain concepts",
                    elem_classes="prompt-card",
                    scale=1
                )
                prompt4 = gr.Button(
                    "💻 Code assistance\nHelp with programming tasks",
                    elem_classes="prompt-card",
                    scale=1
                )
            
            # Chat interface
            chatbot = gr.Chatbot(
                label="",
                height=450,
                show_copy_button=True,
                type="tuples",
                elem_id="chatbot",
                avatar_images=(None, "🤖"),
                bubble_full_width=False
            )
            
            # Input area
            with gr.Row():
                msg = gr.Textbox(
                    label="",
                    placeholder="Ask me anything...",
                    lines=1,
                    scale=9,
                    elem_id="msg-input",
                    show_label=False
                )
                submit = gr.Button("➤", scale=1, variant="primary", size="lg")
            
            # Settings in collapsible panel
            with gr.Accordion("⚙️ Generation Settings", open=False, elem_classes="settings-panel"):
                with gr.Row():
                    temperature = gr.Slider(
                        minimum=0.1,
                        maximum=2.0,
                        value=0.8,
                        step=0.1,
                        label="🌡️ Temperature",
                        info="Higher = more creative"
                    )
                    max_length = gr.Slider(
                        minimum=10,
                        maximum=500,
                        value=100,
                        step=10,
                        label="📏 Max Length",
                        info="Maximum tokens"
                    )
                
                with gr.Row():
                    top_k = gr.Slider(
                        minimum=1,
                        maximum=100,
                        value=50,
                        step=1,
                        label="🎯 Top-K",
                        info="Diversity control"
                    )
                    top_p = gr.Slider(
                        minimum=0.1,
                        maximum=1.0,
                        value=0.95,
                        step=0.05,
                        label="🎲 Top-P",
                        info="Nucleus sampling"
                    )
                
                with gr.Row():
                    clear = gr.Button("🗑️ Clear Conversation", variant="secondary")
                    regenerate = gr.Button("🔄 Regenerate", variant="secondary")
            
            # Helper function for prompt buttons
            def use_prompt(prompt_text):
                return prompt_text.split("\n")[0].replace("💡 ", "").replace("✍️ ", "").replace("🔍 ", "").replace("💻 ", "")
            
            # Event handlers
            prompt1.click(
                lambda: "Help me brainstorm creative ideas",
                outputs=msg
            )
            prompt2.click(
                lambda: "Write a short story about",
                outputs=msg
            )
            prompt3.click(
                lambda: "Explain the concept of",
                outputs=msg
            )
            prompt4.click(
                lambda: "Help me write code for",
                outputs=msg
            )
            
            submit.click(
                self.generate_response,
                inputs=[msg, chatbot, temperature, top_k, top_p, max_length],
                outputs=[msg, chatbot]
            )
            
            msg.submit(
                self.generate_response,
                inputs=[msg, chatbot, temperature, top_k, top_p, max_length],
                outputs=[msg, chatbot]
            )
            
            clear.click(lambda: None, None, chatbot, queue=False)
        
        return interface
    
    def launch(self, share: bool = False, server_port: int = 7860):
        """
        Launch Gradio interface
        
        Args:
            share: Create public shareable link
            server_port: Port to run server on
        """
        interface = self.create_interface()
        
        print("\n" + "=" * 60)
        print("Launching Chat Interface...")
        print("=" * 60)
        print(f"Opening in browser at http://localhost:{server_port}")
        print("Press Ctrl+C to stop the server")
        print("=" * 60 + "\n")
        
        try:
            interface.launch(
                server_name="127.0.0.1",
                server_port=server_port,
                share=share,
                show_error=True
            )
        except OSError as e:
            if "address already in use" in str(e).lower():
                print(f"Port {server_port} is already in use. Trying port {server_port + 1}...")
                interface.launch(
                    server_name="127.0.0.1",
                    server_port=server_port + 1,
                    share=share,
                    show_error=True
                )
            else:
                raise


def main():
    """Main function for standalone usage"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Launch chat interface for trained model')
    parser.add_argument(
        '--checkpoint',
        type=str,
        help='Path to checkpoint (default: auto-detect best)'
    )
    parser.add_argument(
        '--share',
        action='store_true',
        help='Create public shareable link'
    )
    parser.add_argument(
        '--port',
        type=int,
        default=7860,
        help='Server port (default: 7860)'
    )
    
    args = parser.parse_args()
    
    try:
        chat = ChatInterface(checkpoint_path=args.checkpoint)
        chat.load_model()
        chat.launch(share=args.share, server_port=args.port)
    except KeyboardInterrupt:
        print("\n\nChat interface stopped.")
    except Exception as e:
        print(f"\nERROR: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()
