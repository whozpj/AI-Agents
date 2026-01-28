 #!/usr/bin/env python3
"""
Simple MMLU Bar Graph Generator

Creates a single bar chart comparing model accuracy.

Usage:
    python simple_bar_graph.py file1.json file2.json [file3.json ...]

Example:
    python simple_bar_graph.py llama_results.json qwen_results.json

Requirements:
    pip install matplotlib numpy
"""

import json
import matplotlib.pyplot as plt
import numpy as np
import sys


def load_results(filepaths):
    """Load all result JSON files"""
    results = []
    for filepath in filepaths:
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
                results.append(data)
                print(f"✓ Loaded: {filepath}")
        except FileNotFoundError:
            print(f"❌ File not found: {filepath}")
        except json.JSONDecodeError:
            print(f"❌ Invalid JSON: {filepath}")
    return results


def get_model_name(result):
    """Extract a short model name for display"""
    model = result.get('model', 'Unknown')
    if 'Llama-3.2-1B' in model:
        return 'Llama-3.2-1B'
    elif 'Qwen2.5-0.5B' in model:
        return 'Qwen2.5-0.5B'
    elif 'Mistral-7B' in model:
        return 'Mistral-7B'
    else:
        return model.split('/')[-1][:20]


def create_bar_graph(results):
    """Create a simple bar graph of model accuracy"""
    model_names = [get_model_name(r) for r in results]
    accuracies = [r['overall_accuracy'] for r in results]
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Create bars with colors
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A', '#98D8C8']
    bars = ax.bar(model_names, accuracies, color=colors[:len(results)], 
                  alpha=0.8, edgecolor='black', linewidth=1.5)
    
    # Add value labels on top of bars
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}%',
                ha='center', va='bottom', fontweight='bold', fontsize=12)
    
    # Styling
    ax.set_ylabel('Accuracy (%)', fontsize=13, fontweight='bold')
    ax.set_title('MMLU Benchmark - Model Comparison', fontsize=15, fontweight='bold', pad=20)
    ax.set_ylim([0, 100])
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.set_axisbelow(True)
    
    # Rotate x-axis labels if needed
    if len(model_names) > 3:
        plt.xticks(rotation=45, ha='right')
    
    plt.tight_layout()
    
    # Save
    output_file = 'mmlu_comparison.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\n✓ Graph saved to: {output_file}")
    
    plt.show()


def main():
    """Main function"""
    print("\n" + "="*60)
    print("MMLU Simple Bar Graph Generator")
    print("="*60 + "\n")
    
    if len(sys.argv) < 2:
        print("❌ No result files specified!")
        print("\nUsage:")
        print("  python simple_bar_graph.py file1.json file2.json")
        print("\nExample:")
        print("  python simple_bar_graph.py llama_results.json qwen_results.json")
        sys.exit(1)
    
    filepaths = sys.argv[1:]
    
    results = load_results(filepaths)
    
    if len(results) == 0:
        print("❌ No valid result files loaded!")
        sys.exit(1)
    
    print(f"\nCreating bar graph for {len(results)} model(s)...\n")
    create_bar_graph(results)
    
    print("\n✅ Done!\n")


if __name__ == "__main__":
    main()