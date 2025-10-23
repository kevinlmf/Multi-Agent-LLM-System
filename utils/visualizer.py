"""
Visualization utilities for reasoning flow analysis.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from typing import Dict, Any, List
from datetime import datetime
import json


def visualize_reasoning_flow(result: Dict[str, Any], output_path: str = "reasoning_flow.png"):
    """
    Create a visual timeline of the reasoning process.

    Args:
        result: Result dictionary from ReasoningGraph.reason()
        output_path: Path to save the visualization
    """
    history = result["reasoning_history"]

    if not history:
        print("⚠️  No reasoning history to visualize")
        return

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), gridspec_kw={'height_ratios': [3, 1]})

    # Color scheme
    colors = {
        "reasoner": "#3498db",  # Blue
        "critic": "#e74c3c",    # Red
        "refiner": "#2ecc71"    # Green
    }

    # Timeline visualization
    y_positions = {"reasoner": 3, "critic": 2, "refiner": 1}

    for i, step in enumerate(history):
        role = step["role"]
        y = y_positions.get(role, 0)

        # Plot point
        ax1.scatter(i, y, s=300, c=colors.get(role, "gray"), alpha=0.7, edgecolors="black", linewidth=2)

        # Add text
        content_preview = step["content"][:50] + "..." if len(step["content"]) > 50 else step["content"]
        ax1.text(i, y - 0.3, f"Step {i+1}", ha='center', fontsize=9, weight='bold')

        # Add confidence if available
        if "confidence_score" in step.get("metadata", {}):
            confidence = step["metadata"]["confidence_score"]
            ax1.text(i, y + 0.3, f"Conf: {confidence:.2f}", ha='center', fontsize=8, style='italic')

    # Connect points
    for i in range(len(history) - 1):
        role1 = history[i]["role"]
        role2 = history[i + 1]["role"]
        y1 = y_positions.get(role1, 0)
        y2 = y_positions.get(role2, 0)
        ax1.plot([i, i + 1], [y1, y2], 'k--', alpha=0.3, linewidth=1)

    # Styling
    ax1.set_yticks([1, 2, 3])
    ax1.set_yticklabels(["Refiner", "Critic", "Reasoner"])
    ax1.set_xlabel("Step Number", fontsize=12, weight='bold')
    ax1.set_ylabel("Agent Role", fontsize=12, weight='bold')
    ax1.set_title(f"Reasoning Flow Timeline\nQuestion: {result['question'][:80]}...",
                  fontsize=14, weight='bold', pad=20)
    ax1.grid(axis='x', alpha=0.3, linestyle='--')
    ax1.set_xlim(-0.5, len(history) - 0.5)
    ax1.set_ylim(0.5, 3.5)

    # Legend
    legend_elements = [
        mpatches.Patch(color=colors["reasoner"], label="Reasoner (generates reasoning)"),
        mpatches.Patch(color=colors["critic"], label="Critic (evaluates quality)"),
        mpatches.Patch(color=colors["refiner"], label="Refiner (synthesizes answer)")
    ]
    ax1.legend(handles=legend_elements, loc='upper right', fontsize=10)

    # Confidence progression (bottom plot)
    confidence_scores = []
    confidence_steps = []

    for i, step in enumerate(history):
        if "confidence_score" in step.get("metadata", {}):
            confidence_scores.append(step["metadata"]["confidence_score"])
            confidence_steps.append(i)

    if confidence_scores:
        ax2.plot(confidence_steps, confidence_scores, marker='o', linewidth=2,
                markersize=8, color='#9b59b6', label='Confidence Score')
        ax2.axhline(y=0.9, color='green', linestyle='--', linewidth=2, alpha=0.5, label='Target (0.9)')
        ax2.fill_between(confidence_steps, confidence_scores, alpha=0.3, color='#9b59b6')

        ax2.set_xlabel("Step Number", fontsize=11, weight='bold')
        ax2.set_ylabel("Confidence", fontsize=11, weight='bold')
        ax2.set_title("Confidence Progression", fontsize=12, weight='bold')
        ax2.set_ylim(0, 1.05)
        ax2.grid(alpha=0.3)
        ax2.legend(fontsize=9)
    else:
        ax2.text(0.5, 0.5, "No confidence scores available",
                ha='center', va='center', fontsize=12, transform=ax2.transAxes)
        ax2.set_xticks([])
        ax2.set_yticks([])

    # Summary text
    summary_text = f"""
    Final Confidence: {result['confidence_score']:.2f}
    Total Iterations: {result['iterations']}
    Total Steps: {len(history)}
    """
    fig.text(0.02, 0.02, summary_text, fontsize=10, family='monospace',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✅ Reasoning flow visualization saved to: {output_path}")

    try:
        plt.show()
    except:
        pass


def create_reasoning_report(result: Dict[str, Any], output_path: str = "reasoning_report.html"):
    """
    Create an HTML report of the reasoning process.

    Args:
        result: Result dictionary from ReasoningGraph.reason()
        output_path: Path to save the HTML report
    """
    html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>Reasoning Report</title>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background-color: #f5f5f5;
        }}
        .header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 30px;
            border-radius: 10px;
            margin-bottom: 30px;
        }}
        .summary {{
            background: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            margin-bottom: 20px;
        }}
        .question {{
            background: #e3f2fd;
            padding: 20px;
            border-left: 5px solid #2196f3;
            border-radius: 5px;
            margin-bottom: 20px;
        }}
        .answer {{
            background: #e8f5e9;
            padding: 20px;
            border-left: 5px solid #4caf50;
            border-radius: 5px;
            margin-bottom: 20px;
        }}
        .step {{
            background: white;
            padding: 20px;
            margin-bottom: 15px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        .step-header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 10px;
            padding-bottom: 10px;
            border-bottom: 2px solid #f0f0f0;
        }}
        .role {{
            font-weight: bold;
            padding: 5px 15px;
            border-radius: 20px;
            text-transform: uppercase;
            font-size: 12px;
        }}
        .role-reasoner {{ background: #3498db; color: white; }}
        .role-critic {{ background: #e74c3c; color: white; }}
        .role-refiner {{ background: #2ecc71; color: white; }}
        .content {{
            line-height: 1.6;
            white-space: pre-wrap;
            font-size: 14px;
        }}
        .timestamp {{
            color: #7f8c8d;
            font-size: 12px;
        }}
        .confidence {{
            background: #9b59b6;
            color: white;
            padding: 5px 10px;
            border-radius: 15px;
            font-size: 12px;
        }}
        .metric {{
            display: inline-block;
            margin: 10px 15px 10px 0;
            padding: 10px 20px;
            background: #ecf0f1;
            border-radius: 5px;
        }}
        .metric-label {{
            font-size: 12px;
            color: #7f8c8d;
            text-transform: uppercase;
        }}
        .metric-value {{
            font-size: 24px;
            font-weight: bold;
            color: #2c3e50;
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>🧠 Multi-Agent Reasoning Report</h1>
        <p>Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
    </div>

    <div class="summary">
        <h2>📊 Summary</h2>
        <div class="metric">
            <div class="metric-label">Iterations</div>
            <div class="metric-value">{result['iterations']}</div>
        </div>
        <div class="metric">
            <div class="metric-label">Confidence</div>
            <div class="metric-value">{result['confidence_score']:.2f}</div>
        </div>
        <div class="metric">
            <div class="metric-label">Total Steps</div>
            <div class="metric-value">{len(result['reasoning_history'])}</div>
        </div>
    </div>

    <div class="question">
        <h2>📝 Question</h2>
        <p>{result['question']}</p>
    </div>

    <div class="answer">
        <h2>🎯 Final Answer</h2>
        <div class="content">{result['final_answer']}</div>
    </div>

    <h2>📚 Reasoning History</h2>
"""

    for i, step in enumerate(result['reasoning_history'], 1):
        role = step['role']
        content = step['content']
        timestamp = step.get('timestamp', 'N/A')
        metadata = step.get('metadata', {})

        confidence_badge = ""
        if 'confidence_score' in metadata:
            conf = metadata['confidence_score']
            confidence_badge = f'<span class="confidence">Confidence: {conf:.2f}</span>'

        html_content += f"""
    <div class="step">
        <div class="step-header">
            <div>
                <span class="role role-{role}">{role}</span>
                <span class="timestamp">Step {i} • {timestamp}</span>
            </div>
            {confidence_badge}
        </div>
        <div class="content">{content}</div>
    </div>
"""

    html_content += """
</body>
</html>
"""

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html_content)

    print(f"✅ Reasoning report saved to: {output_path}")


def export_to_json(result: Dict[str, Any], output_path: str = "reasoning_result.json"):
    """
    Export reasoning result to JSON format.

    Args:
        result: Result dictionary from ReasoningGraph.reason()
        output_path: Path to save the JSON file
    """
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    print(f"✅ Result exported to JSON: {output_path}")
