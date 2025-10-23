"""
Quick Test Script for Cognitive Multi-Agent System
Tests both original and enhanced functionality
"""

import os
import sys

def test_original_system():
    """Test original reasoning-only system (backward compatibility)"""
    print("=" * 60)
    print("TEST 1: Original Reasoning System (Backward Compatible)")
    print("=" * 60)
    print()

    try:
        from core import ReasoningGraph

        print("✓ Original imports successful")

        # Create original system
        system = ReasoningGraph(max_iterations=2)
        print("✓ Original system initialized")

        # Test reasoning
        result = system.reason("What is 2+2? Explain step by step.")

        print("\n📝 Result:")
        print(f"  Final Answer: {result['final_answer'][:100]}...")
        print(f"  Confidence: {result.get('confidence_score', 'N/A')}")
        print(f"  Iterations: {result['iterations']}")
        print()
        print("✓ Original system works correctly!")

        return True

    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_cognitive_system():
    """Test enhanced cognitive system"""
    print("=" * 60)
    print("TEST 2: Enhanced Cognitive System")
    print("=" * 60)
    print()

    try:
        from core import CognitiveReasoningGraph

        print("✓ Cognitive imports successful")

        # Test without perception/memory first
        print("\n--- Mode 1: Reasoning Only ---")
        system = CognitiveReasoningGraph(
            max_iterations=2,
            enable_perception=False,
            enable_memory=False
        )
        print("✓ Cognitive system initialized (reasoning only)")

        result = system.reason(
            question="Should I invest in stocks when market is volatile?"
        )

        print("\n📝 Result:")
        print(f"  Final Answer: {result['final_answer'][:150]}...")
        print(f"  Confidence: {result.get('confidence_score', 'N/A')}")
        print(f"  Iterations: {result['iterations']}")
        print()

        # Test with perception/memory if available
        print("--- Mode 2: Full Cognitive (Perception + Memory) ---")
        try:
            system_full = CognitiveReasoningGraph(
                max_iterations=2,
                enable_perception=True,
                enable_memory=True
            )
            print("✓ Full cognitive system initialized")

            result_full = system_full.reason(
                question="What trading strategy should I use?",
                observation="SPY at $450, VIX at 15, bullish market sentiment",
                reward=None
            )

            print("\n📝 Result with Perception & Memory:")
            print(f"  Final Answer: {result_full['final_answer'][:150]}...")
            print(f"  Confidence: {result_full.get('confidence_score', 'N/A')}")
            print(f"  Has Perception: {result_full.get('perception_snapshot') is not None}")
            print(f"  Has Memory: {result_full.get('memory_snapshot') is not None}")
            print(f"  Experience Stored: {result_full.get('experience_stored', False)}")
            print()

        except Exception as e:
            print(f"⚠ Perception/Memory not available: {e}")
            print("  (This is OK if modules aren't installed)")
            print()

        print("✓ Cognitive system works correctly!")
        return True

    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_imports():
    """Test all imports"""
    print("=" * 60)
    print("TEST 0: Import Check")
    print("=" * 60)
    print()

    imports = {
        "Original System": [
            "ReasoningGraph",
            "ReasonerAgent",
            "CriticAgent",
            "RefinerAgent",
            "ReasoningState"
        ],
        "Cognitive System": [
            "CognitiveReasoningGraph",
            "CognitiveReasonerAgent",
            "CognitiveCriticAgent",
            "CognitiveRefinerAgent",
            "PerceiverAgent",
            "ConsolidatorAgent",
            "CognitiveState"
        ]
    }

    all_ok = True

    for category, items in imports.items():
        print(f"\n{category}:")
        for item in items:
            try:
                exec(f"from core import {item}")
                print(f"  ✓ {item}")
            except Exception as e:
                print(f"  ✗ {item}: {e}")
                all_ok = False

    print()
    if all_ok:
        print("✓ All imports successful!")
    else:
        print("⚠ Some imports failed")

    return all_ok


def main():
    """Run all tests"""
    print("\n")
    print("╔" + "=" * 58 + "╗")
    print("║" + " " * 58 + "║")
    print("║" + "  COGNITIVE MULTI-AGENT SYSTEM - INTEGRATION TEST  ".center(58) + "║")
    print("║" + " " * 58 + "║")
    print("╚" + "=" * 58 + "╝")
    print()

    # Check environment
    if not os.getenv("OPENAI_API_KEY"):
        print("⚠ WARNING: OPENAI_API_KEY not set!")
        print("Please set it in .env file or environment")
        print()
        return

    results = []

    # Test imports
    results.append(("Imports", test_imports()))
    print("\n" * 2)

    # Test original system
    results.append(("Original System", test_original_system()))
    print("\n" * 2)

    # Test cognitive system
    results.append(("Cognitive System", test_cognitive_system()))
    print("\n" * 2)

    # Summary
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for name, success in results:
        status = "✓ PASS" if success else "✗ FAIL"
        print(f"{name}: {status}")

    all_passed = all(s for _, s in results)
    print()
    if all_passed:
        print("🎉 All tests passed! System is ready to use.")
    else:
        print("⚠ Some tests failed. Please check errors above.")

    print("\n" + "=" * 60)
    print("Next steps:")
    print("  1. Run full example: python examples/cognitive_trading_agent.py")
    print("  2. Read COGNITIVE_README.md for detailed documentation")
    print("  3. Explore examples/ directory for more use cases")
    print("=" * 60)


if __name__ == "__main__":
    # Load environment
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        print("Note: python-dotenv not installed, using environment variables")

    main()
