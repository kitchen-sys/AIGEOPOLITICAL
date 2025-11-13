"""
GeoBotv1 Example 09: Answer Quality Database & Training Dataset Generation

Demonstrates the answer feedback loop for continuous improvement:
1. Automatic logging of all Q&A interactions
2. Quality assessment (automatic + human feedback)
3. Separation of good/bad answers
4. Training dataset export for model fine-tuning
5. RLHF (Reinforcement Learning from Human Feedback) preparation

This creates a continuous improvement cycle:
Production Queries → Quality Database → Training Datasets → Fine-tuned Models → Better Production Performance
"""

import sys
from pathlib import Path
from datetime import datetime, timedelta

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from geobot.interface import (
    AnalystAgent,
    AnswerDatabase,
    AnswerQuality,
    AnswerRecord
)


def example_1_automatic_logging():
    """Example 1: Automatic logging of analyst responses."""
    print("="*80)
    print("EXAMPLE 1: Automatic Answer Logging")
    print("="*80)

    # Create analyst with automatic logging enabled
    agent = AnalystAgent(
        llm_backend="mistral-7b",
        enable_answer_logging=True,
        answer_db_path="geobot_answers.db"
    )

    # Run several queries - all automatically logged
    queries = [
        "What is the risk of conflict spreading from Syria to Lebanon in next 30 days?",
        "How would new sanctions on Iran affect regional stability?",
        "Analyze Russia-Ukraine conflict trajectory over next quarter",
        "What caused the recent escalation in Gaza?",
        "Compare current Iran nuclear situation to 2015 JCPOA baseline"
    ]

    print("\nRunning queries (all automatically logged)...\n")

    for i, query in enumerate(queries, 1):
        print(f"{i}. {query}")
        result = agent.analyze(
            query,
            session_id="demo_session_001",
            analyst_id="analyst_123"
        )
        print(f"   ✓ Answer logged (confidence: {result.confidence:.0%})")
        print(f"   Modules used: {', '.join(result.modules_used)}\n")

    # Access database directly
    db = agent.answer_db
    stats = db.get_statistics()

    print("\n" + "="*80)
    print("Database Statistics:")
    print("="*80)
    print(f"Total answers logged: {stats['total_answers']}")
    print(f"Auto-rated answers: {stats['rated_answers']}")
    print(f"Average quality score: {stats['average_quality_score']}")
    print(f"Good answers (>0.7): {stats['good_answers']}")
    print(f"Bad answers (<0.4): {stats['bad_answers']}")
    print(f"Acceptable answers: {stats['acceptable_answers']}")

    return db


def example_2_manual_quality_rating():
    """Example 2: Manual quality ratings with detailed feedback."""
    print("\n\n" + "="*80)
    print("EXAMPLE 2: Manual Quality Rating")
    print("="*80)

    # Reuse database from example 1
    db = AnswerDatabase("geobot_answers.db")

    # Get a recent answer
    cursor = db.conn.cursor()
    cursor.execute('''
        SELECT record_id, query, answer_narrative
        FROM answers
        ORDER BY query_timestamp DESC
        LIMIT 1
    ''')
    record_id, query, answer = cursor.fetchone()

    print(f"\nRecord ID: {record_id}")
    print(f"Query: {query}")
    print(f"Answer (first 200 chars): {answer[:200]}...")

    # Human analyst provides detailed quality rating
    print("\n📝 Analyst Quality Assessment:")

    quality = AnswerQuality(
        factual_accuracy=0.90,  # Claims verified against intelligence sources
        analytical_rigor=0.85,  # Strong use of Hawkes + VAR models
        source_reliability=0.80,  # Mixed OSINT and model-based
        completeness=0.88,  # Addressed all aspects of question
        clarity=0.92,  # Well-structured, clear language
        analyst_usefulness=0.95,  # Very actionable for decision-makers
        actionability=0.87,  # Clear scenarios and recommendations
        confidence_appropriate=0.90,  # Uncertainty well-calibrated
        strengths=[
            "Strong mathematical foundation (Hawkes + VAR)",
            "Clear scenario differentiation",
            "Appropriate confidence intervals",
            "Actionable indicators provided"
        ],
        weaknesses=[
            "Could include more recent OSINT citations",
            "Economic sanctions impact underexplored",
            "Regional power dynamics (Turkey, Saudi) not fully addressed"
        ],
        feedback_notes="Excellent technical analysis. For next iteration, integrate more HUMINT/SIGINT if available on regional ally intentions.",
        rated_by="senior_analyst_jane",
        rated_at=datetime.utcnow()
    )

    # Save rating
    db.rate_answer(record_id, quality)

    print(f"\n✓ Quality Rating Saved:")
    print(f"  Overall Score: {quality.overall_score:.2f}")
    print(f"  Category: {quality.quality_category()}")
    print(f"\n  Strengths:")
    for s in quality.strengths:
        print(f"    • {s}")
    print(f"\n  Weaknesses:")
    for w in quality.weaknesses:
        print(f"    • {w}")
    print(f"\n  Feedback: {quality.feedback_notes}")

    return db


def example_3_good_vs_bad_datasets():
    """Example 3: Separate good and bad answers for training."""
    print("\n\n" + "="*80)
    print("EXAMPLE 3: Good vs Bad Answer Separation")
    print("="*80)

    db = AnswerDatabase("geobot_answers.db")

    # Get good answers (quality >= 0.7)
    good_answers = db.get_good_answers(threshold=0.7, limit=10)
    print(f"\n✅ GOOD ANSWERS (quality >= 0.7): {len(good_answers)} found")

    if good_answers:
        print("\nTop 3 highest-quality answers:")
        for i, record in enumerate(good_answers[:3], 1):
            print(f"\n{i}. Quality: {record.quality.overall_score:.2f} | Category: {record.quality.quality_category()}")
            print(f"   Query: {record.query[:80]}...")
            print(f"   Modules: {', '.join(record.modules_invoked)}")

    # Get bad answers (quality < 0.4)
    bad_answers = db.get_bad_answers(threshold=0.4, limit=10)
    print(f"\n\n❌ BAD ANSWERS (quality < 0.4): {len(bad_answers)} found")

    if bad_answers:
        print("\nLowest-quality answers:")
        for i, record in enumerate(bad_answers[:3], 1):
            print(f"\n{i}. Quality: {record.quality.overall_score:.2f} | Category: {record.quality.quality_category()}")
            print(f"   Query: {record.query[:80]}...")
            if record.quality and record.quality.weaknesses:
                print(f"   Key weakness: {record.quality.weaknesses[0]}")

    print(f"\n\n📊 Dataset Composition:")
    stats = db.get_statistics()
    print(f"  Good answers: {stats['good_answers']} ({stats['good_answer_percentage']:.1f}%)")
    print(f"  Acceptable: {stats['acceptable_answers']}")
    print(f"  Poor answers: {stats['bad_answers']}")

    return good_answers, bad_answers


def example_4_export_training_datasets():
    """Example 4: Export training datasets in multiple formats."""
    print("\n\n" + "="*80)
    print("EXAMPLE 4: Export Training Datasets")
    print("="*80)

    db = AnswerDatabase("geobot_answers.db")
    output_dir = Path(__file__).parent.parent / "training_data"
    output_dir.mkdir(exist_ok=True)

    # Format 1: ChatML format (for fine-tuning chat models)
    print("\n📦 Exporting ChatML format (GPT/Mistral fine-tuning)...")
    chat_dataset = db.export_training_dataset(
        quality_threshold=0.7,
        format="chat",
        output_path=str(output_dir / "geobot_chat_dataset.json"),
        include_metadata=True
    )
    print(f"   ✓ Exported {len(chat_dataset)} examples to geobot_chat_dataset.json")

    if chat_dataset:
        print(f"\n   Sample ChatML example:")
        import json
        print(json.dumps(chat_dataset[0], indent=2))

    # Format 2: Completion format (for base model fine-tuning)
    print("\n\n📦 Exporting Completion format...")
    completion_dataset = db.export_training_dataset(
        quality_threshold=0.7,
        format="completion",
        output_path=str(output_dir / "geobot_completion_dataset.json")
    )
    print(f"   ✓ Exported {len(completion_dataset)} examples to geobot_completion_dataset.json")

    # Format 3: RLHF format (for reward model training)
    print("\n\n📦 Exporting RLHF format (reward model training)...")
    rlhf_dataset = db.export_training_dataset(
        quality_threshold=0.5,  # Include wider range for ranking
        format="rlhf",
        output_path=str(output_dir / "geobot_rlhf_dataset.json")
    )
    print(f"   ✓ Exported {len(rlhf_dataset)} examples to geobot_rlhf_dataset.json")

    print(f"\n\n✅ All datasets exported to: {output_dir}")

    return chat_dataset


def example_5_preference_pairs_for_rlhf():
    """Example 5: Generate preference pairs for RLHF/DPO training."""
    print("\n\n" + "="*80)
    print("EXAMPLE 5: Preference Pairs for RLHF/DPO")
    print("="*80)
    print("\nRLHF (Reinforcement Learning from Human Feedback) requires preference pairs:")
    print("For each query, we need a 'chosen' (good) and 'rejected' (bad) answer.")

    db = AnswerDatabase("geobot_answers.db")
    output_dir = Path(__file__).parent.parent / "training_data"

    # Generate preference pairs
    pairs = db.export_preference_pairs(
        good_threshold=0.7,
        bad_threshold=0.4,
        output_path=str(output_dir / "geobot_preference_pairs.json")
    )

    print(f"\n✓ Generated {len(pairs)} preference pairs")

    if pairs:
        print("\nSample preference pair:")
        print("-" * 80)
        sample = pairs[0]
        print(f"Query: {sample['prompt']}")
        print(f"\nChosen (score: {sample['chosen_score']:.2f}):")
        print(f"  {sample['chosen'][:150]}...")
        print(f"\nRejected (score: {sample['rejected_score']:.2f}):")
        print(f"  {sample['rejected'][:150]}...")
        print(f"\nScore Margin: {sample['score_margin']:.2f}")

    print(f"\n\n✅ Preference pairs saved to: {output_dir / 'geobot_preference_pairs.json'}")
    print("\n💡 Use these for:")
    print("   • DPO (Direct Preference Optimization)")
    print("   • RLHF reward model training")
    print("   • Pairwise ranking models")

    return pairs


def example_6_continuous_improvement_workflow():
    """Example 6: Complete continuous improvement workflow."""
    print("\n\n" + "="*80)
    print("EXAMPLE 6: Continuous Improvement Workflow")
    print("="*80)

    print("""
🔄 CONTINUOUS IMPROVEMENT CYCLE
═══════════════════════════════════════════════════════════════════════════════

Phase 1: PRODUCTION DEPLOYMENT
  │
  ├─► AnalystAgent with answer_logging=True
  ├─► All Q&A interactions automatically logged
  └─► Auto-rating provides initial quality scores

Phase 2: HUMAN FEEDBACK
  │
  ├─► Analysts review random sample of answers
  ├─► Provide detailed quality ratings
  ├─► Flag exceptional (good) and problematic (bad) responses
  └─► Add strengths/weaknesses/feedback notes

Phase 3: DATASET CURATION
  │
  ├─► Export good answers (quality >= 0.7) → Training set
  ├─► Export bad answers (quality < 0.4) → Negative examples
  ├─► Generate preference pairs → RLHF training
  └─► Create test set from recent queries

Phase 4: MODEL FINE-TUNING
  │
  ├─► Supervised Fine-Tuning (SFT) on good examples
  ├─► RLHF/DPO using preference pairs
  ├─► Continual learning on domain-specific geopolitical Q&A
  └─► A/B testing: baseline vs fine-tuned model

Phase 5: DEPLOYMENT & MONITORING
  │
  ├─► Deploy fine-tuned model to production
  ├─► Monitor quality scores (should improve over time)
  ├─► Compare to baseline (% good answers should increase)
  └─► Return to Phase 1 (continue cycle)

═══════════════════════════════════════════════════════════════════════════════
""")

    # Demonstrate workflow metrics
    db = AnswerDatabase("geobot_answers.db")
    stats = db.get_statistics()

    print("\n📊 CURRENT SYSTEM METRICS:")
    print(f"  • Total Q&A interactions: {stats['total_answers']}")
    print(f"  • Human-rated answers: {stats['rated_answers']}")
    print(f"  • Average quality: {stats['average_quality_score']:.2f}/1.00")
    print(f"  • Good answer rate: {stats['good_answer_percentage']:.1f}%")
    print(f"\n  • Available for training:")
    print(f"    - Good examples: {stats['good_answers']}")
    print(f"    - Bad examples: {stats['bad_answers']}")

    print(f"\n\n🎯 NEXT STEPS:")
    print(f"  1. Review {stats['unrated_answers']} unrated answers")
    print(f"  2. Export {stats['good_answers']} good answers for fine-tuning")
    print(f"  3. Fine-tune model on geopolitical domain")
    print(f"  4. Deploy and measure improvement in good_answer_percentage")
    print(f"  5. Iterate monthly or quarterly")


def example_7_quality_analysis():
    """Example 7: Analyze quality patterns and failure modes."""
    print("\n\n" + "="*80)
    print("EXAMPLE 7: Quality Analysis & Failure Modes")
    print("="*80)

    db = AnswerDatabase("geobot_answers.db")

    # Analyze by module usage
    print("\n📈 QUALITY BY MODULE USAGE:")
    cursor = db.conn.cursor()

    cursor.execute('''
        SELECT a.modules_invoked, AVG(q.overall_score) as avg_score, COUNT(*) as count
        FROM answers a
        INNER JOIN quality_ratings q ON a.record_id = q.record_id
        GROUP BY a.modules_invoked
        HAVING count >= 1
        ORDER BY avg_score DESC
    ''')

    print("\n  Module Combination → Avg Quality")
    print("  " + "-"*60)
    for modules_json, avg_score, count in cursor.fetchall():
        import json
        modules = json.loads(modules_json)
        modules_str = ", ".join(modules) if modules else "none"
        print(f"  {modules_str[:40]:40} → {avg_score:.2f} ({count} answers)")

    # Analyze by analysis type
    print("\n\n📊 QUALITY BY ANALYSIS TYPE:")
    cursor.execute('''
        SELECT a.query_intent, AVG(q.overall_score) as avg_score, COUNT(*) as count
        FROM answers a
        INNER JOIN quality_ratings q ON a.record_id = q.record_id
        GROUP BY a.query_intent
        HAVING count >= 1
        ORDER BY avg_score DESC
    ''')

    print("\n  Analysis Type → Avg Quality")
    print("  " + "-"*60)
    for intent_json, avg_score, count in cursor.fetchall():
        import json
        intent = json.loads(intent_json)
        analysis_type = intent.get('analysis_type', 'unknown')
        print(f"  {analysis_type:40} → {avg_score:.2f} ({count} answers)")

    # Common weaknesses
    print("\n\n⚠️  COMMON WEAKNESSES:")
    cursor.execute('''
        SELECT weaknesses FROM quality_ratings
        WHERE weaknesses IS NOT NULL AND weaknesses != '[]'
    ''')

    all_weaknesses = []
    for (weaknesses_json,) in cursor.fetchall():
        import json
        weaknesses = json.loads(weaknesses_json)
        all_weaknesses.extend(weaknesses)

    if all_weaknesses:
        from collections import Counter
        weakness_counts = Counter(all_weaknesses)
        print("\n  Top 5 weaknesses:")
        for weakness, count in weakness_counts.most_common(5):
            print(f"    • {weakness} ({count} occurrences)")

    print("\n\n💡 INSIGHTS FOR IMPROVEMENT:")
    print("  → Focus training on low-quality analysis types")
    print("  → Enhance modules with poor quality scores")
    print("  → Address common weaknesses in fine-tuning")
    print("  → Create test cases for failure modes")


def main():
    """Run all examples."""
    print("\n")
    print("█" * 80)
    print("█" + " " * 78 + "█")
    print("█" + "GeoBotv1 - Answer Quality Database & Training Pipeline".center(78) + "█")
    print("█" + "Continuous Improvement Through Feedback".center(78) + "█")
    print("█" + " " * 78 + "█")
    print("█" * 80)

    # Run examples
    db = example_1_automatic_logging()
    db = example_2_manual_quality_rating()
    good_answers, bad_answers = example_3_good_vs_bad_datasets()
    chat_dataset = example_4_export_training_datasets()
    pairs = example_5_preference_pairs_for_rlhf()
    example_6_continuous_improvement_workflow()
    example_7_quality_analysis()

    print("\n\n" + "="*80)
    print("EXAMPLES COMPLETE")
    print("="*80)
    print("\n🎓 Key Takeaways:")
    print("1. All analyst responses automatically logged with quality ratings")
    print("2. Good/bad answer separation enables targeted training")
    print("3. Multiple export formats support different training approaches")
    print("4. Preference pairs enable RLHF/DPO fine-tuning")
    print("5. Quality analysis identifies systematic improvement opportunities")
    print("6. Continuous cycle: Production → Feedback → Training → Deployment")
    print("\n✅ GeoBotv1 now has complete feedback loop for continuous improvement!")
    print("\n📁 Training datasets saved in: training_data/")
    print("   • geobot_chat_dataset.json (for chat model fine-tuning)")
    print("   • geobot_completion_dataset.json (for completion models)")
    print("   • geobot_rlhf_dataset.json (for reward models)")
    print("   • geobot_preference_pairs.json (for RLHF/DPO)")


if __name__ == "__main__":
    main()
