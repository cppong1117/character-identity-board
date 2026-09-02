#!/usr/bin/env python3
"""V0.2 Master Orchestrator — runs all phases in sequence.

Usage:
    python scripts/run_v02_pipeline.py [--skip-reembed] [--skip-quality] [--skip-cluster] [--skip-benchmark]
"""
import subprocess, sys, os, time, json

BASE = os.path.expanduser('~/character-identity-board')
REPORT_DIR = os.path.expanduser('~/character-identity-board/reports/v02_accuracy_recovery')

def run_phase(name, script, extra_args=None):
    """Run a phase script and report result."""
    print(f"\n{'='*60}")
    print(f"  PHASE: {name}")
    print(f"{'='*60}")
    
    cmd = [sys.executable, os.path.join(BASE, script)]
    if extra_args:
        cmd.extend(extra_args)
    
    t0 = time.time()
    result = subprocess.run(cmd, capture_output=False, timeout=7200)
    elapsed = time.time() - t0
    
    if result.returncode == 0:
        print(f"\n  ✅ {name} PASSED ({elapsed:.1f}s)")
    else:
        print(f"\n  ❌ {name} FAILED (exit code {result.returncode})")
    
    return result.returncode == 0


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--skip-reembed', action='store_true')
    parser.add_argument('--skip-quality', action='store_true')
    parser.add_argument('--skip-cluster', action='store_true')
    parser.add_argument('--skip-benchmark', action='store_true')
    args = parser.parse_args()
    
    t0 = time.time()
    results = {}
    
    # Phase 3: Re-embed (if not already done)
    if not args.skip_reembed:
        # Check if re-embed is needed
        import sqlite3
        conn = sqlite3.connect(os.path.expanduser('~/character-identity-board-data/cib.sqlite3'))
        cur = conn.cursor()
        cur.execute("SELECT COUNT(*) FROM face_observations WHERE excluded=0 AND embedding IS NOT NULL")
        total = cur.fetchone()[0]
        cur.execute("SELECT COUNT(DISTINCT SUBSTR(embedding, 1, 64)) FROM face_observations WHERE excluded=0 AND embedding IS NOT NULL")
        unique = cur.fetchone()[0]
        dup_rate = (total - unique) / total * 100 if total > 0 else 0
        conn.close()
        
        if dup_rate > 50:
            print(f"Re-embed needed: {dup_rate:.1f}% duplication")
            results['reembed'] = run_phase("Re-embed all observations", "scripts/reembed_all.py")
        else:
            print(f"Re-embed already done: {dup_rate:.1f}% duplication (OK)")
            results['reembed'] = True
    else:
        results['reembed'] = True
    
    # Phase 4-7: Quality Gate V2
    if not args.skip_quality and results.get('reembed'):
        results['quality'] = run_phase("Quality Gate V2", "scripts/quality_gate_v2.py")
    
    # Phase 8-10: Clustering V2
    if not args.skip_cluster and results.get('quality'):
        results['cluster'] = run_phase("Clustering V2", "scripts/clustering_v2.py")
    
    # Phase 11-14: Benchmark
    if not args.skip_benchmark and results.get('cluster'):
        results['benchmark'] = run_phase("Full Benchmark", "scripts/benchmark_full.py")
    
    elapsed = time.time() - t0
    
    # Final summary
    print(f"\n{'='*60}")
    print(f"  V0.2 PIPELINE COMPLETE ({elapsed/60:.1f} min)")
    print(f"{'='*60}")
    
    all_passed = all(results.values())
    if all_passed:
        print("  🎉 ALL PHASES PASSED")
    else:
        print("  ⚠️ SOME PHASES FAILED:")
        for phase, passed in results.items():
            if not passed:
                print(f"    - {phase}: FAILED")
    
    # Save summary
    os.makedirs(REPORT_DIR, exist_ok=True)
    with open(os.path.join(REPORT_DIR, 'v02_pipeline_summary.json'), 'w') as f:
        json.dump({
            'elapsed_seconds': round(elapsed, 1),
            'phases': results,
            'all_passed': all_passed,
        }, f, indent=2)
    
    print(f"\n  Summary: {REPORT_DIR}/v02_pipeline_summary.json")


if __name__ == '__main__':
    main()
