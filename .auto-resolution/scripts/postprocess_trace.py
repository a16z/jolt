#!/usr/bin/env python3
"""Convert monitor.rs metrics events to Perfetto counter tracks."""
import json
import sys

def main():
    if len(sys.argv) < 2:
        sys.exit("Usage: postprocess_trace.py <trace.json> [...]")
    
    for filepath in sys.argv[1:]:
        try:
            with open(filepath) as f:
                trace = json.load(f)
        except json.JSONDecodeError as e:
            print(f"{filepath}: ERROR - Invalid/incomplete JSON at line {e.lineno} column {e.colno}: {e.msg}", file=sys.stderr)
            print(f"{filepath}: Skipping file (likely incomplete trace from interrupted run)", file=sys.stderr)
            continue
        except FileNotFoundError:
            print(f"{filepath}: ERROR - File not found", file=sys.stderr)
            continue
        
        events = trace if isinstance(trace, list) else trace.get("traceEvents", [])
        counters = []
        
        for i, e in enumerate(events):
            if 'monitor.rs' in e.get('name', ''):
                for k, v in e.get('args', {}).items():
                    if k.startswith('counters.'):
                        counters.append({
                            "name": k.replace('counters.', ''),
                            "ph": "C",
                            "ts": e['ts'],
                            "pid": e['pid'],
                            "tid": e['tid'],
                            "args": {k.replace('counters.', ''): v}
                        })
                events[i] = None
        
        events = [e for e in events if e] + counters
        
        if isinstance(trace, list):
            trace = events
        else:
            trace["traceEvents"] = events
        
        with open(filepath, 'w') as f:
            json.dump(trace, f, separators=(',', ':'))
        
        print(f"{filepath}: {len(counters)} counters")

if __name__ == '__main__':
    main()
