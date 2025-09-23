"""Post-process chrome format trace JSON to convert monitor events (from monitor.rs) to counter tracks."""
import json
import sys
import argparse

def add_counter_tracks(trace_file):
    with open(trace_file) as f:
        trace = json.load(f)
    
    # Note: This line is for reference but not used in the main logic
    
    trace_events = trace if isinstance(trace, list) else trace.get("traceEvents", [])
    
    counter_events = []
    metrics_to_remove = []
    num_metric_samples = 0
    num_counters_created = 0
    
    for i, event in enumerate(trace_events):
        event_name = event.get('name', '')
        if event_name.startswith('event ') and 'monitor.rs' in event_name:
            num_metric_samples += 1
            ts = event.get('ts', 0)
            pid = event.get('pid', 1)
            tid = event.get('tid', 0)
            args = event.get('args', {})
            
            for key, value in args.items():
                if key.startswith('metrics.'):
                    counter_name = key.replace('metrics.', '')
                    
                    counter_event = {
                        "name": counter_name,
                        "ph": "C",  # Counter phase
                        "ts": ts,
                        "pid": pid,
                        "tid": tid,
                        "args": {counter_name: value}
                    }
                    counter_events.append(counter_event)
                    num_counters_created += 1
            
            metrics_to_remove.append(i)
    
    # Remove in reverse order to maintain indices
    for i in reversed(metrics_to_remove):
        trace_events.pop(i)
    
    trace_events.extend(counter_events)
    
    # Print conversion statistics
    print(f"Conversion statistics:")
    print(f"  - Monitor events processed: {num_metric_samples}")
    print(f"  - Counter events created: {num_counters_created}")
    
    if isinstance(trace, list):
        return trace_events
    else:
        trace["traceEvents"] = trace_events
        return trace


def main():
    parser = argparse.ArgumentParser(description='Post-process chrome format trace JSON to convert monitor events to counter tracks.')
    parser.add_argument('input_file', help='Input trace JSON file')
    parser.add_argument('output_file', nargs='?', help='Output trace JSON file (default: overwrites input file)')
    
    args = parser.parse_args()
    
    input_file = args.input_file
    output_file = args.output_file or input_file
    
    try:
        print(f"Processing {input_file}...")
        processed_trace = add_counter_tracks(input_file)
        
        # Write to output file
        with open(output_file, 'w') as f:
            json.dump(processed_trace, f, separators=(',', ':'))
        
        print(f"Processed trace saved to {output_file}")
        
    except FileNotFoundError:
        print(f"Error: Input file '{input_file}' not found", file=sys.stderr)
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON in input file: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error processing trace: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == '__main__':
    main()
