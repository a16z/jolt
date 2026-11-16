"""Capture what the fixture generator needs for the first transaction of a block.

    python3 fetch.py RPC_URL BLOCK_NUMBER > capture.json

The endpoint must serve the debug namespace (debug_executionWitness,
debug_getRawHeader, debug_getRawTransaction, debug_getRawReceipts).
"""
import json
import sys
import urllib.request


def rpc(url, method, *params):
    body = json.dumps({"jsonrpc": "2.0", "id": 1, "method": method, "params": params})
    request = urllib.request.Request(
        url, body.encode(), {"content-type": "application/json", "user-agent": "jolt"}
    )
    with urllib.request.urlopen(request, timeout=120) as response:
        reply = json.load(response)
    if "error" in reply:
        raise SystemExit(f"{method}: {reply['error']}")
    return reply["result"]


def main():
    url, number = sys.argv[1], int(sys.argv[2])
    block = rpc(url, "eth_getBlockByNumber", hex(number), False)
    witness = rpc(url, "debug_executionWitness", hex(number))
    capture = {
        "header": rpc(url, "debug_getRawHeader", hex(number)),
        "parent_header": rpc(url, "debug_getRawHeader", hex(number - 1)),
        "transaction": rpc(url, "debug_getRawTransaction", block["transactions"][0]),
        "receipt": rpc(url, "debug_getRawReceipts", hex(number))[0],
        "state": witness["state"],
        "codes": witness["codes"],
    }
    json.dump(capture, sys.stdout)


if __name__ == "__main__":
    main()
