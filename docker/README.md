## Build
Build prebuilt files:
```bash
./build.sh build spike
```

## Publish
Publish prebuilt files to the repository release:
```bash
gh repo set-default a16z/jolt
gh release list
gh release create spike-v1.1.1 --title "Spike v1.1.1" --notes "RISC-V ISA Simulator"
find ../prebuilt-files/ -iname 'spike-*.tar.gz' -exec gh release upload spike-1.1.1 {} --clobber \;
```